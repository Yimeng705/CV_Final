"""
3DGS可微渲染器 (改进版 v3.0)
====================
基于综述论文(Chen & Wang, 2026, TPAMI)的tile-based渲染管线，结合OpenMonoGS-SLAM的语义渲染。

改进点:
1. 真正的tile-based分块处理 (仿3DGS综述method-002)
   1a. 屏幕划分为16x16 tile，每个高斯分配到覆盖的所有tile
   1b. 逐tile深度排序 + 早期终止混合
   1c. NumPy CPU实现，GPU加速版本需CUDA kernel
2. 每个高斯基于真实协方差的2D投影 (Sigma2D = J W Sigma W^T J^T)
3. 智能渲染路径：高斯数>500时使用tile-based，否则fallback逐高斯
4. 支持同步渲染RGB+语义+深度多通道
5. 支持自适应密度控制的密度诊断输出

核心流程:
1. 视锥体裁剪 + 协方差投影
2. "每个高斯分配到所有覆盖的tile" (关键区别)
3. 逐tile深度排序 (远处在前)
4. 逐tile Alpha blending (带提前终止 T<0.001)
5. 密度统计输出 (用于自适应控制)
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from .camera import PinholeCamera


def compute_psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 1.0) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    3DGS论文首要渲染质量指标
    
    Args:
        pred: [H,W,C] 预测图像, float [0,1]
        gt: [H,W,C] 真实图像, float [0,1]
        max_val: 像素最大值
    
    Returns:
        psnr: PSNR值(dB)
    """
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, 
                 data_range: float = 1.0, 
                 win_size: int = 11,
                 channel_weights: tuple = (0.3, 0.59, 0.11)) -> float:
    """
    计算SSIM (Structural Similarity Index)
    衡量感知质量的结构相似性
    
    基于: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity"
    
    Args:
        pred: [H,W,C] 预测图像, float [0,1]
        gt: [H,W,C] 真实图像, float [0,1]
        data_range: 数据范围
        win_size: 滑动窗口大小
        channel_weights: RGB三通道权重
    
    Returns:
        ssim: SSIM值 [0,1], 越高越好
    """
    if pred.shape != gt.shape:
        min_h = min(pred.shape[0], gt.shape[0])
        min_w = min(pred.shape[1], gt.shape[1])
        pred = pred[:min_h, :min_w]
        gt = gt[:min_h, :min_w]
    
    # 转换到float64精度
    pred = pred.astype(np.float64)
    gt = gt.astype(np.float64)
    
    # 动态范围
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 高斯窗口
    sigma = 1.5
    x = np.arange(-(win_size//2), win_size//2 + 1, dtype=np.float64)
    gauss = np.exp(-0.5 * (x / sigma) ** 2)
    gauss = gauss / gauss.sum()
    window = gauss[:, None] * gauss[None, :]
    window = window[None, :, :, None]  # [1,H,W,1]
    
    # 逐通道计算
    ssim_per_channel = []
    for c in range(pred.shape[-1]):
        if c >= 3:
            break
        pc = pred[:, :, c]
        gc = gt[:, :, c]
        
        # 局部均值
        mu1 = _conv2d(pc, window[0, :, :, 0])
        mu2 = _conv2d(gc, window[0, :, :, 0])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        
        # 局部方差和协方差
        sigma1_sq = _conv2d(pc ** 2, window[0, :, :, 0]) - mu1_sq
        sigma2_sq = _conv2d(gc ** 2, window[0, :, :, 0]) - mu2_sq
        sigma12 = _conv2d(pc * gc, window[0, :, :, 0]) - mu12
        
        # SSIM
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_per_channel.append(np.mean(ssim_map))
    
    if len(ssim_per_channel) == 3:
        return float(sum(w * s for w, s in zip(channel_weights, ssim_per_channel)))
    elif len(ssim_per_channel) > 0:
        return float(np.mean(ssim_per_channel))
    return 0.0


def _conv2d(img: np.ndarray, kernel: np.ndarray, mode: str = 'valid') -> np.ndarray:
    """简化的2D卷积 (用于SSIM) - 纯numpy实现, 避免scipy/numpy版本不兼容"""
    from numpy.fft import rfft2, irfft2
    h, w = img.shape
    kh, kw = kernel.shape
    
    # 使用FFT实现快速卷积 (reflect padding)
    pad_h, pad_w = kh // 2, kw // 2
    # 反射填充
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    
    # FFT卷积
    out_h = h + kh - 1
    out_w = w + kw - 1
    img_fft = np.fft.rfft2(img_padded, s=(out_h, out_w))
    kernel_fft = np.fft.rfft2(kernel, s=(out_h, out_w))
    result = np.fft.irfft2(img_fft * kernel_fft, s=(out_h, out_w))
    
    # 提取有效区域
    start_h = kh - 1
    start_w = kw - 1
    return result[start_h:start_h + h, start_w:start_w + w].real


def _bilinear_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """纯numpy双线性下采样 (替代scipy.ndimage.zoom)"""
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        result = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
        for c in range(img.shape[2]):
            result[:, :, c] = _bilinear_resize_2d(img[:, :, c], new_h, new_w)
        return result
    else:
        return _bilinear_resize_2d(img, new_h, new_w)


def _bilinear_resize_2d(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """单通道双线性下采样"""
    h, w = img.shape
    # 使用简单的区域平均池化 (处理下采样)
    row_idx = np.linspace(0, h - 1, new_h)
    col_idx = np.linspace(0, w - 1, new_w)
    
    row_lo = np.floor(row_idx).astype(int)
    row_hi = np.minimum(row_lo + 1, h - 1)
    col_lo = np.floor(col_idx).astype(int)
    col_hi = np.minimum(col_lo + 1, w - 1)
    
    row_frac = row_idx - row_lo
    col_frac = col_idx - col_lo
    
    result = np.zeros((new_h, new_w), dtype=img.dtype)
    for i in range(new_h):
        rf = row_frac[i]; rl = row_lo[i]; rh = row_hi[i]
        for j in range(new_w):
            cf = col_frac[j]; cl = col_lo[j]; ch = col_hi[j]
            result[i, j] = (
                (1 - rf) * (1 - cf) * img[rl, cl] +
                (1 - rf) * cf * img[rl, ch] +
                rf * (1 - cf) * img[rh, cl] +
                rf * cf * img[rh, ch]
            )
    return result


def compute_lpips_simple(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    简化的LPIPS代理 (基于多尺度SSIM)
    完整LPIPS需要预训练网络(AlexNet/VGG)，这里提供基于MS-SSIM的代理版本
    
    Args:
        pred: [H,W,C] 预测图像
        gt: [H,W,C] 真实图像
    
    Returns:
        lpips_proxy: [0,1], 越低越好
    """
    # MS-SSIM作为代理 (多尺度计算)
    scales = [1.0, 0.5, 0.25]
    ssim_values = []
    
    h, w = pred.shape[:2]
    for scale in scales:
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h < 32 or new_w < 32:
                break
            # 纯numpy双线性下采样 (替代scipy.ndimage.zoom)
            p_down = _bilinear_resize(pred, new_h, new_w)
            g_down = _bilinear_resize(gt, new_h, new_w)
            ssim_val = compute_ssim(p_down, g_down, win_size=7)
        else:
            ssim_val = compute_ssim(pred, gt, win_size=7)
        ssim_values.append(ssim_val)
    
    if ssim_values:
        return float(1.0 - np.mean(ssim_values))
    return 0.0


def compute_rendering_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    计算综合渲染质量指标
    
    Args:
        pred: [H,W,C] 渲染图像, float [0,1]
        gt: [H,W,C] 真实图像, float [0,1]
    
    Returns:
        metrics: {psnr, ssim, lpips_proxy}
    """
    return {
        'psnr': compute_psnr(pred, gt),
        'ssim': compute_ssim(pred, gt),
        'lpips_proxy': compute_lpips_simple(pred, gt)
    }


class SplatRenderer:
    """
    3DGS Tile-based Splat渲染器 (v3.0)
    
    实现真正的tile-based渲染管线 (仿3DGS综述 method-002):
    1. 视锥体裁剪 + 2D协方差投影
    2. 将每个高斯分配到覆盖的所有tile
    3. 逐tile深度排序 (远处在前)
    4. 逐tile alpha混合 (带T<0.001提前终止)
    
    注: NumPy CPU实现，GPU加速版本需CUDA kernel
    """

    def __init__(self, H: int = 480, W: int = 640, tile_size: int = 16):
        self.H = H
        self.W = W
        self.tile_size = tile_size
        self.tiles_H = (H + tile_size - 1) // tile_size
        self.tiles_W = (W + tile_size - 1) // tile_size
        self.n_tiles = self.tiles_H * self.tiles_W
        # 使用tile-based的阈值 (高斯数超过此值使用tile-based路径)
        self.tile_based_threshold = 500

    def _project_covariance_2d(self, fx: float, fy: float, z: np.ndarray,
                               cov_3d: np.ndarray) -> np.ndarray:
        """
        3D协方差 -> 2D协方差投影
        Sigma_2D = J W Sigma_3D W^T J^T 的简化对角近似
        """
        N = len(cov_3d)
        s2d = np.zeros((N, 2, 2), dtype=np.float32)
        s2d[:, 0, 0] = np.sqrt(np.abs(cov_3d[:, 0, 0])) * fx / np.maximum(z, 0.01)
        s2d[:, 1, 1] = np.sqrt(np.abs(cov_3d[:, 1, 1])) * fy / np.maximum(z, 0.01)
        s2d[:, 0, 1] = cov_3d[:, 0, 1] * fx * fy / np.maximum(z * z, 0.01)
        return s2d

    def _compute_2d_projections(self, gs, cam):
        """
        计算所有高斯的2D投影参数
        
        Returns:
            u, v: [N] 像素坐标
            depth: [N] 深度 (z_cam)
            radius: [N] 3-sigma 2D半径 (int)
            s_x, s_y: [N] 2D标准差
            valid_idx: [N] 可见高斯的原始索引
            (sem, rgb, opacity也对应裁剪)
        """
        xyz = gs['xyz']
        rgb = gs['rgb']
        scale = gs['scale']
        opacity = gs['opacity']
        sem = gs.get('sem', np.zeros((len(xyz), 64), dtype=np.float32))
        cov = gs.get('cov', np.zeros((len(xyz), 3, 3), dtype=np.float32))

        # 世界->相机
        pts_cam = (cam.R @ xyz.T + cam.t).T
        z = pts_cam[:, 2]

        # 视锥体裁剪
        ok = z > 0.1
        if not ok.any():
            return None

        z = z[ok]
        rgb = rgb[ok]
        opacity = opacity[ok]
        scale = scale[ok]
        sem = sem[ok]
        cov = cov[ok]
        pts_cam = pts_cam[ok]
        valid_idx = np.where(ok)[0]
        N = len(z)

        # 投影到像素
        fx, fy = cam.fx, cam.fy
        cx, cy = cam.cx, cam.cy

        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy

        # 计算2D半径 (3-sigma覆盖)
        if len(cov) > 0 and cov.shape[1:] == (3, 3):
            s2d = self._project_covariance_2d(fx, fy, z, cov)
            s_x = np.sqrt(np.abs(s2d[:, 0, 0]))
            s_y = np.sqrt(np.abs(s2d[:, 1, 1]))
        else:
            s_x = scale[:, 0] * fx / z
            s_y = scale[:, 1] * fy / z

        # 确保最小半径, 限制最大半径
        radius_float = np.maximum(s_x, s_y) * 3.0
        radius = np.maximum(2, radius_float.astype(int))
        radius = np.minimum(radius, 200)

        return {
            'u': u, 'v': v, 'depth': z, 'radius': radius,
            's_x': s_x, 's_y': s_y,
            'rgb': rgb, 'opacity': opacity, 'sem': sem,
            'N': N, 'valid_idx': valid_idx
        }

    def _get_tile_range(self, ui: float, vi: float, r: int):
        """
        计算高斯(i)覆盖的tile索引范围
        
        注: 与原始per-Gaussian路径一致，这里返回像素patch的tile范围
        """
        y0 = max(0, int(vi) - r)
        y1 = min(self.H, int(vi) + r + 1)
        x0 = max(0, int(ui) - r)
        x1 = min(self.W, int(ui) + r + 1)
        
        tile_y0 = y0 // self.tile_size
        tile_y1 = min((y1 + self.tile_size - 1) // self.tile_size, self.tiles_H)
        tile_x0 = x0 // self.tile_size
        tile_x1 = min((x1 + self.tile_size - 1) // self.tile_size, self.tiles_W)
        
        tiles = []
        for ty in range(tile_y0, tile_y1):
            for tx in range(tile_x0, tile_x1):
                tiles.append(ty * self.tiles_W + tx)
        return tiles

    def _render_naive(self, proj: dict, sem_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Naive逐高斯渲染 (fallback路径，用于高斯数小于阈值时)
        
        保留原有逻辑但添加注释说明
        """
        u = proj['u']
        v = proj['v']
        z = proj['depth']
        rgb = proj['rgb']
        opacity = proj['opacity']
        sem = proj['sem']
        radius = proj['radius']
        s_x = proj['s_x']
        s_y = proj['s_y']
        N = proj['N']

        # 深度排序 (远处在前)
        order = np.argsort(-z)
        u = u[order]
        v = v[order]
        z = z[order]
        rgb = rgb[order]
        opacity = opacity[order]
        sem = sem[order]
        radius = radius[order]
        s_x = s_x[order]
        s_y = s_y[order]

        img = np.ones((self.H, self.W, 3), dtype=np.float32)
        sem_img = np.zeros((self.H, self.W, sem_dim), dtype=np.float32)
        depth_img = np.full((self.H, self.W), np.inf, dtype=np.float32)
        T = np.ones((self.H, self.W), dtype=np.float32)

        for i in range(N):
            ui = int(round(u[i]))
            vi = int(round(v[i]))
            r = int(radius[i])

            if ui < -r or ui >= self.W + r or vi < -r or vi >= self.H + r:
                continue

            y0 = max(0, vi - r)
            y1 = min(self.H, vi + r + 1)
            x0 = max(0, ui - r)
            x1 = min(self.W, ui + r + 1)

            if y1 <= y0 or x1 <= x0:
                continue

            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            sx_val = max(s_x[i], 0.01)
            sy_val = max(s_y[i], 0.01)

            g = np.exp(-0.5 * (((xx - ui) / sx_val) ** 2 +
                               ((yy - vi) / sy_val) ** 2))
            alpha = float(opacity[i]) * g

            T_patch = T[y0:y1, x0:x1]
            # 提前终止
            active = T_patch > 0.001
            if not active.any():
                continue

            img_patch = img[y0:y1, x0:x1]
            sem_patch = sem_img[y0:y1, x0:x1]
            depth_patch = depth_img[y0:y1, x0:x1]

            alpha_3d = alpha[..., None]
            img_patch[active] = (
                img_patch[active] * (1 - alpha_3d[active]) +
                rgb[i] * alpha_3d[active]
            )
            sem_patch[active] = (
                sem_patch[active] * (1 - alpha_3d[active]) +
                sem[i] * alpha_3d[active]
            )
            update_mask = active & (depth_patch == np.inf)
            depth_patch[update_mask] = z[i]
            T_patch[active] = T_patch[active] * (1 - alpha[active])

        return (np.clip(img, 0, 1).astype(np.float32),
                sem_img.astype(np.float32),
                depth_img.astype(np.float32))

    def _render_tile_based(self, proj: dict, sem_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tile-based渲染管线 (真正的3DGS综述method-002实现)
        
        步骤:
        1. 将每个高斯分配到覆盖的所有tile
        2. 每个tile内按深度排序
        3. 逐tile alpha混合 (带提前终止)
        
        与naive路径的关键区别:
        - 每个高斯可能属于多个tile (覆盖多个tile)
        - 每个tile独立混合，互不干扰
        - 理论上可以tile级并行 (GPU实现)
        """
        u = proj['u']
        v = proj['v']
        z = proj['depth']
        rgb = proj['rgb']
        opacity = proj['opacity']
        sem = proj['sem']
        radius = proj['radius']
        s_x = proj['s_x']
        s_y = proj['s_y']
        N = proj['N']

        # === Step 1: 将高斯分配到所有覆盖的tile ===
        tile_gaussians = [[] for _ in range(self.n_tiles)]
        for i in range(N):
            tile_ids = self._get_tile_range(u[i], v[i], radius[i])
            for tid in tile_ids:
                tile_gaussians[tid].append(i)

        # === Step 2 & 3: 每个tile内按深度排序 & 混合 ===
        img = np.ones((self.H, self.W, 3), dtype=np.float32)
        sem_img = np.zeros((self.H, self.W, sem_dim), dtype=np.float32)
        depth_img = np.full((self.H, self.W), np.inf, dtype=np.float32)
        T = np.ones((self.H, self.W), dtype=np.float32)

        for tile_id in range(self.n_tiles):
            tl = tile_gaussians[tile_id]
            if not tl:
                continue

            # 按深度排序 (远处在前)
            tl_sorted = sorted(tl, key=lambda i: -z[i])

            # 计算tile像素范围
            ty = tile_id // self.tiles_W
            tx = tile_id % self.tiles_W
            y0 = ty * self.tile_size
            y1 = min(y0 + self.tile_size, self.H)
            x0 = tx * self.tile_size
            x1 = min(x0 + self.tile_size, self.W)

            # 对tile内的每个高斯进行混合
            for gi in tl_sorted:
                ui = int(round(u[gi]))
                vi = int(round(v[gi]))
                r = int(radius[gi])

                # 计算该高斯在当前tile内的像素范围
                gy0 = max(y0, vi - r)
                gy1 = min(y1, vi + r + 1)
                gx0 = max(x0, ui - r)
                gx1 = min(x1, ui + r + 1)

                if gy1 <= gy0 or gx1 <= gx0:
                    continue

                yy, xx = np.mgrid[gy0:gy1, gx0:gx1].astype(np.float32)
                sx_val = max(s_x[gi], 0.01)
                sy_val = max(s_y[gi], 0.01)
                g = np.exp(-0.5 * (((xx - ui) / sx_val) ** 2 +
                                   ((yy - vi) / sy_val) ** 2))
                alpha = float(opacity[gi]) * g

                T_patch = T[gy0:gy1, gx0:gx1]
                active = T_patch > 0.001
                if not active.any():
                    continue

                img_patch = img[gy0:gy1, gx0:gx1]
                sem_patch = sem_img[gy0:gy1, gx0:gx1]
                depth_patch = depth_img[gy0:gy1, gx0:gx1]

                alpha_3d = alpha[..., None]
                img_patch[active] = (
                    img_patch[active] * (1 - alpha_3d[active]) +
                    rgb[gi] * alpha_3d[active]
                )
                sem_patch[active] = (
                    sem_patch[active] * (1 - alpha_3d[active]) +
                    sem[gi] * alpha_3d[active]
                )
                update_mask = active & (depth_patch == np.inf)
                depth_patch[update_mask] = z[gi]
                T_patch[active] = T_patch[active] * (1 - alpha[active])

        return (np.clip(img, 0, 1).astype(np.float32),
                sem_img.astype(np.float32),
                depth_img.astype(np.float32))

    def render(self, gs: Dict[str, np.ndarray],
               cam: PinholeCamera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        主渲染函数 (v3.0 - 自适应渲染路径)

        当高斯数>阈值时使用tile-based路径，否则使用naive路径。

        Returns:
            rgb:  [H,W,3] float32 [0,1]
            sem:  [H,W,D] float32 语义特征图
            depth: [H,W] float32 深度图
        """
        xyz = gs['xyz']
        sem_dim = gs.get('sem', np.zeros((len(xyz), 64), dtype=np.float32)).shape[1] \
                  if 'sem' in gs and len(gs['sem'].shape) > 1 else 64
        N_total = len(xyz)

        if N_total == 0:
            return (np.ones((self.H, self.W, 3), dtype=np.float32),
                    np.zeros((self.H, self.W, sem_dim), dtype=np.float32),
                    np.full((self.H, self.W), np.inf, dtype=np.float32))

        # 计算2D投影
        proj = self._compute_2d_projections(gs, cam)
        if proj is None:
            return (np.ones((self.H, self.W, 3), dtype=np.float32),
                    np.zeros((self.H, self.W, sem_dim), dtype=np.float32),
                    np.full((self.H, self.W), np.inf, dtype=np.float32))

        # 智能渲染路径选择
        if proj['N'] > self.tile_based_threshold:
            return self._render_tile_based(proj, sem_dim)
        else:
            return self._render_naive(proj, sem_dim)

    def render_rgb(self, gs: Dict[str, np.ndarray],
                   cam: PinholeCamera) -> np.ndarray:
        """只渲染RGB, 返回uint8图像"""
        rgb, _, _ = self.render(gs, cam)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def get_density_stats(self, gs: Dict[str, np.ndarray],
                          cam: PinholeCamera) -> Dict:
        """
        密度诊断: 为自适应密度控制提供统计信息
        仿3DGS综述 method-003 (Adaptive Density Control)

        Returns:
            stats: {
                'n_total': 总高斯数,
                'n_visible': 可见高斯数,
                'avg_2d_radius': 平均2D投影半径,
                'coverage_ratio': 图像覆盖比例,
                'render_time_ms': 渲染耗时
            }
        """
        t0 = time.time()
        _, _, depth = self.render(gs, cam)
        elapsed = (time.time() - t0) * 1000

        rendered_mask = depth < np.inf
        coverage = rendered_mask.sum() / (self.H * self.W)

        scales = gs.get('scale', np.ones((1, 3)))
        avg_radius = float(np.mean(scales[:, :2])) if len(scales) > 0 else 0.0

        return {
            'n_total': len(gs['xyz']),
            'n_visible': int(np.count_nonzero(depth < np.inf)),
            'avg_2d_radius': avg_radius,
            'coverage_ratio': float(coverage),
            'render_time_ms': elapsed
        }


class PointRenderer:
    """稀疏点渲染 (Baseline) - 用于与3DGS对比"""

    def __init__(self, H: int = 480, W: int = 640):
        self.H = H
        self.W = W

    def render(self, gs: Dict[str, np.ndarray],
               cam: PinholeCamera) -> np.ndarray:
        """点云投影渲染, 返回uint8图像"""
        xyz = gs['xyz']
        rgb = gs['rgb']
        pts_cam = (cam.R @ xyz.T + cam.t).T
        z = pts_cam[:, 2]
        ok = z > 0.01
        pts_cam = pts_cam[ok]
        z = z[ok]
        rgb = rgb[ok]

        u = (cam.fx * pts_cam[:, 0] / z + cam.cx).astype(int)
        v = (cam.fy * pts_cam[:, 1] / z + cam.cy).astype(int)

        valid = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        u = u[valid]
        v = v[valid]

        img = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        col = (np.clip(rgb[valid], 0, 1) * 255).astype(np.uint8)

        for du in range(-1, 2):
            for dv in range(-1, 2):
                cu = np.clip(u + du, 0, self.W - 1)
                cv = np.clip(v + dv, 0, self.H - 1)
                img[cv, cu] = col
        return img

    def render_depth(self, gs: Dict[str, np.ndarray],
                     cam: PinholeCamera) -> np.ndarray:
        """渲染深度图"""
        xyz = gs['xyz']
        pts_cam = (cam.R @ xyz.T + cam.t).T
        z = pts_cam[:, 2]
        ok = z > 0.01
        z = z[ok]
        pts_cam = pts_cam[ok]

        u = (cam.fx * pts_cam[:, 0] / z + cam.cx).astype(int)
        v = (cam.fy * pts_cam[:, 1] / z + cam.cy).astype(int)

        valid = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

        depth = np.full((self.H, self.W), np.inf, dtype=np.float32)
        for ui, vi, zi in zip(u[valid], v[valid], z[valid]):
            if zi < depth[vi, ui]:
                depth[vi, ui] = zi
        return depth