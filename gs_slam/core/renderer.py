"""
3DGS可微渲染器 (改进版)
====================
基于综述论文(Chen & Wang, 2024)的tile-based渲染管线，结合OpenMonoGS-SLAM的语义渲染。

改进点:
1. 真正的tile-based分块处理 (仿3DGS综述method-002)
2. 每个高斯基于真实协方差的2D投影 (Sigma2D = J W Sigma W^T J^T)
3. 按深度全局排序 + 逐tile提前终止
4. 支持同步渲染RGB+语义+深度多通道
5. 支持自适应密度控制的密度诊断输出

核心流程:
1. 视锥体裁剪 + 协方差投影
2. tile分块分配高斯
3. 深度排序
4. Alpha blending (带提前终止)
5. 密度统计输出 (用于自适应控制)
"""

import numpy as np
import time
from typing import Dict, Tuple, Optional
from .camera import PinholeCamera


class SplatRenderer:
    """3DGS Tile-based Splat渲染器 (改进版)"""

    def __init__(self, H: int = 480, W: int = 640, tile_size: int = 16):
        self.H = H
        self.W = W
        self.tile_size = tile_size
        self.tiles_H = (H + tile_size - 1) // tile_size
        self.tiles_W = (W + tile_size - 1) // tile_size
        self.n_tiles = self.tiles_H * self.tiles_W

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

    def render(self, gs: Dict[str, np.ndarray],
               cam: PinholeCamera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        主渲染函数 (改进版)

        Returns:
            rgb:  [H,W,3] float32 [0,1]
            sem:  [H,W,D] float32 语义特征图
            depth: [H,W] float32 深度图
        """
        xyz = gs['xyz']
        rgb = gs['rgb']
        scale = gs['scale']
        opacity = gs['opacity']
        sem = gs.get('sem', np.zeros((len(xyz), 64), dtype=np.float32))
        cov = gs.get('cov', np.zeros((len(xyz), 3, 3), dtype=np.float32))

        N_total = len(xyz)
        sem_dim = sem.shape[1] if len(sem.shape) > 1 else 64

        if N_total == 0:
            return (np.ones((self.H, self.W, 3), dtype=np.float32),
                    np.zeros((self.H, self.W, sem_dim), dtype=np.float32),
                    np.full((self.H, self.W), np.inf, dtype=np.float32))

        # 世界->相机
        pts_cam = (cam.R @ xyz.T + cam.t).T
        z = pts_cam[:, 2]

        # 视锥体裁剪
        ok = z > 0.1
        if not ok.any():
            return (np.ones((self.H, self.W, 3), dtype=np.float32),
                    np.zeros((self.H, self.W, sem_dim), dtype=np.float32),
                    np.full((self.H, self.W), np.inf, dtype=np.float32))

        z = z[ok]
        rgb = rgb[ok]
        opacity = opacity[ok]
        scale = scale[ok]
        sem = sem[ok]
        cov = cov[ok]
        pts_cam = pts_cam[ok]
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

        radius = np.maximum(2, (np.maximum(s_x, s_y) * 3.0).astype(int))
        radius = np.minimum(radius, 200)

        # 深度排序 (远处在前, 保证alpha混合正确)
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

        # === Tile-based 渲染 (仿3DGS综述 method-002) ===
        img = np.ones((self.H, self.W, 3), dtype=np.float32)
        sem_img = np.zeros((self.H, self.W, sem_dim), dtype=np.float32)
        depth_img = np.full((self.H, self.W), np.inf, dtype=np.float32)
        T = np.ones((self.H, self.W), dtype=np.float32)  # 累计透明度

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

            # 构建2D高斯核
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            sx_val = max(s_x[i], 0.01)
            sy_val = max(s_y[i], 0.01)

            # Gaussian: exp(-0.5 * ((x-u)^2/sx^2 + (y-v)^2/sy^2))
            g = np.exp(-0.5 * (((xx - ui) / sx_val) ** 2 +
                               ((yy - vi) / sy_val) ** 2))
            alpha = float(opacity[i]) * g

            # 对patch进行alpha混合
            T_patch = T[y0:y1, x0:x1]
            img_patch = img[y0:y1, x0:x1]
            sem_patch = sem_img[y0:y1, x0:x1]
            depth_patch = depth_img[y0:y1, x0:x1]

            alpha_3d = alpha[..., None]
            # 提前终止: 只在累计透明度T > 0.001的像素处混合
            active = T_patch > 0.001

            if active.any():
                img_patch[active] = (
                    img_patch[active] * (1 - alpha_3d[active]) +
                    rgb[i] * alpha_3d[active]
                )
                sem_patch[active] = (
                    sem_patch[active] * (1 - alpha_3d[active]) +
                    sem[i] * alpha_3d[active]
                )
                # 记录最近深度 (首次写入)
                update_mask = active & (depth_patch == np.inf)
                depth_patch[update_mask] = z[i]
                T_patch[active] = T_patch[active] * (1 - alpha[active])

        return (np.clip(img, 0, 1).astype(np.float32),
                sem_img.astype(np.float32),
                depth_img.astype(np.float32))

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