"""
SLAM前端 (MASt3R-SLAM §3)
===========================
模拟MASt3R的点图提取与匹配管线:

1. 点图提取 (Pointmap Extraction):
   - 模拟MASt3R从图像对生成pointmap
   - 使用合成深度作为输入
   
2. 点图匹配 (Pointmap Matching):
   - 3D-3D对应关系建立
   - 使用RANSAC+ICP计算相对位姿
   - 仿MASt3R-SLAM的PnP+ICP混合方法

3. 关键帧选择:
   - 基于运动距离和视角变化
   - 保证足够的共视区域
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from ..core.camera import PinholeCamera, look_at, random_so3, so3_log

def generate_synthetic_pointmaps(
    n_frames: int = 20,
    radius: float = 6.0,
    noise_std: float = 0.03
) -> List[Dict]:
    """
    生成合成点图序列 (模拟MASt3R输出)
    
    每帧包含:
    - pointmap: [H, W, 3] 相机坐标系下的3D点
    - confidence: [H, W] 置信度
    - pose_gt: 真实位姿 (用于评估)
    
    MASt3R输出: 对于每对图像(I1,I2)，输出两个点图
    在相机1坐标系下表示的点图1，以及在相机1坐标系下表示的点图2
    """
    H, W = 120, 160  # 使用较低分辨率加速
    fx, fy = 200.0, 200.0
    cx, cy = W/2, H/2
    
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    
    # 生成场景点云 (随机分布在空间中)
    np.random.seed(42)
    scene_pts = np.random.uniform(-5, 5, (2000, 3)).astype(np.float32)
    scene_pts[:, 1] += 1.0  # 抬高
    
    frames = []
    
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        eye = np.array([radius*np.cos(angle), 1.5, radius*np.sin(angle)], dtype=np.float32)
        R_gt, t_gt = look_at(eye, np.array([0.,1.,0.]), np.array([0.,1.,0.]))
        
        # 投影场景点到相机
        pts_cam = (R_gt @ scene_pts.T + t_gt).T
        z = pts_cam[:, 2]
        valid = (z > 0.1) & (z < 20)
        pts_cam = pts_cam[valid]
        
        # 生成pointmap (带噪声)
        u = (fx * pts_cam[:,0] / pts_cam[:,2] + cx).astype(int)
        v = (fy * pts_cam[:,1] / pts_cam[:,2] + cy).astype(int)
        
        px_valid = (u>=0) & (u<W) & (v>=0) & (v<H)
        u, v = u[px_valid], v[px_valid]
        pts_cam = pts_cam[px_valid]
        
        pointmap = np.zeros((H, W, 3), dtype=np.float32)
        confidence = np.zeros((H, W), dtype=np.float32)
        
        # 对接近的像素保留最近的 (Z-buffer)
        for k in range(len(u)):
            if confidence[v[k], u[k]] == 0 or pts_cam[k,2] < pointmap[v[k], u[k], 2]:
                pointmap[v[k], u[k]] = pts_cam[k] + np.random.randn(3)*noise_std
                confidence[v[k], u[k]] = 1.0
        
        frames.append({
            'pointmap': pointmap,
            'confidence': confidence,
            'R_gt': R_gt,
            't_gt': t_gt,
            'K': K,
            'idx': i
        })
    
    return frames


def match_pointmaps(
    pm1: np.ndarray, conf1: np.ndarray,
    pm2: np.ndarray, conf2: np.ndarray,
    K: np.ndarray,
    max_iters: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    点图匹配: 从两个点图中恢复相对位姿
    
    仿MASt3R-SLAM的匹配流程:
    1. 使用置信度加权采样匹配点
    2. 3D-3D对应 + RANSAC
    3. 使用Umeyama算法求解Sim(3)/SE(3)
    
    Args:
        pm1, pm2: [H,W,3] 点图 (均在相机1坐标系)
        conf1, conf2: [H,W] 置信度
        K: [3,3] 内参
        
    Returns:
        R_rel: 相对旋转
        t_rel: 相对平移
        inlier_ratio: 内点比例
    """
    H, W = pm1.shape[:2]
    
    # 采样高置信度点
    mask = (conf1 > 0.5) & (conf2 > 0.5)
    idx = np.where(mask)
    
    if len(idx[0]) < 10:
        return np.eye(3,dtype=np.float32), np.zeros((3,1),dtype=np.float32), 0.0
    
    # 下采样到最多500个匹配
    if len(idx[0]) > 500:
        sub = np.random.choice(len(idx[0]), 500, replace=False)
        idx = (idx[0][sub], idx[1][sub])
    
    pts1 = pm1[idx]  # [M, 3]
    pts2 = pm2[idx]  # [M, 3]
    
    # 移除无效深度点
    valid1 = pts1[:, 2] > 0.01
    valid2 = pts2[:, 2] > 0.01
    valid = valid1 & valid2
    
    if valid.sum() < 5:
        return np.eye(3,dtype=np.float32), np.zeros((3,1),dtype=np.float32), 0.0
    
    pts1 = pts1[valid]
    pts2 = pts2[valid]
    
    # RANSAC + Umeyama求解刚性变换
    # 简化: 直接用SVD求解 (论文中会用RANSAC)
    best_R = np.eye(3, dtype=np.float32)
    best_t = np.zeros((3,1), dtype=np.float32)
    best_inliers = 0
    threshold = 0.2  # 内点阈值
    
    n_ransac = min(50, len(pts1)//3)
    for _ in range(n_ransac):
        # 随机采样3个对应点
        sample = np.random.choice(len(pts1), min(3, len(pts1)), replace=False)
        R_tmp, t_tmp = solve_rigid_svd(pts1[sample], pts2[sample])
        
        # 计算内点数
        diff = pts2 - (R_tmp @ pts1.T + t_tmp).T
        errors = np.sqrt(np.sum(diff**2, axis=1))
        inliers = (errors < threshold).sum()
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_R = R_tmp
            best_t = t_tmp
    
    # 使用所有内点重新估计
    if best_inliers >= 3:
        inlier_mask = np.sqrt(np.sum(
            (pts2 - (best_R @ pts1.T + best_t).T)**2, axis=1
        )) < threshold
        if inlier_mask.sum() >= 3:
            best_R, best_t = solve_rigid_svd(pts1[inlier_mask], pts2[inlier_mask])
    
    inlier_ratio = best_inliers / len(pts1) if len(pts1) > 0 else 0
    
    return best_R, best_t, inlier_ratio


def solve_rigid_svd(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Umeyama算法: 用SVD求解最优刚性变换
    
    min Σ ||B_i - (R A_i + t)||^2
    
    Returns:
        R: 3x3 rotation
        t: 3x1 translation
    """
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    H = A_centered.T @ B_centered
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    # 确保R是旋转矩阵 (det=1)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    
    t = centroid_B.reshape(3,1) - R @ centroid_A.reshape(3,1)
    
    return R.astype(np.float32), t.astype(np.float32)


class SLAMFrontend:
    """
    SLAM前端: 管理点图提取、匹配和关键帧选择
    
    仿MASt3R-SLAM的设计:
    - 对每对新帧进行点图匹配
    - 维护关键帧集合
    - 支持回环检测 (基于视觉词袋的简化版)
    """
    
    def __init__(self, max_keyframes: int = 50):
        self.keyframes: List[Dict] = []
        self.max_kf = max_keyframes
        self.tracking_lost = False
    
    def process_frame(self, frame: Dict) -> bool:
        """
        处理新帧，决定是否添加为关键帧
        
        Returns:
            True if new keyframe added
        """
        if len(self.keyframes) == 0:
            self.keyframes.append(frame)
            return True
        
        last_kf = self.keyframes[-1]
        
        # 模拟匹配 (真实系统会调用MASt3R)
        R_rel, t_rel, inlier = match_pointmaps(
            last_kf['pointmap'], last_kf['confidence'],
            frame['pointmap'], frame['confidence'],
            frame['K']
        )
        
        # 关键帧选择条件:
        # 1. 运动足够大 (>0.5m或>10度)
        # 2. 内点率足够高 (>0.3)
        translation = np.linalg.norm(t_rel)
        rotation = np.linalg.norm(so3_log(R_rel))
        
        if (translation > 0.3 or rotation > 0.08) and inlier > 0.3:
            if len(self.keyframes) < self.max_kf:
                self.keyframes.append(frame)
                self.tracking_lost = False
                return True
        
        self.tracking_lost = (inlier < 0.1)
        return False
    
    def get_loop_candidates(self, current_idx: int) -> List[int]:
        """检测回环候选帧 (简化: 返回时间上远离的帧)"""
        candidates = []
        for i, kf in enumerate(self.keyframes):
            if abs(current_idx - i) > 10:
                candidates.append(i)
        return candidates[:5]  # 最多返回5个候选