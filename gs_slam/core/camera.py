"""
相机模型与投影 (基于MASt3R-SLAM论文第3节)

仿MASt3R-SLAM的相机模型: 不假设固定参数模型, 仅假设唯一光心。
支持:
- 针孔相机投影
- LookAt外参构建
- 点图与深度图互转 (MASt3R的pointmap概念)
"""

import numpy as np
from typing import Tuple, Optional

class PinholeCamera:
    """
    针孔相机模型
    
    与MASt3R-SLAM兼容的相机表示:
    - 内参K: 支持变焦, 但假设主点恒定
    - 外参[R|t]: 世界到相机变换
    """
    
    def __init__(self, 
                 fx: float = 500.0, fy: float = 500.0,
                 cx: float = 320.0, cy: float = 240.0,
                 width: int = 640, height: int = 480):
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy
        self.W = width; self.H = height
        
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.R = np.eye(3, dtype=np.float32)
        self.t = np.zeros((3, 1), dtype=np.float32)
    
    def set_pose(self, R: np.ndarray, t: np.ndarray):
        """设置外参 (世界->相机)"""
        self.R = R.astype(np.float32)
        self.t = t.reshape(3, 1).astype(np.float32)
    
    def world_to_camera(self, pts_3d: np.ndarray) -> np.ndarray:
        """世界坐标 -> 相机坐标"""
        return (self.R @ pts_3d.T + self.t).T
    
    def camera_to_pixel(self, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """相机坐标 -> 像素坐标 + 深度"""
        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
        valid = Z > 1e-6
        u = np.zeros(len(pts_cam))
        v = np.zeros(len(pts_cam))
        if valid.sum() > 0:
            u[valid] = self.fx * X[valid] / Z[valid] + self.cx
            v[valid] = self.fy * Y[valid] / Z[valid] + self.cy
        return np.stack([u, v], axis=1), Z
    
    def pixel_to_ray(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """像素坐标 -> 归一化相机射线"""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=1)
    
    def get_w2c_matrix(self) -> np.ndarray:
        """3x4 世界到相机矩阵 [R|t]"""
        return np.hstack([self.R, self.t])


def look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    构建LookAt外参矩阵
    
    Args:
        eye: 相机在世界坐标系中的位置
        center: 视线目标点
        up: 上方向
        
    Returns:
        R: 3x3旋转矩阵 (世界->相机)
        t: 3x1平移向量 (世界->相机)
    """
    eye = np.asarray(eye, dtype=np.float32).flatten()
    center = np.asarray(center, dtype=np.float32).flatten()
    up = np.asarray(up, dtype=np.float32).flatten()
    
    f = center - eye
    f = f / (np.linalg.norm(f) + 1e-10)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-10)
    u = np.cross(r, f)
    
    R = np.vstack([r, u, f])  # 相机Z轴指向center方向
    t = -R @ eye
    return R.astype(np.float32), t.reshape(3, 1).astype(np.float32)


def pointmap_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    从深度图生成点图 (MASt3R的pointmap概念)
    
    pointmap: 每像素的3D相机坐标 [H, W, 3]
    这是MASt3R模型输出的核心表示
    
    Args:
        depth: [H, W] 深度图
        K: [3, 3] 内参矩阵
    
    Returns:
        pointmap: [H, W, 3] 点图
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def depth_from_pointmap(pointmap: np.ndarray) -> np.ndarray:
    """从点图提取深度"""
    return pointmap[..., 2]


def random_so3(scale: float = 1.0) -> np.ndarray:
    """在SO(3)上采样随机旋转"""
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    angle = scale * np.random.randn()
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]
    ], dtype=np.float32)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3)对数映射 -> 旋转向量"""
    cos_t = (np.trace(R) - 1) / 2
    cos_t = np.clip(cos_t, -1+1e-7, 1-1e-7)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return np.zeros(3)
    return theta / (2*np.sin(theta)) * np.array([
        R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]
    ], dtype=np.float32)