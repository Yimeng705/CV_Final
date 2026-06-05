"""
增量建图模块 (OpenMonoGS-SLAM §3.2)
====================================
基于3DGS的稠密建图，支持:
1. 关键帧触发高斯生成
2. 语义特征关联 (开放集语义)
3. 冗余高斯剪枝
"""

import numpy as np
from ..core.gaussian_model import GaussianCloud
from ..core.camera import PinholeCamera

class DenseMapper:
    """增量稠密建图器"""
    
    def __init__(self, capacity: int = 10000):
        self.map = GaussianCloud(capacity)
        self.kf_count = 0
    
    def add_keyframe(self, pos: np.ndarray, rgb: np.ndarray,
                     sem: np.ndarray = None):
        self.map.add(pos, rgb, sem)
        self.kf_count += 1
    
    def add_pointcloud(self, pts_world: np.ndarray, colors: np.ndarray):
        self.map.add(pts_world, colors)
    
    def prune(self):
        self.map.prune(0.05)
    
    def get_map(self):
        return self.map.pack()
    
    def size(self):
        return len(self.map)
    
    def assign_semantic_regions(self):
        """
        按空间位置分配语义特征 (模拟OpenMonoGS-SLAM的语义)
        在实际系统中由SAM+CLIP生成
        """
        xyz = self.map.xyz[:len(self.map)]
        N = len(self.map)
        
        # 三个语义区域
        sem = np.zeros((N, 64), dtype=np.float32)
        # Region A: 中心区域 (模拟"家具"类别)
        mask_a = np.linalg.norm(xyz, axis=1) < 2.5
        sem[mask_a, :21] = 1.0
        # Region B: X正向偏移 (模拟"墙壁")
        mask_b = xyz[:, 0] > 2.5
        sem[mask_b, 21:42] = 1.0
        # Region C: Y负向 (模拟"地面")
        mask_c = xyz[:, 1] < -2.5
        sem[mask_c, 42:] = 1.0
        
        self.map.sem[:N] = sem
        return sem