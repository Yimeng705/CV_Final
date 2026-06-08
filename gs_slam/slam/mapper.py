"""
增量建图模块 (改进版)
====================
基于OpenMonoGS-SLAM + 3DGS综述的稠密建图

改进点 (我们提出的方法):
1. 语义感知的自适应高斯密度控制 (Semantic-Aware Adaptive Densification)
   - 在语义边界区域增加高斯密度以提升重建质量
   - 相比原始3DGS纯几何驱动的密度控制, 语义边界检测提供额外引导

2. 更真实的语义特征分配策略 (改进原有空间距离法)
   - 基于高斯聚类的语义区域划分
   - 模拟SAM+CLIP的输出特性

支持:
- 关键帧触发高斯生成
- 语义特征关联 (开放集语义)
- 冗余高斯剪枝
- 自适应密度控制
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from ..core.gaussian_model import GaussianCloud
from ..core.adaptive_density import (
    AdaptiveDensityController,
    run_adaptive_densification_cycle
)
from ..core.camera import PinholeCamera


class DenseMapper:
    """增量稠密建图器 (改进版)"""

    def __init__(self, capacity: int = 10000,
                 use_adaptive_density: bool = True,
                 sem_weight: float = 0.3):
        self.map = GaussianCloud(capacity)
        self.kf_count = 0
        self.use_adaptive_density = use_adaptive_density
        self.density_ctrl = AdaptiveDensityController(sem_grad_weight=sem_weight)
        self.density_history: List[Dict] = []

    def add_keyframe(self, pos: np.ndarray, rgb: np.ndarray,
                     sem: np.ndarray = None):
        """添加关键帧点云到高斯地图"""
        self.map.add(pos, rgb, sem)
        self.kf_count += 1

    def add_pointcloud(self, pts_world: np.ndarray, colors: np.ndarray):
        """添加世界坐标系下的稠密点云"""
        self.map.add(pts_world, colors)

    def prune(self):
        """剪枝低质量高斯"""
        self.map.prune(0.05)

    def get_map(self) -> Dict[str, np.ndarray]:
        """获取打包的高斯地图"""
        return self.map.pack()

    def size(self) -> int:
        return len(self.map)

    def run_densification(self, n_cycles: int = 3) -> Dict:
        """
        运行自适应密度控制 (改进方法核心)

        Returns:
            stats: 密度控制统计
        """
        if not self.use_adaptive_density:
            return {'status': 'disabled'}

        gs_data = self.get_map()
        initial_n = len(gs_data['xyz'])

        gs_data = run_adaptive_densification_cycle(
            gs_data, self.density_ctrl, n_cycles
        )

        final_n = len(gs_data['xyz'])
        stats = self.density_ctrl.get_stats()
        stats['initial_n'] = initial_n
        stats['final_n'] = final_n
        stats['growth_ratio'] = final_n / max(initial_n, 1)

        self.density_history.append(stats)

        # 将更新后的数据写回高斯云
        self._update_map_from_data(gs_data)

        return stats

    def _update_map_from_data(self, gs_data: Dict[str, np.ndarray]):
        """将字典数据写回GaussianCloud"""
        N = len(gs_data['xyz'])
        self.map._N = min(N, self.map._cap)
        n = self.map._N
        self.map.xyz[:n] = gs_data['xyz'][:n]
        self.map.rgb[:n] = gs_data['rgb'][:n]
        if 'opacity' in gs_data and gs_data['opacity'].shape[1] == 1:
            self.map.opacity[:n] = gs_data['opacity'][:n]
        else:
            opac = gs_data['opacity']
            if opac.ndim == 1:
                self.map.opacity[:n, 0] = opac[:n]
            else:
                self.map.opacity[:n] = opac[:n]
        if 'scale' in gs_data:
            self.map.scale[:n] = np.log(np.maximum(gs_data['scale'][:n], 1e-6))
        if 'sem' in gs_data:
            self.map.sem[:n] = gs_data['sem'][:n]

    def assign_semantic_regions(self, n_regions: int = 4) -> np.ndarray:
        """
        按空间位置和聚类分配语义特征 (模拟OpenMonoGS-SLAM的语义)
        在实际系统中由SAM+CLIP生成

        改进: 使用K-means风格的语义分配, 而非简单距离阈值

        Args:
            n_regions: 语义区域数量

        Returns:
            sem: [N, 64] 语义特征矩阵
        """
        xyz = self.map.xyz[:len(self.map)]
        N = len(self.map)

        if N == 0:
            return np.zeros((0, 64), dtype=np.float32)

        sem = np.zeros((N, 64), dtype=np.float32)
        dim_per_region = 64 // n_regions

        # 使用简单的K-means in空间维度进行聚类
        # (实际系统中由SAM生成mask, CLIP生成特征)
        max_iters = 10
        # 初始化聚类中心
        rng = np.random.RandomState(42)
        centers = xyz[rng.choice(N, min(n_regions, N), replace=False)]

        labels = np.zeros(N, dtype=int)
        for _ in range(max_iters):
            # 分配
            dists = np.sum((xyz[:, None] - centers[None]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)
            # 更新
            for k in range(n_regions):
                mask = labels == k
                if mask.any():
                    centers[k] = xyz[mask].mean(axis=0)

        # 为每个区域分配语义特征
        for k in range(n_regions):
            mask = labels == k
            start = k * dim_per_region
            end = start + dim_per_region
            # 在对应维度段中填充特征
            sem[mask, start:end] = 1.0
            # 加一些随机变化模拟真实CLIP特征
            noise = rng.randn(mask.sum(), dim_per_region).astype(np.float32) * 0.05
            sem[mask, start:end] += noise

        # 归一化 (CLIP特征通常归一化)
        norms = np.linalg.norm(sem, axis=1, keepdims=True) + 1e-8
        sem = sem / norms

        self.map.sem[:N] = sem
        return sem

    def compute_semantic_boundaries(self) -> np.ndarray:
        """
        计算语义边界得分 (用于可视化)
        在语义特征变化的区域标记边界
        """
        sem = self.map.sem[:len(self.map)]
        N = len(sem)
        if N < 2:
            return np.zeros(N, dtype=np.float32)

        ctrl = AdaptiveDensityController()
        return ctrl.compute_semantic_boundary_score(self.get_map())