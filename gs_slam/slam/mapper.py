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

    def run_densification(self, n_cycles: int = 3, camera=None) -> Dict:
        """
        运行自适应密度控制 (改进方法核心)

        Args:
            n_cycles: 密度控制循环轮数
            camera: PinholeCamera 观测相机 (用于几何重要性代理)

        Returns:
            stats: 密度控制统计
        """
        if not self.use_adaptive_density:
            return {'status': 'disabled'}

        gs_data = self.get_map()
        initial_n = len(gs_data['xyz'])

        gs_data = run_adaptive_densification_cycle(
            gs_data, self.density_ctrl, n_cycles,
            camera=camera
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
        分配语义特征 (模拟OpenMonoGS-SLAM的语义, v3.2改进版)
        在实际系统中由SAM+CLIP生成

        v3.2 改进: 空间K-means聚类 → 每个空间簇获得正交语义特征
        - 高斯来自8个关键帧沿轨迹分布, 空间K-means(4区域)产生自然簇边界
        - 每个簇在64-dim嵌入空间中占据互为正交的子空间
        - 不加空间平滑 → 边界处高斯与邻域内其他簇的高斯语义距离显著大
          → compute_semantic_boundary_score有效检测到边界
        - 簇半径 ~5m (场景跨度10m / 2), 边界得分 ~0.3-0.5

        Args:
            n_regions: 语义区域数量 (4个空间簇)

        Returns:
            sem: [N, 64] 语义特征矩阵
        """
        xyz = self.map.xyz[:len(self.map)]
        N = len(self.map)

        if N == 0:
            return np.zeros((0, 64), dtype=np.float32)

        if N < n_regions:
            n_regions = max(2, N // 50)

        dim_per_region = 64 // n_regions
        rng = np.random.RandomState(42)

        # --- K-means 空间聚类 (3D位置) ---
        max_iters = 15
        centers = xyz[rng.choice(N, n_regions, replace=False)]
        labels = np.zeros(N, dtype=int)

        for _ in range(max_iters):
            dists = np.sum((xyz[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(n_regions):
                mask = labels == k
                if mask.any():
                    centers[k] = xyz[mask].mean(axis=0)

        # --- 构建语义特征: 每个簇占 dim_per_region 维 ---
        # 不同簇在嵌入空间的正交子空间中 → 簇间语义距离 ~sqrt(2)
        sem = np.zeros((N, 64), dtype=np.float32)

        for k in range(n_regions):
            mask = labels == k
            start = k * dim_per_region
            end = start + dim_per_region
            if mask.any():
                sem[mask, start:end] = 1.0
                # 小幅噪声模拟CLIP特征变异性
                noise = rng.randn(mask.sum(), dim_per_region).astype(np.float32) * 0.05
                sem[mask, start:end] += noise

        # 归一化到单位球面 (保证内积=余弦相似度)
        norms = np.linalg.norm(sem, axis=1, keepdims=True) + 1e-12
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