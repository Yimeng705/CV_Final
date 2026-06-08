"""
自适应密度控制 (改进版)
=====================
基于3DGS综述 method-003 (Adaptive Density Control) 的改进

提出改进: 语义感知的自适应高斯密度控制
(Semantic-Aware Adaptive Gaussian Densification)

动机:
- 原始3DGS的自适应密度控制仅基于梯度大小, 在语义边界区域可能欠采样
- 语义边界(不同物体交界)是3D重建中最容易出现模糊和细节丢失的区域
- OpenMonoGS-SLAM提供了语义特征, 但未利用于几何优化

方法:
1. 计算每个高斯的语义梯度 (特征空间变化)
2. 在语义边界区域降低密度控制阈值, 允许更多高斯分裂
3. 同时保持几何梯度引导, 形成双驱动密度控制

与原始方法的对比:
- 原始: 密度控制仅依赖视图空间梯度
- 改进: 结合几何梯度 + 语义边界检测, 在语义边界增加高斯密度
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class AdaptiveDensityController:
    """
    自适应高斯密度控制器 (改进版)
    结合语义感知的密度调整策略
    """

    def __init__(self,
                 grad_threshold: float = 0.0002,
                 scale_threshold: float = 0.01,
                 opacity_threshold: float = 0.05,
                 sem_grad_weight: float = 0.3):
        """
        Args:
            grad_threshold: 几何梯度阈值 (触发克隆/分裂)
            scale_threshold: 尺度阈值 (决定克隆还是分裂)
            opacity_threshold: 不透明度阈值 (低于此值则移除)
            sem_grad_weight: 语义梯度权重 (我们的改进: 0=纯几何, 1=纯语义)
        """
        self.grad_threshold = grad_threshold
        self.scale_threshold = scale_threshold
        self.opacity_threshold = opacity_threshold
        self.sem_grad_weight = sem_grad_weight

        # 统计
        self.stats = {
            'n_cloned': 0,
            'n_split': 0,
            'n_pruned': 0,
            'n_semantic_boost': 0
        }

    def compute_semantic_boundary_score(self,
                                        gs_data: Dict[str, np.ndarray],
                                        k_neighbors: int = 8) -> np.ndarray:
        """
        计算每个高斯的语义边界得分

        通过计算每个高斯与其K近邻的语义特征距离来估计语义边界。
        语义特征差异大的位置表明语义边界, 需要更多高斯来精确建模。

        Args:
            gs_data: 高斯数据
            k_neighbors: K近邻数量

        Returns:
            sem_boundary_score: [N] 浮点数, 0=非边界, 1=强边界
        """
        xyz = gs_data['xyz']
        sem = gs_data.get('sem', np.zeros((len(xyz), 64), dtype=np.float32))
        N = len(xyz)

        if N < k_neighbors + 1:
            return np.zeros(N, dtype=np.float32)

        scores = np.zeros(N, dtype=np.float32)

        # 简化版: 用空间距离 + 语义距离计算边界得分
        # 构建简单的KD-tree近似 (使用距离矩阵采样)
        sample_size = min(N, 500)
        indices = np.random.choice(N, sample_size, replace=False) if N > sample_size else np.arange(N)

        for i in indices:
            # 找空间最近邻
            dist = np.sum((xyz - xyz[i]) ** 2, axis=1)
            nn_idx = np.argsort(dist)[1:k_neighbors+1]

            # 计算语义特征差异
            sem_diff = np.mean(np.sum((sem[nn_idx] - sem[i]) ** 2, axis=1))

            # 归一化
            scores[i] = np.clip(sem_diff / (np.max(sem_diff) if sem_diff > 0 else 1.0), 0, 1)

        # 扩张到全量
        if sample_size < N:
            for i in range(N):
                if i not in indices:
                    dist = np.sum((xyz - xyz[i]) ** 2, axis=1)
                    nn_idx = np.argsort(dist)[1:k_neighbors+1]
                    sem_diff = np.mean(np.sum((sem[nn_idx] - sem[i]) ** 2, axis=1))
                    scores[i] = np.clip(sem_diff, 0, 1)

        return scores

    def should_densify(self,
                       geom_grad: np.ndarray,
                       sem_boundary_score: np.ndarray,
                       scales: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        判断哪些高斯需要密度调整 (改进版)

        改进: 将语义边界得分融入梯度阈值, 降低语义边界处的分裂阈值

        Args:
            geom_grad: [N] 视图空间几何梯度范数
            sem_boundary_score: [N] 语义边界得分 [0,1]
            scales: [N,3] 高斯尺度

        Returns:
            clone_mask: [N] bool, 需要克隆的高斯
            split_mask: [N] bool, 需要分裂的高斯
            prune_mask: [N] bool, 需要移除的高斯
        """
        # 改进: 语义感知的梯度阈值
        # 在语义边界处降低阈值, 鼓励更多分裂
        adjusted_threshold = self.grad_threshold / (
            1.0 + self.sem_grad_weight * sem_boundary_score
        )

        # 大小判断: 小高斯克隆, 大高斯分裂
        scale_norm = np.max(scales, axis=1)
        is_small = scale_norm < self.scale_threshold

        clone_mask = (geom_grad > adjusted_threshold) & is_small
        split_mask = (geom_grad > adjusted_threshold) & ~is_small

        # 统计语义边界对分裂的贡献
        sem_boosted = sem_boundary_score > 0.3
        self.stats['n_semantic_boost'] += int(np.sum(clone_mask & sem_boosted) +
                                              np.sum(split_mask & sem_boosted))

        return clone_mask, split_mask

    def prune(self, gs_data: Dict[str, np.ndarray],
              opacities: np.ndarray,
              max_world_size: float = 10.0) -> np.ndarray:
        """
        判断需要移除的高斯

        Args:
            gs_data: 高斯数据
            opacities: [N] 不透明度值
            max_world_size: 世界空间最大尺度

        Returns:
            prune_mask: [N] bool, 需要移除的高斯
        """
        xyz = gs_data['xyz']
        scales = gs_data['scale']
        N = len(xyz)

        # 低不透明度移除
        low_opacity = opacities < self.opacity_threshold

        # 过大尺度移除
        too_large = np.max(scales, axis=1) > max_world_size

        # 无穷远处移除
        too_far = np.linalg.norm(xyz, axis=1) > 50.0

        prune_mask = low_opacity | too_large | too_far
        self.stats['n_pruned'] += int(np.sum(prune_mask))

        return prune_mask

    def execute_clone(self, gs_data: Dict[str, np.ndarray],
                      clone_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """执行克隆操作: 在当前位置创建副本"""
        N = len(gs_data['xyz'])
        n_clone = int(np.sum(clone_mask))

        if n_clone == 0:
            return gs_data

        new_data = {}
        clone_indices = np.where(clone_mask)[0]

        for key in gs_data:
            val = gs_data[key]
            if key == 'scale':
                # 克隆的scale略缩小
                new_val = val[clone_indices] * 0.8
            elif key == 'sem':
                new_val = val[clone_indices].copy()
            elif key in ('xyz', 'rgb', 'opacity', 'rot'):
                new_val = val[clone_indices].copy()
            else:
                if val.shape[0] == N:
                    new_val = val[clone_indices].copy()
                else:
                    new_val = val

            if val.shape[0] == N and key in ('xyz', 'rgb', 'opacity', 'scale', 'rot', 'sem', 'cov'):
                gs_data[key] = np.concatenate([val, new_val], axis=0)
            # 其他情况保持不变

        self.stats['n_cloned'] += n_clone
        return gs_data

    def execute_split(self, gs_data: Dict[str, np.ndarray],
                      split_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """执行分裂操作: 将一个高斯分裂为两个"""
        N = len(gs_data['xyz'])
        n_split = int(np.sum(split_mask))

        if n_split == 0:
            return gs_data

        split_indices = np.where(split_mask)[0]
        xyz = gs_data['xyz']
        scales_orig = gs_data['scale']

        new_parts = {}

        for key in gs_data:
            val = gs_data[key]
            if val.shape[0] != N:
                continue

            if key == 'xyz':
                # 沿最大尺度方向偏移
                for idx in split_indices:
                    max_axis = np.argmax(scales_orig[idx])
                    offset = np.zeros(3, dtype=np.float32)
                    offset[max_axis] = scales_orig[idx, max_axis] * 0.5
                    new_parts.setdefault(key, []).append(val[idx] + offset)
                    new_parts.setdefault(key, []).append(val[idx] - offset)
            elif key == 'scale':
                # 缩小尺度
                for idx in split_indices:
                    new_parts.setdefault(key, []).append(val[idx] * 0.7)
                    new_parts.setdefault(key, []).append(val[idx] * 0.7)
            elif key in ('rgb', 'opacity', 'sem', 'rot'):
                for idx in split_indices:
                    new_parts.setdefault(key, []).append(val[idx].copy())
                    new_parts.setdefault(key, []).append(val[idx].copy())

        # 更新数据
        for key, parts in new_parts.items():
            gs_data[key] = np.concatenate(
                [gs_data[key]] + [np.stack(parts, axis=0)], axis=0
            )

        self.stats['n_split'] += n_split * 2  # 每个分裂产生2个
        return gs_data

    def reset_stats(self):
        """重置统计"""
        self.stats = {
            'n_cloned': 0,
            'n_split': 0,
            'n_pruned': 0,
            'n_semantic_boost': 0
        }

    def get_stats(self) -> Dict:
        return self.stats


def run_adaptive_densification_cycle(
    gs_data: Dict[str, np.ndarray],
    controller: AdaptiveDensityController,
    n_iterations: int = 5
) -> Dict[str, np.ndarray]:
    """
    运行一轮完整的自适应密度控制周期

    模拟3DGS的周期性密度调整:
    1. 评估语义边界得分
    2. 模拟几何梯度 (用随机噪声模拟, 实际系统由反向传播提供)
    3. 执行克隆/分裂/剪枝
    """
    for it in range(n_iterations):
        N = len(gs_data['xyz'])
        if N == 0:
            break

        # 模拟几何梯度 (实际系统中由反向传播提供)
        geom_grad = np.random.uniform(0, 0.001, N).astype(np.float32)

        # 计算语义边界得分
        sem_score = controller.compute_semantic_boundary_score(gs_data)

        # 判断密度调整
        clone_mask, split_mask = controller.should_densify(
            geom_grad, sem_score, gs_data['scale']
        )

        # 执行克隆
        if clone_mask.any():
            gs_data = controller.execute_clone(gs_data, clone_mask)

        # 执行分裂
        if split_mask.any():
            gs_data = controller.execute_split(gs_data, split_mask)

        # 执行剪枝
        opacities = gs_data['opacity']
        prune_mask = controller.prune(gs_data, opacities)

        if prune_mask.any():
            keep = ~prune_mask
            for key in ('xyz', 'rgb', 'opacity', 'scale', 'rot', 'sem', 'cov'):
                if key in gs_data and gs_data[key].shape[0] == len(keep):
                    gs_data[key] = gs_data[key][keep]

    return gs_data