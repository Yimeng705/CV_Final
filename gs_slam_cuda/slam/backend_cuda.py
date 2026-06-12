"""
CUDA-Accelerated SLAM Backend
===============================
Based on MASt3R-SLAM (method-005) and MASt3R-Fusion (method-004).

Implements:
- Global factor graph optimization
- Loop closure detection and verification
- Hierarchical optimization (sliding window + global)
- GPU-accelerated sparse Cholesky via torch.linalg
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from ..core.camera import CameraPose
from ..core.factor_graph_cuda import (
    CUDAFactorGraph, VisualFactor, IMUFactor, GNSSFactor,
    exp_so3, compute_ate, compute_rpe
)
from ..core.cuda_wrapper import CudaContext


class CUDABackend:
    """
    CUDA-accelerated SLAM backend.
    
    Pipeline:
    1. Receive keyframe poses and visual factors from frontend
    2. Build hierarchical factor graph
    3. Run real-time sliding window optimization
    4. Detect loop closures (simulated for synthetic data)
    5. Run global batch optimization
    6. Output optimized trajectory and metrics
    """

    def __init__(self, device=None):
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.factor_graph = CUDAFactorGraph(device=self.device)
        self.optimized_poses: List[CameraPose] = []
        self.loop_closure_edges: List[tuple] = []

    def add_keyframe(self,
                     pose: CameraPose,
                     prev_pose: CameraPose,
                     matching_score: float):
        """
        Add a new keyframe with visual and odometry factors.
        
        Args:
            pose: Current keyframe pose
            prev_pose: Previous keyframe pose
            matching_score: Pointmap matching quality (0-1)
        """
        kf_idx = len(self.optimized_poses)
        self.optimized_poses.append(pose)

        if kf_idx > 0:
            # Add visual factor (Sim(3) compact Hessian)
            # In production, this would come from MASt3R pointmap alignment
            H_vis = np.eye(7, dtype=np.float32) * matching_score
            v_vis = np.zeros(7, dtype=np.float32)
            self.factor_graph.add_visual_factor(
                kf_idx - 1, kf_idx, H_vis, v_vis,
                weight=matching_score
            )

            # Add IMU pre-integration factor (simulated)
            # In production, this comes from actual IMU measurements
            delta_pose = np.zeros(6, dtype=np.float32)
            info = np.eye(6, dtype=np.float32) * 100.0
            self.factor_graph.add_imu_factor(
                kf_idx - 1, kf_idx, delta_pose, info
            )

    def detect_loop_closures(self,
                              current_pose: CameraPose,
                              history_poses: List[CameraPose],
                              distance_threshold: float = 2.0) -> List[int]:
        """
        Simulated loop closure detection based on spatial proximity.
        
        In production: Replace with ASMK retrieval (MASt3R-SLAM method-004).
        
        Args:
            current_pose: Current keyframe pose
            history_poses: Previous keyframe poses
            distance_threshold: Maximum distance for loop closure (m)
        
        Returns:
            candidate_indices: List of candidate keyframe indices
        """
        candidates = []
        current_t = current_pose.t.flatten()[:3]
        
        for i, pose in enumerate(history_poses):
            # Skip consecutive frames
            if i >= len(history_poses) - 5:
                continue
            
            hist_t = pose.t.flatten()[:3]
            dist = np.linalg.norm(current_t - hist_t)
            
            if dist < distance_threshold:
                candidates.append(i)
        
        return candidates

    def add_loop_closure(self,
                          i: int, j: int,
                          confidence: float = 0.5):
        """
        Add a verified loop closure edge.
        
        Args:
            i, j: Keyframe indices forming loop closure
            confidence: Loop closure confidence (0-1)
        """
        # Simulated compact Hessian for loop closure
        H_lc = np.eye(7, dtype=np.float32) * confidence * 10.0
        v_lc = np.zeros(7, dtype=np.float32)
        self.factor_graph.add_loop_closure(i, j, H_lc, v_lc)
        self.loop_closure_edges.append((i, j))

    def run_sliding_window_optimization(self,
                                         window_size: int = 8) -> List[CameraPose]:
        """Run real-time sliding window optimization."""
        if len(self.optimized_poses) < 2:
            return self.optimized_poses
        
        self.optimized_poses = self.factor_graph.optimize_sliding_window(
            self.optimized_poses, window_size
        )
        return self.optimized_poses

    def run_global_optimization(self) -> List[CameraPose]:
        """Run global factor graph optimization."""
        if len(self.optimized_poses) < 3:
            return self.optimized_poses
        
        self.optimized_poses = self.factor_graph.optimize_global(
            self.optimized_poses
        )
        return self.optimized_poses

    def evaluate_trajectory(self,
                             gt_poses: List[CameraPose]) -> Dict:
        """
        Evaluate trajectory accuracy.
        
        Returns:
            metrics: {
                'ate_rmse': ATE RMSE (m),
                'rpe_trans': Relative translation error (%),
                'rpe_rot': Relative rotation error (deg/m),
                'n_loop_closures': Number of verified loop closures
            }
        """
        if len(gt_poses) >= 2:
            ate = compute_ate(self.optimized_poses, gt_poses)
            rpe_t, rpe_r = compute_rpe(self.optimized_poses, gt_poses)
        else:
            ate = 0.0
            rpe_t, rpe_r = 0.0, 0.0
        
        return {
            'ate_rmse': ate,
            'rpe_trans': rpe_t,
            'rpe_rot': rpe_r,
            'n_loop_closures': len(self.loop_closure_edges),
            'n_keyframes': len(self.optimized_poses)
        }

    def get_optimized_trajectory(self) -> np.ndarray:
        """Get optimized trajectory positions."""
        return np.array([p.t.flatten()[:3] for p in self.optimized_poses])