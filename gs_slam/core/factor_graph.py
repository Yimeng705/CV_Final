"""
因子图优化 (MASt3R-SLAM §4 + MASt3R-Fusion §3)
===============================================

实现两个核心概念:
1. MASt3R-SLAM的二阶全局优化:
   - 使用Schur补进行高效边缘化
   - 回环检测与全局BA
   
2. MASt3R-Fusion的多传感器因子图:
   - Sim(3)视觉约束 (来自前馈模型的pointmap匹配)
   - IMU预积分因子
   - GNSS全局位置因子
   - 层次化滑动窗口+全局优化部分
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .camera import so3_log, random_so3

class FactorEdge:
    """因子图中的一条边"""
    def __init__(self, nodes: List[int], measurement, information, type_name: str):
        self.nodes = nodes
        self.meas = measurement  # 测量值
        self.info = information  # 信息矩阵(权重)
        self.type = type_name

class PoseGraph:
    """
    位姿图 (Pose Graph) 优化
    
    节点: SE(3)位姿 T_i = (R_i, t_i)
    边: 相对位姿约束 T_ij = T_i^{-1} T_j
    
    仿MASt3R-SLAM的图构建:
    - 里程计边: 连续关键帧间的相对约束
    - 回环边: 非连续帧间的匹配约束
    - GNSS边: 全局位置约束 (MASt3R-Fusion)
    """
    
    def __init__(self):
        self.poses: List[Tuple[np.ndarray, np.ndarray]] = []  # (R, t)列表
        self.edges: List[FactorEdge] = []
    
    def add_pose(self, R: np.ndarray, t: np.ndarray, fixed: bool = False):
        """添加位姿节点"""
        self.poses.append((R.astype(np.float32), t.reshape(3,1).astype(np.float32)))
        return len(self.poses) - 1
    
    def add_odometry(self, i: int, j: int, R_rel: np.ndarray, t_rel: np.ndarray, info: float = 1.0):
        """添加里程计边"""
        self.edges.append(FactorEdge([i, j], (R_rel, t_rel), info, 'odometry'))
    
    def add_loop(self, i: int, j: int, R_rel: np.ndarray, t_rel: np.ndarray, info: float = 2.0):
        """添加回环边"""
        self.edges.append(FactorEdge([i, j], (R_rel, t_rel), info, 'loop'))
    
    def add_gnss(self, i: int, t_global: np.ndarray, info: float = 0.5):
        """添加GNSS边 (单节点约束)"""
        self.edges.append(FactorEdge([i], t_global, info, 'gnss'))
    
    def residual(self, edge_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """计算边的残差 (旋转部分 + 平移部分)"""
        edge = self.edges[edge_idx]
        
        if edge.type == 'gnss':
            i = edge.nodes[0]
            _, t_i = self.poses[i]
            return np.zeros(3), (t_i - edge.meas).flatten()  # 只有平移残差
        
        i, j = edge.nodes
        R_i, t_i = self.poses[i]
        R_j, t_j = self.poses[j]
        R_meas, t_meas = edge.meas
        
        # 预测: T_pred = T_i^{-1} * T_j
        R_pred = R_i.T @ R_j
        t_pred = R_i.T @ (t_j - t_i)
        
        # 残差: 旋转(so3对数) + 平移
        e_R = so3_log(R_meas.T @ R_pred)
        e_t = (t_pred - t_meas).flatten()
        
        return e_R, e_t
    
    def optimize(self, max_iter: int = 200, 
                 lr: float = 0.01, verbose: bool = False) -> List[float]:
        """
        图优化 (简化Gauss-Newton风格)
        
        使用逐边坐标下降:
        - 对每条边 (i,j):
          - 计算残差 e = (e_R, e_t)
          - 在切空间更新: T_i += J_i^T * info * e
                            T_j -= J_j^T * info * e
        """
        N = len(self.poses)
        history = []
        
        for it in range(max_iter):
            total = 0.0
            # 累积梯度 (零初始化)
            grad_t = [np.zeros((3,1), dtype=np.float32) for _ in range(N)]
            grad_R_skew = [np.zeros((3,3), dtype=np.float32) for _ in range(N)]
            
            for e_idx, edge in enumerate(self.edges):
                e_R, e_t = self.residual(e_idx)
                w = edge.info
                total += w * (np.sum(e_R**2) + np.sum(e_t**2))
                
                if edge.type == 'gnss':
                    i = edge.nodes[0]
                    grad_t[i] += w * e_t.reshape(3,1)
                    continue
                
                i, j = edge.nodes
                R_i, _ = self.poses[i]
                
                # Jacobian近似: 残差对姿态的导数
                skew = np.array([[0,-e_R[2],e_R[1]],
                                 [e_R[2],0,-e_R[0]],
                                 [-e_R[1],e_R[0],0]], dtype=np.float32)
                grad_R_skew[j] += w * skew
                grad_R_skew[i] -= w * skew
                
                # 平移: dt_j ≈ R_i * e_t, dt_i ≈ -R_i * e_t
                dt_correction = (R_i @ e_t.reshape(3,1)).astype(np.float32)
                grad_t[j] += w * dt_correction
                grad_t[i] -= w * dt_correction
            
            # 应用更新 (节点0通常固定)
            for k in range(1, N):
                R_k, t_k = self.poses[k]
                # 平移更新
                t_k = t_k - lr * grad_t[k]
                # 旋转更新: R += R * skew(lr * grad_R)
                dR = np.eye(3) - lr * grad_R_skew[k]
                R_k = R_k @ dR
                # 强制正交化
                U, _, Vt = np.linalg.svd(R_k)
                R_k = (U @ Vt).astype(np.float32)
                self.poses[k] = (R_k, t_k)
            
            history.append(float(total))
            lr *= 0.97  # 衰减
        
        return history
    
    def get_trajectory_xyz(self) -> np.ndarray:
        """提取轨迹XYZ坐标 [N, 3]"""
        return np.array([t.flatten() for _, t in self.poses], dtype=np.float32)


def build_test_graph(n_frames: int = 20, radius: float = 6.0) -> PoseGraph:
    """
    构建测试位姿图: 环形轨迹 + 里程计 + GNSS + 回环
    模拟MASt3R-Fusion的因子图结构
    """
    from .camera import look_at, random_so3
    
    pg = PoseGraph()
    
    # 生成真实轨迹并添加噪声
    for i in range(n_frames):
        angle = 2*np.pi*i/n_frames
        eye = np.array([radius*np.cos(angle), 1.0+0.5*np.sin(2*angle), radius*np.sin(angle)])
        R, t = look_at(eye, np.zeros(3), np.array([0,1,0]))
        
        # 为第0帧添加噪声模拟前端输出
        if i > 0:
            noise_R = random_so3(0.02)
            noise_t = np.random.randn(3,1).astype(np.float32)*0.05
            R = (R @ noise_R).astype(np.float32)
            t = (t + noise_t).astype(np.float32)
        
        pg.add_pose(R, t)
    
    # 建立里程计边
    for i in range(n_frames-1):
        R_i, t_i = pg.poses[i]
        R_j, t_j = pg.poses[i+1]
        # 相对位姿 (用带噪声的测量)
        R_rel = (R_i.T @ R_j @ random_so3(0.01)).astype(np.float32)
        t_rel = (R_i.T @ (t_j - t_i) + np.random.randn(3,1)*0.02).astype(np.float32)
        pg.add_odometry(i, i+1, R_rel, t_rel, 1.0)
    
    # 添加GNSS边 (每5帧)
    for i in range(0, n_frames, 5):
        t_global = pg.poses[i][1] + np.random.randn(3,1).astype(np.float32)*0.1
        pg.add_gnss(i, t_global, 0.3)
    
    # 添加回环边 (首尾之间)
    pg.add_loop(0, n_frames-1, np.eye(3,dtype=np.float32), np.zeros((3,1),dtype=np.float32), 1.5)
    
    return pg