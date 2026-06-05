"""
3D Gaussian Splatting 模型
==========================
基于综述论文 (Chen & Wang, 2024) 中的公式，与OpenMonoGS-SLAM的表示兼容。

每个3D高斯: Θ = {μ, q, s, α, c, f_sem}
- μ ∈ R³: 均值(位置)
- q ∈ R⁴: 旋转四元数
- s ∈ R³: 尺度(对角)
- α ∈ [0,1]: 不透明度
- c ∈ R³: RGB颜色(SH系数0阶简化)
- f_sem ∈ R^D: 语义特征(OpenMonoGS-SLAM创新)

协方差: Σ = (R(q) · diag(s)) · (R(q) · diag(s))^T

渲染: C = Σ c_i·α_i·G'_i(x)·Π_{j<i}(1-α_j·G'_j(x))
"""
import numpy as np
from typing import Dict, Optional, Tuple

class GaussianCloud:
    """可微分高斯云 (NumPy简化版)"""
    
    def __init__(self, capacity: int = 20000):
        self._cap = capacity
        self._N = 0
        
        self.xyz = np.zeros((capacity, 3), dtype=np.float32)      # 位置
        self.rot = np.zeros((capacity, 4), dtype=np.float32)       # 四元数
        self.scale = np.zeros((capacity, 3), dtype=np.float32)     # 尺度(log)
        self.opacity = np.zeros((capacity, 1), dtype=np.float32)   # 不透明度(logit)
        self.rgb = np.ones((capacity, 3), dtype=np.float32) * 0.5 # RGB
        self.sem = np.zeros((capacity, 64), dtype=np.float32)      # 语义特征
        
        self.rot[:, 0] = 1.0  # w=1 → 无旋转
        self.scale[:] = np.log(0.8)   # 初始尺度 ~0.8m (增大以覆盖更大区域)
        self.opacity[:] = 0.8   # sigmoid(0.8) ≈ 0.69 (更高不透明度)
    
    def __len__(self) -> int:
        return self._N
    
    @property
    def scales_actual(self) -> np.ndarray:
        return np.exp(self.scale[:self._N])
    
    @property
    def opacities_actual(self) -> np.ndarray:
        x = self.opacity[:self._N]
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def _quat2rot(self, q: np.ndarray) -> np.ndarray:
        """四元数→旋转矩阵(批量)"""
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-10)
        w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
        N = len(q)
        R = np.zeros((N, 3, 3), dtype=np.float32)
        R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
        R[:,1,0]=2*(x*y+w*z); R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
        R[:,2,0]=2*(x*z-w*y); R[:,2,1]=2*(y*z+w*x); R[:,2,2]=1-2*(x*x+y*y)
        return R
    
    def get_covariances(self) -> np.ndarray:
        """Σ = R S S^T R^T"""
        R = self._quat2rot(self.rot[:self._N])
        S = self.scales_actual[:, :, None] * np.eye(3)[None]  # [N,3,3] diag
        RS = R @ S
        return RS @ RS.transpose(0, 2, 1)
    
    def add(self, pos: np.ndarray, rgb: Optional[np.ndarray] = None,
            sem: Optional[np.ndarray] = None) -> int:
        """批量添加高斯"""
        k = min(len(pos), self._cap - self._N)
        if k <= 0: return 0
        s = self._N
        self.xyz[s:s+k] = pos[:k]
        self.rot[s:s+k] = [1.0, 0, 0, 0]
        if rgb is not None: self.rgb[s:s+k] = np.clip(rgb[:k], 0, 1)
        if sem is not None: self.sem[s:s+k] = sem[:k]
        self._N += k
        return k
    
    def prune(self, min_opacity: float = 0.05):
        """移除低不透明度高斯"""
        a = self.opacities_actual
        keep = a[:, 0] > min_opacity
        n = keep.sum()
        if n < self._N:
            for attr in ['xyz', 'rot', 'scale', 'opacity', 'rgb', 'sem']:
                arr = getattr(self, attr)
                arr[:n] = arr[:self._N][keep]
            self._N = int(n)
    
    def pack(self) -> Dict[str, np.ndarray]:
        """打包为渲染用的字典"""
        N = self._N
        return {'xyz': self.xyz[:N], 'rgb': self.rgb[:N],
                'scale': self.scales_actual, 'opacity': self.opacities_actual,
                'sem': self.sem[:N], 'cov': self.get_covariances(),
                'rot': self.rot[:N]}


def make_test_scene(n: int = 300) -> GaussianCloud:
    """创建测试场景: 球+盒+地"""
    gc = GaussianCloud(n*2)
    m = n // 3
    
    # Sphere
    phi, theta = np.random.uniform(0, np.pi, m), np.random.uniform(0, 2*np.pi, m)
    r = 2.0 + np.random.uniform(-0.3, 0.3, m)
    s = np.column_stack([r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)])
    
    # Box
    b = np.random.uniform(-1.5, 1.5, (m, 3)).astype(np.float32)
    b[:,0] += 4.0; b[:,2] -= 1.0
    
    # Floor
    f = np.column_stack([np.random.uniform(-5,5,m), np.full(m,-3.0), np.random.uniform(-5,5,m)])
    
    pos = np.vstack([s, b, f]).astype(np.float32)
    col = np.vstack([np.tile([0.2,0.6,0.9],(m,1)),
                     np.tile([0.9,0.3,0.3],(m,1)),
                     np.tile([0.3,0.8,0.3],(m,1))]).astype(np.float32)
    gc.add(pos, col)
    return gc