"""
CUDA-Accelerated 3D Gaussian Model
===================================
PyTorch-based Gaussian representation with GPU tensors.

Each Gaussian: Θ = {μ ∈ R³, s ∈ R³, q ∈ R⁴, α ∈ [0,1], c ∈ R³, f_sem ∈ R^D}

Memory footprint (RTX 3060 8GB, 500K Gaussians):
- xyz: 500K × 3 × 4B = 6MB
- rgb: 500K × 3 × 4B = 6MB
- scale: 500K × 3 × 4B = 6MB
- rot: 500K × 4 × 4B = 8MB
- opacity: 500K × 1 × 4B = 2MB
- sem: 500K × 64 × 4B = 128MB
- cov (cached): 500K × 9 × 4B = 18MB
- Total: ~174MB (well within 8GB budget)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from .cuda_wrapper import CudaContext


class GaussianCloudCUDA:
    """
    GPU-based 3D Gaussian representation.
    
    Enhanced with:
    - Batch covariance computation
    - GPU-specific memory layout optimization
    - In-place back-project operations
    - Semantic feature support (OpenMonoGS-SLAM style)
    """

    def __init__(self, max_gaussians: int = 500000,
                 sem_dim: int = 64,
                 device=None):
        """
        Initialize Gaussian cloud on GPU.
        
        Args:
            max_gaussians: Maximum number of Gaussians pre-allocated
            sem_dim: Semantic feature dimension (64 for DINOv2/CLIP)
            device: CUDA device (default: cuda:0)
        """
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.sem_dim = sem_dim
        self._cap = max_gaussians
        self._N = 0  # Current count
        
        # Allocate GPU tensors (use torch.empty for speed, fill later)
        self.xyz = torch.zeros(max_gaussians, 3, device=device, dtype=torch.float32)
        self.rgb = torch.ones(max_gaussians, 3, device=device, dtype=torch.float32) * 0.5
        self.scale_raw = torch.full((max_gaussians, 3), -0.223, device=device, dtype=torch.float32)  # log(0.8)
        self.rot = torch.zeros(max_gaussians, 4, device=device, dtype=torch.float32)
        self.rot[:, 0] = 1.0  # Identity quaternion
        self.opacity_raw = torch.full((max_gaussians, 1), 0.8, device=device, dtype=torch.float32)
        self.sem = torch.zeros(max_gaussians, sem_dim, device=device, dtype=torch.float32)
        
        # Pre-allocate covariance cache (updated lazily)
        self._cov_cache = None
        self._cov_valid = False

    def __len__(self) -> int:
        return self._N

    @property
    def scales(self) -> torch.Tensor:
        """Actual scales: exp(log_scale)."""
        return torch.exp(self.scale_raw[:self._N])

    @property
    def opacities(self) -> torch.Tensor:
        """Actual opacities: sigmoid(raw)."""
        return torch.sigmoid(self.opacity_raw[:self._N])

    @property
    def covariances(self) -> torch.Tensor:
        """3D covariance matrices: Σ = R S Sᵀ Rᵀ."""
        if self._cov_cache is None or not self._cov_valid:
            self._cov_cache = self._compute_covariances()
            self._cov_valid = True
        return self._cov_cache

    def _compute_covariances(self) -> torch.Tensor:
        """Compute all covariance matrices in batch."""
        N = self._N
        if N == 0:
            return torch.zeros(0, 3, 3, device=self.device)

        # Quaternion to rotation
        q = self.rot[:N] / (torch.norm(self.rot[:N], dim=1, keepdim=True) + 1e-10)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(N, 3, 3, device=self.device)
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)
        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w*x)
        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)

        # Scale matrix (diagonal), then RS = R @ S
        s = self.scales
        RS = torch.zeros(N, 3, 3, device=self.device)
        RS[:, :, 0] = R[:, :, 0] * s[:, 0:1]
        RS[:, :, 1] = R[:, :, 1] * s[:, 1:2]
        RS[:, :, 2] = R[:, :, 2] * s[:, 2:3]

        # Σ = RS @ RSᵀ
        cov = RS @ RS.transpose(1, 2)
        return cov

    def add(self, positions: np.ndarray,
            colors: np.ndarray = None,
            semantics: np.ndarray = None) -> int:
        """
        Add Gaussians from numpy arrays.
        
        Args:
            positions: [K, 3] 3D positions
            colors: [K, 3] RGB colors (0-1)
            semantics: [K, D] semantic features
        
        Returns:
            Number added
        """
        k = positions.shape[0]
        add_n = min(k, self._cap - self._N)
        if add_n <= 0:
            return 0

        s = self._N
        e = s + add_n
        self.xyz[s:e] = torch.from_numpy(positions[:add_n]).to(self.device)
        if colors is not None:
            self.rgb[s:e] = torch.from_numpy(
                np.clip(colors[:add_n], 0, 1)).to(self.device)
        if semantics is not None:
            self.sem[s:e] = torch.from_numpy(
                semantics[:add_n, :self.sem_dim]).to(self.device)
        self._N += add_n
        self._cov_valid = False
        return add_n

    def add_tensor(self, xyz: torch.Tensor,
                   rgb: torch.Tensor = None,
                   sem: torch.Tensor = None):
        """Add Gaussians directly from CUDA tensors."""
        k = xyz.shape[0]
        add_n = min(k, self._cap - self._N)
        if add_n <= 0:
            return
        s, e = self._N, self._N + add_n
        self.xyz[s:e] = xyz[:add_n].to(self.device, dtype=torch.float32)
        if rgb is not None:
            self.rgb[s:e] = rgb[:add_n].to(self.device, dtype=torch.float32)
        if sem is not None:
            self.sem[s:e] = sem[:add_n, :self.sem_dim].to(self.device, dtype=torch.float32)
        self._N += add_n
        self._cov_valid = False

    def prune(self, keep_mask: torch.Tensor):
        """Prune Gaussians by boolean mask."""
        keep_n = int(keep_mask.sum().item())
        if keep_n < self._N:
            for attr in ['xyz', 'rgb', 'scale_raw', 'rot', 'opacity_raw', 'sem']:
                arr = getattr(self, attr)
                arr[:keep_n] = arr[:self._N][keep_mask]
            self._N = keep_n
            self._cov_valid = False

    def prune_by_opacity(self, min_opacity: float = 0.05):
        """Remove Gaussians with opacity below threshold."""
        opac = self.opacities.flatten()
        keep = opac > min_opacity
        self.prune(keep)

    def pack(self) -> Dict[str, torch.Tensor]:
        """Pack Gaussian data for rendering."""
        n = self._N
        return {
            'xyz': self.xyz[:n],
            'rgb': self.rgb[:n],
            'scale': self.scales,
            'opacity': self.opacities,
            'cov': self.covariances,
            'rot': self.rot[:n],
            'sem': self.sem[:n] if self.sem_dim > 0 else None
        }

    def export_ply(self, path: str, semantic_highlight: bool = True):
        """
        Export Gaussians as PLY file for Open3D visualization.
        
        This enables 3D inspection of the Gaussian cloud after SA-AGD
        density control, showing which Gaussians were added at semantic
        boundaries.
        
        Args:
            path: Output .ply file path
            semantic_highlight: If True, color-code Gaussians by
                semantic cluster membership for boundary visualization
        """
        import struct
        
        n = self._N
        xyz_np = self.xyz[:n].cpu().numpy()
        rgb_np = (self.rgb[:n].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        opacity_np = self.opacities.cpu().numpy()
        
        # If semantic features exist, highlight boundaries
        if semantic_highlight and self.sem_dim > 0:
            sem_np = self.sem[:n].cpu().numpy()
            # Use semantic feature magnitude as cluster indicator
            sem_mag = np.linalg.norm(sem_np, axis=1)
            sem_mag = (sem_mag - sem_mag.min()) / (sem_mag.max() - sem_mag.min() + 1e-8)
            
            # Blend cluster color with original
            cluster_colors = np.zeros_like(rgb_np, dtype=np.float32)
            for c in range(3):
                cluster_colors[:, c] = sem_mag * 255.0
            
            # 50% blend with semantic boundary highlight
            rgb_np = (rgb_np.astype(np.float32) * 0.5 + cluster_colors * 0.5).clip(0, 255).astype(np.uint8)
        
        # Write PLY binary
        with open(path, 'wb') as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {n}\n".encode())
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
            f.write(b"property float opacity\n")
            f.write(b"end_header\n")
            
            # Vertex data
            for i in range(n):
                f.write(struct.pack('<fffBBBf',
                    float(xyz_np[i, 0]), float(xyz_np[i, 1]), float(xyz_np[i, 2]),
                    int(rgb_np[i, 0]), int(rgb_np[i, 1]), int(rgb_np[i, 2]),
                    float(opacity_np[i])))
        
        print(f"  [PLY Export] {n} Gaussians → {path}")

    def chamfer_distance(self, other: 'GaussianCloudCUDA') -> float:
        """
        Compute Chamfer Distance between two Gaussian clouds.
        
        CD(A,B) = 1/|A| Σ_{a∈A} min_{b∈B} ||a-b||² + 1/|B| Σ_{b∈B} min_{a∈A} ||b-a||²
        
        This quantifies the geometric precision improvement from SA-AGD
        by comparing position distributions before vs after densification.
        
        From 3DGS-Survey (Chen & Wang, 2026) method-002 evaluation.
        
        Args:
            other: Another GaussianCloudCUDA to compare against
            
        Returns:
            chamfer_dist: Average Chamfer distance (lower = more similar)
        """
        n_a = self._N
        n_b = other._N
        
        if n_a == 0 or n_b == 0:
            return float('inf')
        
        a_xyz = self.xyz[:n_a]  # [n_a, 3]
        b_xyz = other.xyz[:n_b]  # [n_b, 3]
        
        # Compute pairwise distances using batched GPU ops
        # dist[i,j] = ||a_i - b_j||^2
        # Strategy: compute in chunks for memory efficiency
        
        chunk_size = 4096
        
        # A → B minimum distances
        a_to_b_min = torch.full((n_a,), float('inf'), device=self.device)
        for a_start in range(0, n_a, chunk_size):
            a_end = min(a_start + chunk_size, n_a)
            a_chunk = a_xyz[a_start:a_end]  # [chunk, 3]
            # dist^2 = |a|^2 + |b|^2 - 2 a·b^T
            a_sq = (a_chunk ** 2).sum(dim=1, keepdim=True)  # [chunk, 1]
            b_sq = (b_xyz ** 2).sum(dim=1).unsqueeze(0)  # [1, n_b]
            ab = a_chunk @ b_xyz.T  # [chunk, n_b]
            dist_chunk = a_sq + b_sq - 2 * ab
            a_to_b_min[a_start:a_end] = dist_chunk.min(dim=1).values
        
        # B → A minimum distances
        b_to_a_min = torch.full((n_b,), float('inf'), device=self.device)
        for b_start in range(0, n_b, chunk_size):
            b_end = min(b_start + chunk_size, n_b)
            b_chunk = b_xyz[b_start:b_end]
            b_sq = (b_chunk ** 2).sum(dim=1, keepdim=True)
            a_sq = (a_xyz ** 2).sum(dim=1).unsqueeze(0)
            ba = b_chunk @ a_xyz.T
            dist_chunk = b_sq + a_sq - 2 * ba
            b_to_a_min[b_start:b_end] = dist_chunk.min(dim=1).values
        
        cd = (a_to_b_min.mean() + b_to_a_min.mean()).item()
        return float(cd)

    def get_numpy_map(self) -> Dict[str, np.ndarray]:
        """Export Gaussian map as numpy arrays (for saving/visualization)."""
        return {
            'xyz': self.xyz[:self._N].cpu().numpy(),
            'rgb': self.rgb[:self._N].cpu().numpy(),
            'scale': self.scales.cpu().numpy(),
            'opacity': self.opacities.cpu().numpy(),
            'cov': self.covariances.cpu().numpy(),
            'sem': self.sem[:self._N].cpu().numpy()
        }

    def save(self, path: str):
        """Save Gaussian model to file."""
        data = self.get_numpy_map()
        torch.save({
            'xyz': self.xyz[:self._N].cpu(),
            'rgb': self.rgb[:self._N].cpu(),
            'scale_raw': self.scale_raw[:self._N].cpu(),
            'rot': self.rot[:self._N].cpu(),
            'opacity_raw': self.opacity_raw[:self._N].cpu(),
            'sem': self.sem[:self._N].cpu(),
            '_N': self._N,
            'sem_dim': self.sem_dim
        }, path)

    def load(self, path: str):
        """Load Gaussian model from file."""
        data = torch.load(path, map_location=self.device)
        self._N = data['_N']
        self.xyz[:self._N] = data['xyz']
        self.rgb[:self._N] = data['rgb']
        self.scale_raw[:self._N] = data['scale_raw']
        self.rot[:self._N] = data['rot']
        self.opacity_raw[:self._N] = data['opacity_raw']
        self.sem[:self._N, :data['sem'].shape[1]] = data['sem']
        self._cov_valid = False


def create_test_scene_cuda(device=None, n_gaussians: int = 1000) -> GaussianCloudCUDA:
    """
    Create a synthetic test scene with spheres and boxes.
    Generates more realistic geometry than simple random points.
    """
    import numpy as np
    
    gc = GaussianCloudCUDA(max_gaussians=n_gaussians * 2, device=device)
    
    m = n_gaussians // 3
    
    # Sphere (blueish)
    phi = np.random.uniform(0, np.pi, m)
    theta = np.random.uniform(0, 2*np.pi, m)
    r = 2.0 + np.random.uniform(-0.3, 0.3, m)
    sphere = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ]).astype(np.float32)
    sphere_col = np.tile(np.array([0.2, 0.6, 0.9], dtype=np.float32), (m, 1))
    
    # Box (reddish)
    box = np.random.uniform(-1.5, 1.5, (m, 3)).astype(np.float32)
    box[:, 0] += 4.0
    box[:, 2] -= 1.0
    box_col = np.tile(np.array([0.9, 0.3, 0.3], dtype=np.float32), (m, 1))
    
    # Floor (greenish)
    floor = np.column_stack([
        np.random.uniform(-5, 5, m),
        np.full(m, -3.0, dtype=np.float32),
        np.random.uniform(-5, 5, m)
    ])
    floor_col = np.tile(np.array([0.3, 0.8, 0.3], dtype=np.float32), (m, 1))
    
    # Wall (yellowish)
    wall = np.column_stack([
        np.full(m, -5.0, dtype=np.float32),
        np.random.uniform(-3, 3, m),
        np.random.uniform(-5, 5, m)
    ])
    wall_col = np.tile(np.array([0.9, 0.9, 0.3], dtype=np.float32), (m, 1))
    
    pos = np.vstack([sphere, box, floor, wall]).astype(np.float32)
    col = np.vstack([sphere_col, box_col, floor_col, wall_col]).astype(np.float32)
    
    gc.add(pos, col)
    return gc