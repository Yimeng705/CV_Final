"""
Camera Models for 3DGS-SLAM CUDA
=================================
Supporting both pinhole and generic camera models.

Architecture:
- PinholeCamera: Standard perspective projection (default for KITTI-360/EuRoC/TUM)
- GenericCamera: Ray-based implicit camera model (for MASt3R-SLAM compatibility)
- CameraPose: SE(3) + scale representation with Lie algebra operations
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Intrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480


class PinholeCamera:
    """
    Pinhole camera model with SE(3) pose.
    
    Coordinate convention:
    - World: right-handed, Y-up (KITTI) or Z-up (common CV)
    - Camera: right-handed, Z-forward, X-right, Y-down
    - Projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
    
    Supports:
    - Intrinsic calibration
    - SE(3) pose representation
    - Projection and back-projection
    - Look-at pose generation
    """

    def __init__(self,
                 fx: float = 525.0,
                 fy: float = 525.0,
                 cx: float = 319.5,
                 cy: float = 239.5,
                 width: int = 640,
                 height: int = 480):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        # Default: identity pose
        self.R = np.eye(3, dtype=np.float32)
        self.t = np.zeros((3, 1), dtype=np.float32)
    
    @classmethod
    def from_kitti360(cls):
        """KITTI-360 camera intrinsics (Perspective camera, image_00)."""
        return cls(
            fx=552.55426, fy=552.55426,
            cx=682.04945, cy=238.76955,
            width=1408, height=376
        )
    
    @classmethod
    def from_tum_rgbd(cls):
        """TUM RGB-D dataset intrinsics (Freburg1)."""
        return cls(
            fx=517.3, fy=516.5,
            cx=318.6, cy=255.3,
            width=640, height=480
        )
    
    @classmethod
    def from_euroc(cls):
        """EuRoC MAV dataset intrinsics (Machine Hall 01)."""
        return cls(
            fx=458.654, fy=457.296,
            cx=367.215, cy=248.375,
            width=752, height=480
        )
    
    @property
    def K(self) -> np.ndarray:
        """Intrinsic matrix K."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K
    
    @property
    def pose(self) -> np.ndarray:
        """SE(3) pose as 4x4 matrix [R|t; 0|1]."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.R
        T[:3, 3:4] = self.t.reshape(3, 1)
        return T
    
    def set_pose(self, R: np.ndarray, t: np.ndarray):
        """Set camera pose from rotation matrix and translation vector."""
        self.R = R.astype(np.float32).reshape(3, 3)
        self.t = t.astype(np.float32).reshape(3, 1)
    
    def set_pose_matrix(self, T: np.ndarray):
        """Set camera pose from 4x4 SE(3) matrix."""
        self.R = T[:3, :3].astype(np.float32)
        self.t = T[:3, 3:4].astype(np.float32)
    
    def project(self, pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points from world to image coordinates.
        
        Args:
            pts_world: [N, 3] points in world frame
        
        Returns:
            u, v: [N] pixel coordinates
            depth: [N] depth in camera frame
        """
        pts_cam = (self.R @ pts_world.T + self.t).T
        z = pts_cam[:, 2]
        u = self.fx * pts_cam[:, 0] / (z + 1e-10) + self.cx
        v = self.fy * pts_cam[:, 1] / (z + 1e-10) + self.cy
        return u, v, z
    
    def back_project(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Back-project pixel coordinates to 3D world coordinates.
        
        Args:
            u, v: pixel coordinates
            depth: depth in camera frame
        
        Returns:
            pts_world: [N, 3] points in world frame
        """
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth
        pts_cam = np.column_stack([x_cam, y_cam, z_cam])
        pts_world = (self.R.T @ (pts_cam.T - self.t)).T
        return pts_world
    
    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates."""
        return (-self.R.T @ self.t).flatten()


@dataclass
class CameraPose:
    """
    SE(3) + scale camera pose for factor graph optimization.
    
    Used in MASt3R-Fusion's hybrid factor graph design:
    - SE(3) component for IMU/GNSS metric-scale measurements
    - Scale component for visual Sim(3) constraints
    
    Lie algebra mapping (homogeneous to SE(3)×R):
    Λ = diag(s·I, s, 1) maps SE(3)+scale tangent space to Sim(3)
    """
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    s: float = 1.0  # scale factor
    
    def __post_init__(self):
        self.R = self.R.astype(np.float32).reshape(3, 3)
        self.t = self.t.astype(np.float32).reshape(3, 1)
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix (SE(3) only)."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.R
        T[:3, 3:4] = self.t
        return T
    
    def transform(self, pts: np.ndarray) -> np.ndarray:
        """Transform points by this pose (world→camera or camera→world)."""
        return (self.R @ pts.T + self.t).T
    
    @classmethod
    def from_se3(cls, T: np.ndarray, scale: float = 1.0):
        """Create CameraPose from 4x4 SE(3) matrix and scale."""
        return cls(R=T[:3, :3], t=T[:3, 3:4], s=scale)
    
    @classmethod
    def identity(cls):
        """Identity pose."""
        return cls(R=np.eye(3, dtype=np.float32),
                   t=np.zeros((3, 1), dtype=np.float32), s=1.0)


def look_at(camera_pos: np.ndarray,
            target: np.ndarray,
            up: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate camera rotation and translation for a look-at view.
    
    Args:
        camera_pos: [3] camera position in world
        target: [3] look-at target in world
        up: [3] up vector (default: Y-up)
    
    Returns:
        R: 3x3 rotation matrix (world→camera)
        t: 3x1 translation vector (camera center in camera frame)
    """
    if up is None:
        up = np.array([0., 1., 0.], dtype=np.float32)
    else:
        up = np.array(up, dtype=np.float32)
    
    camera_pos = np.array(camera_pos, dtype=np.float32).flatten()
    target = np.array(target, dtype=np.float32).flatten()
    
    # Camera Z-axis (forward direction = from camera to target)
    z = camera_pos - target
    z = z / (np.linalg.norm(z) + 1e-10)
    
    # Camera X-axis (right)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-10)
    
    # Camera Y-axis (down)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-10)
    
    # R transforms world→camera (R @ world_point + t = camera_point)
    R = np.column_stack([x, y, z]).T.astype(np.float32)
    t = (-R @ camera_pos).reshape(3, 1).astype(np.float32)
    
    return R, t


def generate_helical_trajectory(n_poses: int = 50,
                                radius: float = 5.0,
                                height_range: Tuple[float, float] = (-1.0, 3.0),
                                target: np.ndarray = None) -> list:
    """
    Generate a helical camera trajectory for synthetic evaluation.
    
    Args:
        n_poses: Number of camera poses
        radius: Radius of the helix
        height_range: Min and max height
        target: Look-at target (default: origin)
    
    Returns:
        List of (R, t) tuples
    """
    if target is None:
        target = np.array([0., 1., 0.], dtype=np.float32)
    
    poses = []
    for i in range(n_poses):
        angle = 2 * np.pi * i / n_poses
        height = height_range[0] + (height_range[1] - height_range[0]) * i / n_poses
        
        pos = np.array([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ], dtype=np.float32)
        
        R, t = look_at(pos, target, np.array([0., 1., 0.]))
        poses.append((R, t))
    
    return poses