"""
Real-world SLAM Dataset Loaders for gs_slam_cuda
==================================================
Supports standard SLAM benchmarks on Linux with RTX 3060 8GB.

Datasets:
- TUM RGB-D: Standard indoor SLAM benchmark
- EuRoC MAV: UAV VI-SLAM benchmark  
- Replica: Dense reconstruction benchmark

From 3DGS-Survey (Chen & Wang, 2026) method-002 evaluation framework.
"""

import os
import sys
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Attempt PIL import for image loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class DatasetFrame:
    """Single frame from a SLAM dataset."""
    idx: int
    rgb: np.ndarray          # [H, W, 3] uint8 RGB
    depth: Optional[np.ndarray]  # [H, W] float32 depth (meters)
    gt_pose: Optional[np.ndarray]  # [4, 4] ground truth camera-to-world
    timestamp: float
    intrinsics: Dict[str, float]   # fx, fy, cx, cy


DATASET_CONFIGS = {
    'tum_fr1_desk': {
        'type': 'tum',
        'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3,
        'width': 640, 'height': 480,
        'fps': 30,
        'description': 'TUM RGB-D fr1/desk - office desk scene'
    },
    'tum_fr2_xyz': {
        'type': 'tum',
        'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7,
        'width': 640, 'height': 480,
        'fps': 30,
        'description': 'TUM RGB-D fr2/xyz - simple translation'
    },
    'tum_fr3_long_office': {
        'type': 'tum',
        'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6,
        'width': 640, 'height': 480,
        'fps': 30,
        'description': 'TUM RGB-D fr3/long_office - long trajectory'
    },
    'euroc_mh01': {
        'type': 'euroc',
        'fx': 458.654, 'fy': 457.296, 'cx': 367.215, 'cy': 248.375,
        'width': 752, 'height': 480,
        'fps': 20,
        'description': 'EuRoC MAV MH_01_easy - Machine Hall'
    },
    'euroc_v101': {
        'type': 'euroc',
        'fx': 458.654, 'fy': 457.296, 'cx': 367.215, 'cy': 248.375,
        'width': 752, 'height': 480,
        'fps': 20,
        'description': 'EuRoC MAV V1_01_easy - Vicon Room'
    },
    'replica_room0': {
        'type': 'replica',
        'fx': 600.0, 'fy': 600.0, 'cx': 599.5, 'cy': 339.5,
        'width': 1200, 'height': 680,
        'fps': 30,
        'description': 'Replica room0 - small room'
    },
    'synthetic': {
        'type': 'synthetic',
        'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5,
        'width': 640, 'height': 480,
        'fps': 30,
        'description': 'Synthetic helical trajectory (default)'
    }
}


class TUMDataset:
    """
    TUM RGB-D dataset loader.
    
    Directory structure:
    <path>/<sequence>/
        rgb.txt        - RGB timestamps + filenames
        depth.txt      - depth timestamps + filenames  
        groundtruth.txt - ground truth trajectory
        rgb/           - RGB images (*.png)
        depth/         - depth images (*.png)
    
    From 3DGS-Survey evaluation framework.
    """
    
    def __init__(self, 
                 data_path: str,
                 config_name: str = 'tum_fr1_desk',
                 start_frame: int = 0,
                 max_frames: int = 500):
        """
        Args:
            data_path: Root path to TUM RGB-D dataset
            config_name: Dataset configuration key
            start_frame: Starting frame index
            max_frames: Maximum number of frames to load
        """
        self.config = DATASET_CONFIGS.get(config_name, DATASET_CONFIGS['tum_fr1_desk'])
        self.data_path = Path(data_path)
        self.max_frames = max_frames
        
        # Extract sequence name from config (e.g., 'fr1/desk' from 'tum_fr1_desk')
        parts = config_name.split('_', 1)
        self.sequence = '/'.join(parts[1].split('_', 1)) if len(parts) > 1 else parts[0]
        
        self.rgb_files = []
        self.depth_files = []
        self.gt_poses = []  # list of (timestamp, xyz, quaternion)
        
        self._load_associations()
        
        # Slice frames
        end = min(start_frame + max_frames, len(self.rgb_files))
        self.rgb_files = self.rgb_files[start_frame:end]
        self.depth_files = self.depth_files[start_frame:end]
        if self.gt_poses:
            self.gt_poses = self.gt_poses[start_frame:end]
    
    def _load_associations(self):
        """Load and associate RGB, depth, and GT data."""
        seq_path = self.data_path / self.sequence
        
        # Load RGB timestamps
        rgb_file = seq_path / 'rgb.txt'
        if rgb_file.exists():
            with open(rgb_file, 'r') as f:
                lines = f.readlines()[3:]  # Skip header
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ts = float(parts[0])
                    fname = parts[1]
                    self.rgb_files.append((ts, str(seq_path / fname)))
        
        # Load depth timestamps
        depth_file = seq_path / 'depth.txt'
        if depth_file.exists():
            with open(depth_file, 'r') as f:
                lines = f.readlines()[3:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    ts = float(parts[0])
                    fname = parts[1]
                    self.depth_files.append((ts, str(seq_path / fname)))
        
        # Load ground truth
        gt_file = seq_path / 'groundtruth.txt'
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                lines = f.readlines()[3:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 8:
                    ts = float(parts[0])
                    t = np.array([float(x) for x in parts[1:4]])
                    q = np.array([float(x) for x in parts[4:8]])  # x,y,z,w
                    self.gt_poses.append((ts, t, q))
    
    def __len__(self) -> int:
        return len(self.rgb_files)
    
    def __getitem__(self, idx: int) -> DatasetFrame:
        """Load and return a single frame."""
        ts, rgb_path = self.rgb_files[idx]
        
        # Load RGB
        rgb = None
        if HAS_PIL and os.path.exists(rgb_path):
            img = Image.open(rgb_path)
            rgb = np.array(img.convert('RGB'))
        
        # Load depth
        depth = None
        if idx < len(self.depth_files):
            _, depth_path = self.depth_files[idx]
            if HAS_PIL and os.path.exists(depth_path):
                depth_img = Image.open(depth_path)
                depth = np.array(depth_img).astype(np.float32) / 5000.0  # TUM scale
        
        # GT pose
        gt_pose = None
        if idx < len(self.gt_poses):
            _, t, q = self.gt_poses[idx]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = self._quat_to_rot(q)
            T[:3, 3] = t
            gt_pose = T
        
        return DatasetFrame(
            idx=idx,
            rgb=rgb,
            depth=depth,
            gt_pose=gt_pose,
            timestamp=ts,
            intrinsics={
                'fx': self.config['fx'],
                'fy': self.config['fy'],
                'cx': self.config['cx'],
                'cy': self.config['cy'],
                'width': self.config['width'],
                'height': self.config['height']
            }
        )
    
    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [x,y,z,w] to rotation matrix."""
        x, y, z, w = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ], dtype=np.float32)
        return R


class EuRoCDataset:
    """
    EuRoC MAV dataset loader.
    
    Directory structure:
    <path>/<sequence>/
        mav0/
            cam0/data/*.png
            cam0/sensor.yaml
            state_groundtruth_estimate0/data.csv
    
    From 3DGS-Survey evaluation framework.
    """
    
    def __init__(self,
                 data_path: str,
                 config_name: str = 'euroc_mh01',
                 start_frame: int = 0,
                 max_frames: int = 500):
        self.config = DATASET_CONFIGS.get(config_name, DATASET_CONFIGS['euroc_mh01'])
        self.data_path = Path(data_path)
        
        parts = config_name.split('_', 1)
        self.sequence = parts[1].upper() if len(parts) > 1 else 'MH_01_easy'
        
        self.image_files = []
        self.gt_poses = []  # (timestamp, T_cam_world)
        
        self._load_data()
        
        end = min(start_frame + max_frames, len(self.image_files))
        self.image_files = self.image_files[start_frame:end]
        if self.gt_poses:
            self.gt_poses = self._interpolate_gt(self.gt_poses, start_frame, end)
    
    def _load_data(self):
        """Load image list and ground truth."""
        seq_path = self.data_path / self.sequence / 'mav0'
        cam_path = seq_path / 'cam0' / 'data'
        
        if cam_path.exists():
            img_files = sorted(cam_path.glob('*.png'))
            self.image_files = [(i, str(f)) for i, f in enumerate(img_files)]
        
        gt_csv = seq_path / 'state_groundtruth_estimate0' / 'data.csv'
        if gt_csv.exists():
            import csv
            with open(gt_csv, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 17:
                        ts = int(row[0])
                        # p_RS_R_x [m], p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z
                        t = np.array([float(row[1]), float(row[2]), float(row[3])])
                        q = np.array([float(row[5]), float(row[6]), float(row[7]), float(row[4])])
                        T = np.eye(4, dtype=np.float32)
                        T[:3, :3] = TUMDataset._quat_to_rot(q)
                        T[:3, 3] = t
                        self.gt_poses.append((ts * 1e-9, T))
    
    def _interpolate_gt(self, gt_list, start, end):
        """Simplified: just slice GT for matching frame count."""
        n = end - start
        if len(gt_list) >= n:
            return gt_list[:n]
        return gt_list
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> DatasetFrame:
        _, img_path = self.image_files[idx]
        
        rgb = None
        if HAS_PIL:
            img = Image.open(img_path)
            rgb = np.array(img.convert('RGB'))
        
        gt_pose = None
        if idx < len(self.gt_poses):
            _, gt_pose = self.gt_poses[idx]
        
        return DatasetFrame(
            idx=idx,
            rgb=rgb,
            depth=None,
            gt_pose=gt_pose,
            timestamp=float(idx) / self.config['fps'],
            intrinsics={
                'fx': self.config['fx'],
                'fy': self.config['fy'],
                'cx': self.config['cx'],
                'cy': self.config['cy'],
                'width': self.config['width'],
                'height': self.config['height']
            }
        )


class ReplicaDataset:
    """
    Replica dataset loader (dense reconstruction benchmark).
    
    From 3DGS-Survey (Chen & Wang, 2026) method-002.
    
    Note: Replica uses rendered RGB-D sequences with GT camera poses.
    """
    
    def __init__(self,
                 data_path: str,
                 config_name: str = 'replica_room0',
                 start_frame: int = 0,
                 max_frames: int = 500):
        self.config = DATASET_CONFIGS.get(config_name, DATASET_CONFIGS['replica_room0'])
        self.data_path = Path(data_path)
        self.sequence = config_name.split('_', 1)[1] if '_' in config_name else 'room0'
        self.max_frames = max_frames
        
        # Replica typically provides traj.txt + images
        self._load_trajectory()
    
    def _load_trajectory(self):
        """Load camera trajectory from traj.txt."""
        seq_path = self.data_path / self.sequence
        traj_file = seq_path / 'traj.txt'
        self.poses = []
        
        if traj_file.exists():
            with open(traj_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 12:
                        T = np.array([float(x) for x in parts[:12]]).reshape(3, 4)
                        T_full = np.eye(4, dtype=np.float32)
                        T_full[:3, :] = T
                        self.poses.append(T_full)
        
        self.image_dir = seq_path / 'results'
        self.rgb_files = sorted(self.image_dir.glob('frame*.jpg')) if self.image_dir.exists() else []
        
        # Limit to available data
        n = min(len(self.rgb_files), self.max_frames)
        self.rgb_files = self.rgb_files[:n]
        self.poses = self.poses[:n]
    
    def __len__(self) -> int:
        return min(len(self.rgb_files), len(self.poses))
    
    def __getitem__(self, idx: int) -> DatasetFrame:
        rgb = None
        if idx < len(self.rgb_files) and HAS_PIL:
            img = Image.open(self.rgb_files[idx])
            rgb = np.array(img.convert('RGB'))
        
        gt_pose = self.poses[idx] if idx < len(self.poses) else None
        
        return DatasetFrame(
            idx=idx,
            rgb=rgb,
            depth=None,
            gt_pose=gt_pose,
            timestamp=float(idx) / self.config['fps'],
            intrinsics={
                'fx': self.config['fx'],
                'fy': self.config['fy'],
                'cx': self.config['cx'],
                'cy': self.config['cy'],
                'width': self.config['width'],
                'height': self.config['height']
            }
        )


class SyntheticDataset:
    """
    Synthetic helical trajectory dataset (for testing/debugging).
    Uses procedural scene geometry and predetermined camera path.
    """
    
    def __init__(self, 
                 n_frames: int = 50,
                 config_name: str = 'synthetic'):
        self.config = DATASET_CONFIGS.get(config_name, DATASET_CONFIGS['synthetic'])
        self.n_frames = n_frames
        self._generate()
    
    def _generate(self):
        """Generate synthetic helical trajectory."""
        from ..core.camera import generate_helical_trajectory, look_at
        self.camera_poses = generate_helical_trajectory(
            n_poses=self.n_frames, radius=8.0, height_range=(-2.0, 4.0)
        )
    
    def __len__(self) -> int:
        return len(self.camera_poses)
    
    def __getitem__(self, idx: int) -> DatasetFrame:
        R, t = self.camera_poses[idx]
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()[:3]
        
        return DatasetFrame(
            idx=idx,
            rgb=None,
            depth=None,
            gt_pose=T,
            timestamp=float(idx) / self.config['fps'],
            intrinsics={
                'fx': self.config['fx'],
                'fy': self.config['fy'],
                'cx': self.config['cx'],
                'cy': self.config['cy'],
                'width': self.config['width'],
                'height': self.config['height']
            }
        )


def create_dataloader(dataset_name: str = 'synthetic',
                       data_path: str = None,
                       max_frames: int = 500,
                       batch_size: int = 1) -> Tuple[object, Dict]:
    """
    Factory function to create the appropriate dataset.
    
    Args:
        dataset_name: One of DATASET_CONFIGS keys
        data_path: Path to dataset root
        max_frames: Maximum frames to load
        batch_size: Batch size (always 1 for SLAM)
    
    Returns:
        dataset, config dict
    """
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['synthetic'])
    ds_type = config.get('type', 'synthetic')
    
    if ds_type == 'tum':
        if data_path is None:
            raise ValueError("data_path required for TUM dataset")
        dataset = TUMDataset(data_path, dataset_name, max_frames=max_frames)
    elif ds_type == 'euroc':
        if data_path is None:
            raise ValueError("data_path required for EuRoC dataset")
        dataset = EuRoCDataset(data_path, dataset_name, max_frames=max_frames)
    elif ds_type == 'replica':
        if data_path is None:
            raise ValueError("data_path required for Replica dataset")
        dataset = ReplicaDataset(data_path, dataset_name, max_frames=max_frames)
    else:
        dataset = SyntheticDataset(n_frames=max_frames, config_name=dataset_name)
    
    return dataset, config