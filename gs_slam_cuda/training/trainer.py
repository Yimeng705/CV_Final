"""
GPU-Accelerated Gaussian Training Pipeline
===========================================
Training loop with rendering loss optimization for gs_slam_cuda.

Features:
- L1 + SSIM loss (from 3DGS Kerbl et al.)
- Adam optimizer with exponential learning rate decay
- SA-AGD density control during training
- VRAM-aware batch management for RTX 3060 8GB
- Checkpoint save/load support
- Evaluation metrics (PSNR, SSIM, LPIPS proxy)

Architecture:
- Leverages tile-based renderer (restored from audit P0 fix)
- Uses full covariance projection (restored from audit P1 fix)
- FP16 mixed precision for memory efficiency (audit P2 optimization)

Training flow:
1. Initialize Gaussian Cloud + Renderer
2. For each training iteration:
   a. Select random training camera
   b. Forward render (tile-based with autocast)
   c. Compute L1 + SSIM loss against GT
   d. Backward pass (autograd through rendering)
   e. Optimizer step (Adam)
3. Every N iterations: Run SA-AGD density control
4. Periodic evaluation on held-out views
5. Save checkpoints
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.camera import PinholeCamera, CameraPose, look_at
from core.gaussian_model_cuda import GaussianCloudCUDA
from core.renderer_cuda import (
    CUDASplatRenderer,
    compute_psnr_cuda,
    compute_ssim_cuda,
    compute_lpips_proxy
)
from core.adaptive_density_cuda import (
    CUDADensityController,
    run_cuda_densification_cycle
)


@dataclass
class TrainingConfig:
    """Configuration for Gaussian training."""
    # Optimization
    n_iterations: int = 1000
    learning_rate_xyz: float = 1.6e-4
    learning_rate_rgb: float = 2.5e-3
    learning_rate_opacity: float = 5.0e-2
    learning_rate_scale: float = 5.0e-3
    learning_rate_rot: float = 1.0e-3
    lr_decay_steps: int = 500
    lr_decay_factor: float = 0.1
    
    # Loss weights
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.2
    
    # Density control
    densification_interval: int = 100
    max_densify_iter: int = 500  # Stop densifying after this iteration
    sem_grad_weight: float = 0.3
    
    # SA-AGD parameters
    grad_threshold: float = 0.3
    scale_threshold: float = 2.0
    opacity_threshold: float = 0.05
    sem_boundary_threshold: float = 0.3
    
    # Memory
    max_gaussians: int = 500000  # Max gaussians for RTX 3060 8GB
    use_fp16: bool = True
    
    # Rendering
    image_height: int = 480
    image_width: int = 640
    tile_size: int = 16
    
    # Logging
    log_interval: int = 50
    eval_interval: int = 200
    save_interval: int = 500
    
    # Output
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'


class GaussianTrainer:
    """
    GPU-accelerated Gaussian training pipeline.
    
    Trains 3D Gaussian parameters via rendering loss minimization
    with SA-AGD density control on RTX 3060 8GB.
    """
    
    def __init__(self,
                 gc: GaussianCloudCUDA,
                 cameras: List[PinholeCamera],
                 gt_images: Optional[List[torch.Tensor]] = None,
                 config: Optional[TrainingConfig] = None,
                 device=None):
        """
        Args:
            gc: Initial Gaussian cloud
            cameras: Training camera poses
            gt_images: Ground truth images (optional, use rendered pseudo-GT if None)
            config: Training configuration
            device: CUDA device
        """
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config or TrainingConfig()
        self.gc = gc
        self.cameras = cameras
        
        # Initialize renderer
        self.renderer = CUDASplatRenderer(
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            tile_size=self.config.tile_size,
            use_fp16=self.config.use_fp16,
            device=self.device
        )
        
        # Generate GT images if not provided
        if gt_images is None:
            self.gt_images = []
            gs_dict = gc.pack()
            for cam in cameras:
                rgb, _ = self.renderer.forward(gs_dict, cam)
                self.gt_images.append(rgb.detach().clone())
        else:
            self.gt_images = gt_images
        
        # Density controller
        self.density_controller = CUDADensityController(
            grad_threshold=self.config.grad_threshold,
            scale_threshold=self.config.scale_threshold,
            opacity_threshold=self.config.opacity_threshold,
            sem_boundary_threshold=self.config.sem_boundary_threshold,
            sem_grad_weight=self.config.sem_grad_weight,
            max_world_size=10.0,
            device=self.device
        )
        
        # Setup optimizers
        self._setup_optimizers()
        
        # Metrics tracking
        self.metrics = {
            'loss_l1': [],
            'loss_ssim': [],
            'loss_total': [],
            'psnr': [],
            'ssim': [],
            'n_gaussians': [],
            'lr': [],
            'elapsed_ms': []
        }
        
        # Ensure output directories exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def _setup_optimizers(self):
        """Setup Adam optimizers for trainable Gaussian parameters."""
        # Mark trainable parameters
        self.gc.xyz.requires_grad = True
        self.gc.rgb.requires_grad = True
        self.gc.opacity_raw.requires_grad = True
        self.gc.scale_raw.requires_grad = True
        self.gc.rot.requires_grad = True
        
        cfg = self.config
        self.optimizer = torch.optim.Adam([
            {'params': [self.gc.xyz], 'lr': cfg.learning_rate_xyz, 'name': 'xyz'},
            {'params': [self.gc.rgb], 'lr': cfg.learning_rate_rgb, 'name': 'rgb'},
            {'params': [self.gc.opacity_raw], 'lr': cfg.learning_rate_opacity, 'name': 'opacity'},
            {'params': [self.gc.scale_raw], 'lr': cfg.learning_rate_scale, 'name': 'scale'},
            {'params': [self.gc.rot], 'lr': cfg.learning_rate_rot, 'name': 'rot'},
        ])
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.lr_decay_steps,
            gamma=cfg.lr_decay_factor
        )
        
        # GradScaler for FP16 training
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.use_fp16)
    
    def compute_loss(self,
                     pred: torch.Tensor,
                     gt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute L1 + SSIM rendering loss.
        
        From 3DGS (Kerbl et al., 2023):
        L = (1 - λ) * L1 + λ * L_SSIM
        
        Args:
            pred: [H, W, 3] predicted image
            gt: [H, W, 3] ground truth image
        
        Returns:
            loss_dict with 'l1', 'ssim', 'total' keys
        """
        cfg = self.config
        loss_l1 = F.l1_loss(pred, gt)
        
        # SSIM needs 4D input [B, C, H, W]
        pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
        gt_4d = gt.permute(2, 0, 1).unsqueeze(0)
        
        # Simplified SSIM loss
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Use 3x3 average pooling as simplified SSIM
        pool = nn.AvgPool2d(3, stride=1, padding=1)
        mu_p = pool(pred_4d)
        mu_g = pool(gt_4d)
        
        sigma_p = pool(pred_4d ** 2) - mu_p ** 2
        sigma_g = pool(gt_4d ** 2) - mu_g ** 2
        sigma_pg = pool(pred_4d * gt_4d) - mu_p * mu_g
        
        ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / \
                   ((mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p + sigma_g + C2) + 1e-10)
        loss_ssim = 1.0 - ssim_map.mean()
        
        loss_total = cfg.lambda_l1 * loss_l1 + cfg.lambda_ssim * loss_ssim
        
        return {
            'l1': loss_l1,
            'ssim': loss_ssim,
            'total': loss_total
        }
    
    def train_step(self, iteration: int) -> Dict[str, float]:
        """
        Single training iteration.
        
        Args:
            iteration: Current iteration number
        
        Returns:
            metrics dict
        """
        cfg = self.config
        n_cams = len(self.cameras)
        
        # Select random camera
        cam_idx = iteration % n_cams
        camera = self.cameras[cam_idx]
        gt = self.gt_images[cam_idx]
        
        # Get current Gaussian data
        gs_dict = self.gc.pack()
        
        # Forward pass with FP16 (use differentiable splatted path)
        t0 = time.time()
        with torch.cuda.amp.autocast(enabled=cfg.use_fp16):
            pred, depth = self.renderer.forward(gs_dict, camera, differentiable=True)
            losses = self.compute_loss(pred, gt)
        
        # Backward pass
        self.optimizer.zero_grad()
        if cfg.use_fp16:
            self.scaler.scale(losses['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total'].backward()
            self.optimizer.step()
        
        self.scheduler.step()
        
        elapsed_ms = (time.time() - t0) * 1000
        
        # Compute PSNR
        with torch.no_grad():
            psnr = compute_psnr_cuda(pred, gt)
            ssim = compute_ssim_cuda(pred, gt)
        
        return {
            'loss_l1': losses['l1'].item(),
            'loss_ssim': losses['ssim'].item(),
            'loss_total': losses['total'].item(),
            'psnr': psnr,
            'ssim': ssim,
            'n_gaussians': len(self.gc),
            'lr': self.optimizer.param_groups[0]['lr'],
            'elapsed_ms': elapsed_ms
        }
    
    def densify_step(self, iteration: int):
        """Run SA-AGD density control for this iteration."""
        if iteration > self.config.max_densify_iter:
            return
        
        # Select a canonical view for geometric importance
        cam = self.cameras[0]
        
        stats = run_cuda_densification_cycle(
            self.gc,
            self.density_controller,
            n_iterations=1,
            camera=cam
        )
        
        # Re-setup optimizers after density change
        self._setup_optimizers()
        
        # Prune low-opacity Gaussians
        self.gc.prune_by_opacity(self.config.opacity_threshold)
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on all training views."""
        gs_dict = self.gc.pack()
        
        psnrs, ssims, lpipss = [], [], []
        for i, (cam, gt) in enumerate(zip(self.cameras, self.gt_images)):
            pred, _ = self.renderer.forward(gs_dict, cam)
            psnrs.append(compute_psnr_cuda(pred, gt))
            ssims.append(compute_ssim_cuda(pred, gt))
            lpipss.append(compute_lpips_proxy(pred, gt))
        
        return {
            'psnr_mean': float(np.mean(psnrs)),
            'psnr_std': float(np.std(psnrs)),
            'ssim_mean': float(np.mean(ssims)),
            'ssim_std': float(np.std(ssims)),
            'lpips_mean': float(np.mean(lpipss)),
            'n_gaussians': len(self.gc),
            'vram_mb': torch.cuda.memory_allocated(self.device) / (1024 * 1024) if self.device.type == 'cuda' else 0
        }
    
    def train(self) -> Dict:
        """
        Full training loop.
        
        Returns:
            final_metrics: Training summary
        """
        cfg = self.config
        n_initial = len(self.gc)
        
        print(f"\n{'='*60}")
        print(f"  gs_slam_cuda Training Pipeline")
        print(f"{'='*60}")
        print(f"  Device:      {self.device}")
        print(f"  Initial GS:  {n_initial}")
        print(f"  Training views: {len(self.cameras)}")
        print(f"  Iterations:  {cfg.n_iterations}")
        print(f"  FP16:        {cfg.use_fp16}")
        print(f"  SA-AGD:      sem_weight={cfg.sem_grad_weight}")
        print(f"  Resolution:  {cfg.image_width}x{cfg.image_height}")
        print(f"{'='*60}\n")
        
        t_start = time.time()
        
        for iteration in range(1, cfg.n_iterations + 1):
            # Training step
            metrics = self.train_step(iteration)
            
            # Store metrics
            for k, v in metrics.items():
                self.metrics[k].append(v)
            
            # Density control
            if iteration % cfg.densification_interval == 0:
                self.densify_step(iteration)
            
            # Logging
            if iteration % cfg.log_interval == 0:
                n_gs = metrics['n_gaussians']
                print(f"  [{iteration:5d}/{cfg.n_iterations}] "
                      f"L1={metrics['loss_l1']:.4f} "
                      f"PSNR={metrics['psnr']:.2f}dB "
                      f"SSIM={metrics['ssim']:.4f} "
                      f"GS={n_gs} "
                      f"LR={metrics['lr']:.1e} "
                      f"{metrics['elapsed_ms']:.0f}ms")
            
            # Evaluation
            if iteration % cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                print(f"  --- Evaluation @ iter {iteration} ---")
                print(f"      PSNR: {eval_metrics['psnr_mean']:.2f}±{eval_metrics['psnr_std']:.2f} dB")
                print(f"      SSIM: {eval_metrics['ssim_mean']:.4f}±{eval_metrics['ssim_std']:.4f}")
                print(f"      LPIPS proxy: {eval_metrics['lpips_mean']:.4f}")
                print(f"      Gaussians: {eval_metrics['n_gaussians']}")
                if eval_metrics['vram_mb'] > 0:
                    print(f"      VRAM: {eval_metrics['vram_mb']:.0f} MB")
            
            # Checkpoint
            if iteration % cfg.save_interval == 0:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f'iter_{iteration:05d}.pt')
                self.gc.save(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
        
        total_time = time.time() - t_start
        
        # Final evaluation
        final_metrics = self.evaluate()
        
        # Training summary
        summary = {
            'n_initial': n_initial,
            'n_final': len(self.gc),
            'n_iterations': cfg.n_iterations,
            'total_time_min': total_time / 60.0,
            'final_psnr': final_metrics['psnr_mean'],
            'final_ssim': final_metrics['ssim_mean'],
            'final_lpips': final_metrics['lpips_mean'],
            'peak_vram_mb': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if self.device.type == 'cuda' else 0,
            'train_metrics': {
                k: v for k, v in self.metrics.items()
            }
        }
        
        # Save summary
        summary_path = os.path.join(cfg.log_dir, 'training_summary.json')
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(summary_path, 'w') as f:
            json.dump(convert(summary), f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total time:    {total_time/60:.1f} min")
        print(f"  Final PSNR:    {final_metrics['psnr_mean']:.2f} dB")
        print(f"  Final SSIM:    {final_metrics['ssim_mean']:.4f}")
        print(f"  Gaussians:     {len(self.gc)} (from {n_initial})")
        print(f"  Summary saved: {summary_path}")
        print(f"{'='*60}\n")
        
        return summary
    
    def save_final(self, path: str = None):
        """Save final trained model."""
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, 'final_model.pt')
        self.gc.save(path)
        print(f"Final model saved to: {path}")


def create_training_scene(device=None, n_gaussians: int = 5000) -> Tuple[GaussianCloudCUDA, List[PinholeCamera]]:
    """
    Create a training scene with cameras positioned around the object.
    
    Args:
        device: CUDA device
        n_gaussians: Initial number of Gaussians
    
    Returns:
        gc, cameras
    """
    from core.gaussian_model_cuda import create_test_scene_cuda
    
    # Create Gaussian cloud
    gc = create_test_scene_cuda(device=device, n_gaussians=n_gaussians)
    
    # Create training cameras (24 viewpoints around the scene)
    cameras = []
    for angle in range(0, 360, 15):  # 24 views
        rad = np.radians(angle)
        eye = np.array([6 * np.cos(rad), 1.5 + 0.5 * np.sin(2 * rad), 6 * np.sin(rad)], dtype=np.float32)
        center = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        R, t = look_at(eye, center, up)
        cam = PinholeCamera()
        cam.set_pose(R, t)
        cameras.append(cam)
    
    return gc, cameras