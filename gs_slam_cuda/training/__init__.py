"""
Training Pipeline for gs_slam_cuda
====================================
GPU-accelerated training loop for Gaussian parameters.
Optimized for RTX 3060 8GB on Linux.
"""

from .trainer import GaussianTrainer, TrainingConfig