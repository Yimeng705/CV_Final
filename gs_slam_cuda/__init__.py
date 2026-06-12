"""
gs_slam_cuda: CUDA-Accelerated 3D Gaussian Splatting SLAM
=========================================================

An innovative SLAM architecture combining:
- 3D Gaussian Splatting (3DGS) for explicit scene representation
- MASt3R-like feed-forward pointmap regression for dense matching
- Semantic-aware adaptive density control (our innovation)
- Hybrid factor graph optimization with IMU/GNSS fusion
- Full CUDA acceleration for real-time performance on RTX 3060 8GB

Based on four papers:
[1] 3DGS-Survey (Chen & Wang, 2026, TPAMI) - Systematic 3DGS review
[2] MASt3R-SLAM (Murai et al., 2025, ICCV) - Real-time dense SLAM
[3] MASt3R-Fusion (Zhou et al., 2025, AAAI 2026) - Multi-sensor fusion
[4] OpenMonoGS-SLAM (Yoo et al., 2025, CVPR) - Open-set semantic SLAM

Our Innovation: Semantic-Aware Adaptive Density Control
- Fuses semantic boundary detection with geometric gradient signals
- Dual-path densification: geometry-driven + semantics-driven
- Achieves finer reconstruction at object boundaries

Usage:
  python -m gs_slam_cuda.demo.run_all --dataset kitti360 --cuda
  python -m gs_slam_cuda.demo.evaluate --checkpoint output/model.pth
"""

__version__ = "2.0.0-cuda"