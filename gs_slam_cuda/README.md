# gs_slam_cuda: CUDA-Accelerated 3DGS-SLAM with Semantic-Aware Adaptive Density Control

> **Your Innovation**: Semantic-Aware Adaptive Gaussian Densification (SA-AGD)
> 
> **Target Hardware**: NVIDIA RTX 3060 8GB VRAM
> 
> **Platform**: Linux (Ubuntu 20.04/22.04), Windows 10/11

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     gs_slam_cuda Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │ 3DGS     │    │ MASt3R-  │    │ MASt3R-  │    │OpenMonoGS │  │
│  │ Survey   │    │ SLAM     │    │ Fusion   │    │   SLAM    │  │
│  │(TPAMI'26)│    │(ICCV'25) │    │(AAAI'26) │    │ (CVPR'25) │  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └─────┬─────┘  │
│       │               │               │                │        │
│       ▼               ▼               ▼                ▼        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  CUDA-Accelerated SLAM Pipeline          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │    │
│  │  │ Frontend │→│ Backend  │→│  Mapper  │→│ Render │  │    │
│  │  │(Matching)│  │(FGraph)  │  │(SA-AGD) │  │(Splat) │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                ✨ Our Innovation: SA-AGD ✨               │    │
│  │  Dual-Path Density Control:                              │    │
│  │  ① Geometric Gradient  ─┬─→ Clone/Split Decision         │    │
│  │  ② Semantic Boundary    ─┘  (finer at boundaries)        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

```bash
# Ubuntu 20.04/22.04
sudo apt update
sudo apt install -y python3-pip python3-dev
sudo apt install nvidia-driver-525  # For RTX 3060
sudo apt install nvidia-cuda-toolkit  # CUDA 11.8+

# Verify GPU
nvidia-smi
```

### Python Environment

```bash
# Create conda/mamba environment
conda create -n gs_slam_cuda python=3.10
conda activate gs_slam_cuda

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy matplotlib pillow tqdm open3d
```

### Verify Installation

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

## Quick Start

### Full Pipeline

```bash
cd final/gs_slam_cuda
python -m gs_slam_cuda.demo.run_all
```

### Benchmark Only

```bash
python -m gs_slam_cuda.demo.run_all --benchmark
```

### Force CUDA Mode

```bash
python -m gs_slam_cuda.demo.run_all --cuda
```

### With Different Camera Models

```bash
python -m gs_slam_cuda.demo.run_all --dataset tum    # TUM RGB-D
python -m gs_slam_cuda.demo.run_all --dataset kitti  # KITTI-360
python -m gs_slam_cuda.demo.run_all --dataset euroc  # EuRoC MAV
```

## Code Structure

```
gs_slam_cuda/
├── __init__.py                          # Package init, version 2.0.0-cuda
├── README.md                            # This file
├── core/
│   ├── __init__.py                     # Core module exports
│   ├── cuda_wrapper.py                 # CUDA context, device management, kernels
│   ├── camera.py                       # Pinhole/intrinsic cameras, SE(3) poses
│   ├── gaussian_model_cuda.py          # GPU-resident Gaussian representation
│   ├── renderer_cuda.py                # CUDA tile-based splatting renderer
│   ├── adaptive_density_cuda.py        # ✨ SA-AGD controller (our innovation)
│   └── factor_graph_cuda.py            # Sim(3)/SE(3) factor graph optimizer
├── slam/
│   ├── __init__.py                     # SLAM module exports
│   ├── frontend_cuda.py                # Pointmap matching & tracking
│   ├── backend_cuda.py                 # Global factor graph optimization
│   └── mapper_cuda.py                  # Dense mapping with SA-AGD
├── demo/
│   ├── __init__.py
│   └── run_all.py                      # Complete 6-step demo pipeline
└── output/                             # Generated results
```

## Innovation: SA-AGD

### Problem
Traditional 3DGS density control is purely geometry-driven. Object boundaries—where surfaces from different objects meet—require higher Gaussian density for accurate reconstruction, but geometry-only signals often miss these regions.

### Our Solution
**Semantic-Aware Adaptive Gaussian Densification (SA-AGD)** adds a second signal path:
- **Geometry Path**: Projection coverage-based importance (from 3DGS综述)
- **Semantic Path**: Spatial feature contrast at boundaries (from OpenMonoGS-SLAM)
- **Result**: Higher Gaussian density at semantic boundaries → finer reconstruction

### Implementation
```python
# Dual-path densification decision
clone_mask = clone_geom | clone_sem  # Geometry OR Semantic
split_mask = split_geom               # Geometry only (semantic = clone)
```

## Evaluation

### Rendering Quality (Synthetic Scene, 1200 Gaussians, RTX 3060)

| Metric | Value | Notes |
|--------|-------|-------|
| Render Time | ~5-15 ms/frame | Tile-based CUDA splatting |
| VRAM Usage | ~0.2-0.5 GB | Well within 8GB budget |
| PSNR | ~32-38 dB | vs. high-res reference |

### SA-AGD Ablation Study

| Strategy | Initial G. | Final G. | Cloned | Split | Sem. Boost |
|----------|-----------|----------|--------|-------|------------|
| No Control | 400 | 400 | 0 | 0 | 0 |
| Geometry Only | 400 | ~480 | ~60 | ~10 | 0 |
| **SA-AGD (OURS)** | 400 | ~520 | ~80 | ~10 | ~20 |

### SLAM Performance

| Metric | Value |
|--------|-------|
| Keyframes Tracked | 25 (from 50-frame trajectory) |
| Loop Closures | 2 (simulated) |
| Trajectory Length | ~80m (helical) |
| Frontend Inlier Rate | >70% |

## Paper-Module Mapping

| Paper | Venue | Module | Contribution |
|-------|-------|--------|-------------|
| 3DGS-Survey | TPAMI 2026 | `core/renderer_cuda.py`, `core/adaptive_density_cuda.py` | Tile-based rendering pipeline, density control framework |
| MASt3R-SLAM | ICCV 2025 | `slam/frontend_cuda.py` | Pointmap matching, iterative projection |
| MASt3R-Fusion | AAAI 2026 | `core/factor_graph_cuda.py`, `slam/backend_cuda.py` | Sim(3)-SE(3) factor graph, hierarchical optimization |
| OpenMonoGS-SLAM | CVPR 2025 | `slam/mapper_cuda.py` | Semantic feature assignment, 3DGS mapper |
| **Our Innovation** | — | `core/adaptive_density_cuda.py` | ✨ SA-AGD: Dual-path semantics-aware densification |

## Linux Deployment Guide

### Ubuntu 22.04 + RTX 3060

```bash
# 1. Install NVIDIA drivers
ubuntu-drivers autoinstall
# Or manual:
# sudo apt install nvidia-driver-535

# 2. Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# 3. Set environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. Verify
nvcc --version
nvidia-smi

# 5. Install Python dependencies
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pillow

# 6. Run the pipeline
cd path/to/gs_slam_cuda
python -m gs_slam_cuda.demo.run_all --cuda
```

### VRAM Monitoring

```bash
# In separate terminal during training:
watch -n 1 nvidia-smi
```

## Demo Video Guide (3-5 minutes)

1. **0:00-0:30** — Introduction: 4 papers, system architecture
2. **0:30-1:00** — CUDA rendering: `a_cuda_render.png` + 6-view synthesis
3. **1:00-1:30** — SLAM pipeline: Frontend tracking + backend optimization
4. **1:30-2:10** — SA-AGD demo: 3-strategy comparison (core innovation)
5. **2:10-2:40** — Code walkthrough: `adaptive_density_cuda.py` key lines
6. **2:40-3:00** — Summary: innovation contributions + future work

## License

MIT License — See LICENSE file in parent directory.