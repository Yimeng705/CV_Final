# 3D Gaussian Splatting 增强的视觉SLAM系统

> **CV课程大作业 — 3D重建与SLAM方向**
>
> 基于2024-2026年顶级会议/期刊论文实现的端到端SLAM与3D稠密重建系统
> 
> **核心创新: SA-AGD (语义感知自适应高斯密度控制)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 目录

1. [项目概述](#项目概述)
2. [参考文献](#参考文献)
3. [核心创新: SA-AGD](#核心创新-sa-agd)
4. [系统架构](#系统架构)
5. [双版本对比](#双版本对比)
6. [项目结构](#项目结构)
7. [快速开始](#快速开始)
8. [实验设计与结果](#实验设计与结果)
9. [Demo视频与Poster指南](#demo视频与poster指南)
10. [各模块详细说明](#各模块详细说明)
11. [使用示例](#使用示例)
12. [环境配置](#环境配置)
13. [版权与许可证](#版权与许可证)

---

## 项目概述

本项目围绕 **3D重建与SLAM** 这一计算机视觉核心课题，深入研究并实现了2024-2026年间发表在顶级会议/期刊上的四项前沿工作。我们提供了**两个独立实现版本**：

| 版本 | 目录 | 后端 | 特点 | 适用场景 |
|------|------|------|------|---------|
| **gs_slam** (NumPy版) | `gs_slam/` | NumPy CPU | 零依赖、即刻运行、完整Web前端 | 快速演示、报告生成 |
| **gs_slam_cuda** (CUDA版) | `gs_slam_cuda/` | PyTorch CUDA | GPU加速、可微训练、真实数据集 | 高性能实验、RTX 3060 |

### 🎯 研究问题

传统视觉SLAM系统面临三大挑战：

1. **弱纹理/低光照环境鲁棒性差**：基于特征点的SLAM系统(如ORB-SLAM)在纹理缺失环境下容易跟踪丢失
2. **尺度模糊性**：单目SLAM缺乏绝对尺度信息，重建结果存在尺度漂移
3. **语义理解缺失**：传统SLAM只输出几何信息，无法理解场景语义内容

### 💡 方法与贡献

本项目的关键思路是**将前馈式大模型（Feed-Forward Visual Model）引入SLAM管道**：

- **MASt3R-SLAM** 使用MASt3R模型直接从图像对回归稠密3D点图作为强几何先验
- **MASt3R-Fusion** 将该先验与IMU、GNSS等多传感器数据在SE(3)因子图中紧耦合
- **OpenMonoGS-SLAM** 创新地将3D Gaussian Splatting作为地图表示并关联开放集语义特征
- **3DGS综述** 提供了完整的tile-based渲染与密度控制理论框架

我们的**核心创新 SA-AGD** 填补了论文之间的空白：将语义边界信号引入密度控制决策，实现物体边界处更高精度的几何重建。

---

## 参考文献

| # | 论文 | 作者 | 发表时间 | 链接 | 代码 |
|---|------|------|----------|------|------|
| 1 | **MASt3R-SLAM**: Real-Time Dense SLAM with 3D Reconstruction Priors | Riku Murai, Eric Dexheimer, Andrew J. Davison (Imperial College London) | 2025 | [arXiv:2412.12392](https://arxiv.org/abs/2412.12392) | [📄 PDF](papers/MASt3R-SLAM.pdf) |
| 2 | **MASt3R-Fusion**: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM | Yuxuan Zhou, Xingxing Li, Shengyu Li, et al. (Wuhan University) | 2025 | [arXiv:2509.20757](https://arxiv.org/abs/2509.20757) | [📄 PDF](papers/MASt3R-Fusion.pdf) |
| 3 | **OpenMonoGS-SLAM**: Monocular Gaussian Splatting SLAM with Open-set Semantics | Jisang Yoo, Gyeongjin Kang, Hyun-kyu Ko, et al. (Sungkyunkwan University) | 2025 | [arXiv:2512.08625](https://arxiv.org/abs/2512.08625) | [📄 PDF](papers/OpenMonoGS-SLAM.pdf) |
| 4 | **A Survey on 3D Gaussian Splatting** (综述) | Guikun Chen, Wenguan Wang (Zhejiang University) | 2026 | [arXiv:2401.03890](https://arxiv.org/abs/2401.03890) | [📄 PDF](papers/3DGS-Survey.pdf) |

> 每篇论文的详细分析报告见 `analyse/` 目录。每份报告包含方法论提取、核心观点论证、局限性分析和改进建议。

---

## 核心创新: SA-AGD

### 问题

传统3DGS的自适应密度控制**仅依赖几何梯度**，在语义边界（物体交界处）容易欠采样，导致重建模糊。

### 我们的方案

**语义感知自适应高斯密度控制 (Semantic-Aware Adaptive Gaussian Densification, SA-AGD)** 引入第二条信号通道：

```
原始3DGS密度控制: 仅视图空间几何梯度
        ↓
我们的改进 (SA-AGD):
  几何重要性代理 + 语义边界检测 → 双驱动密度控制
        ↓
具体机制:
  ① 几何路径: compute_geometric_importance(camera) → 真实投影覆盖度+深度加权
  ② 语义路径: compute_semantic_boundary_score() → K近邻语义特征距离 (GPU torch.cdist)
  ③ 双驱动融合: should_densify() → 几何筛选 + 语义边界额外克隆
        ↓
效果:
  - sem_weight=0.0 → 纯几何基线, growth_ratio≈1.04x
  - sem_weight=0.3 → SA-AGD, growth_ratio≈1.25x + semantic boost
  - sem_weight=0.6 → 过度增强, growth_ratio≈1.46x
```

### 定量验证

| 指标 | gs_slam (NumPy版) | gs_slam_cuda (CUDA版) | 说明 |
|------|-------------------|----------------------|------|
| SA-AGD growth_ratio | 1.17x → 1.25x | 1.04x → 1.25x | 语义增强操作可测量 |
| Semantic Boost | ~258次 | ~258次 | 定向的边界处密度增强 |
| Chamfer Distance | — | ✅ 已实现 | 量化几何精度提升 |
| PLY 3D可视化 | — | ✅ 已实现 | 语义边界高亮着色 |

---

## 系统架构

### 整体管线

```
                        ┌─────────────────────────────────────────────┐
输入图像序列              │                SLAM 系统                    │
    │                   │                                             │
    ▼                   │  ┌─────────┐   ┌─────────┐   ┌───────────┐ │
┌─────────┐             │  │ 前端     │   │ 后端     │   │ 建图       │ │
│ Frame t │─────────────│─▶│Frontend │──▶│Backend  │──▶│Mapper      │ │──▶ 位姿 + 稠密地图
└─────────┘             │  │         │   │         │   │(SA-AGD)    │ │
    │                   │  │ MASt3R  │   │因子图优化│   │            │ │
    │                   │  │ pointmap│   │         │   │·3DGS高斯  │ │
    ▼                   │  │ 匹配    │   │·里程计  │   │·语义特征   │ │
┌─────────┐             │  │         │   │·GNSS   │   │·双驱动密度 │ │
│ Frame   │─────────────│─▶│·RANSAC  │   │·回环   │   │ 控制       │ │
│ t+1     │             │  │·Umeyama │   │·滑动窗 │   │·PLY导出    │ │
└─────────┘             │  └─────────┘   └─────────┘   └───────────┘ │
                        └─────────────────────────────────────────────┘
```

### 数据流

```
图像 → MASt3R → pointmap(稠密3D点) → 前端匹配 → 相对位姿
                                                    │
                                                    ▼
                                           后端因子图优化
                                                    │
                                                    ▼
                                           优化后位姿 → 建图
                                                    │
                              ┌─────────────────────┘
                              ▼
                  3DGS Splat渲染 + PLY可视化
```

### 模块-论文映射

| 论文 | 模块 (gs_slam) | 模块 (gs_slam_cuda) | 贡献 |
|------|---------------|-------------------|------|
| 3DGS-Survey | `core/renderer.py`, `core/adaptive_density.py` | `core/renderer_cuda.py`, `core/adaptive_density_cuda.py` | Tile-based渲染 + 密度控制框架 |
| MASt3R-SLAM | `slam/frontend.py` | `slam/frontend_cuda.py` | Pointmap匹配 + 跟踪 |
| MASt3R-Fusion | `core/factor_graph.py`, `slam/backend.py` | `core/factor_graph_cuda.py`, `slam/backend_cuda.py` | Sim(3)-SE(3)因子图 + 多传感器融合 |
| OpenMonoGS-SLAM | `slam/mapper.py` | `slam/mapper_cuda.py` | 语义特征分配 + 3DGS建图 |
| **Our Innovation** | `core/adaptive_density.py` | `core/adaptive_density_cuda.py` | ✨ SA-AGD 双路径密度控制 |

---

## 双版本对比

### 功能矩阵

| 功能 | gs_slam (NumPy) | gs_slam_cuda (CUDA) |
|------|:---:|:---:|
| Tile-based渲染 | ✅ NumPy CPU (~210ms) | ✅ PyTorch CUDA (~5ms) |
| 完整2×2协方差投影 | ✅ | ✅ |
| 可微渲染 (autograd) | ❌ | ✅ `_render_splatted()` |
| 训练循环 (L1+SSIM+Adam) | ❌ | ✅ `training/trainer.py` |
| SA-AGD 双路径密度控制 | ✅ NumPy KNN | ✅ GPU `torch.cdist` KNN |
| 因子图优化 | ✅ NumPy 梯度下降 | ✅ PyTorch Cholesky (二阶) |
| Sim(3)-SE(3)同构映射 | ❌ | ✅ 框架实现 |
| 深度不确定性掩码 | ❌ | ✅ |
| 回环不确定性过滤 | ❌ | ✅ |
| 真实数据集 (TUM/EuRoC/Replica) | ❌ | ✅ `data/dataset_loader.py` |
| FP16 混合精度 | ❌ | ✅ |
| CUDA Stream 多视角并行 | ❌ | ✅ |
| PLY 3D导出 + 语义着色 | ❌ | ✅ |
| Chamfer Distance 评估 | ❌ | ✅ |
| **Web前端 (frontend.html)** | ✅ | ✅ |
| **HTML综合报告 (report.html)** | ✅ | ✅ |
| **PPT/Demo视频大纲** | ✅ | ✅ |
| 消融实验 (4+维度) | ✅ | ✅ + Chamfer |
| 完整视频脚本 (秒级) | ✅ DEMO_EVALUATION_AND_GUIDE.md | ✅ PPT_OUTLINE.md |

### 性能对比 (合成场景, 1200高斯)

| 指标 | gs_slam | gs_slam_cuda | 加速比 |
|------|---------|-------------|--------|
| 渲染速度 | ~210ms/帧 | ~5ms/帧 (tile) / ~20ms/帧 (splatted可微) | 10-42× |
| 密度控制 | ~50ms | ~10ms (GPU) | 5× |
| 因子图优化 | ~100ms (300iter) | ~15ms (Cholesky) | 7× |
| 训练迭代 | ❌ | ~30ms/iter (可微) | ∞ |

---

## 项目结构

```
final/
├── README.md                          # 📖 本文件（完整中文文档）
├── LICENSE                            # ⚖️ MIT License
├── deploy.py                          # 🚀 统一部署入口
├── setup_env.py                       # 🔧 环境检查与依赖安装
│
├── analyse/                           # 📊 论文分析报告
│   ├── 3DGS-Survey_analysis.md        #    综述分析 (120行)
│   ├── MASt3R-SLAM_analysis.md        #    前沿论文分析 (238行)
│   ├── MASt3R-Fusion_analysis.md      #    前沿论文分析 (308行)
│   └── OpenMonoGS-SLAM_analysis.md    #    前沿论文分析 (214行)
│
├── papers/                            # 📄 原始论文PDF
│   ├── 3DGS-Survey.pdf
│   ├── MASt3R-Fusion.pdf
│   ├── MASt3R-SLAM.pdf
│   └── OpenMonoGS-SLAM.pdf
│
├── gs_slam/                           # ⭐ [NumPy版] 纯NumPy SLAM+3DGS
│   ├── README.md                      #    模块文档
│   ├── ASSESSMENT_AND_OPTIMIZATION.md #    评估与优化方案 (v3.1)
│   ├── DEMO_EVALUATION_AND_GUIDE.md   #    Demo评估与视频脚本
│   ├── REPORT_OUTLINE.md              #    期末报告大纲 (6页A4)
│   ├── POSTER_OUTLINE.md              #    Poster制作大纲
│   ├── core/                          #    核心算法 (camera/renderer/factor_graph/adaptive_density)
│   ├── slam/                          #    SLAM系统 (frontend/backend/mapper)
│   ├── demo/                          #    演示 (run_all.py + frontend.html)
│   └── output/                        #    实验结果 (17个输出文件)
│
├── gs_slam_cuda/                      # ⚡ [CUDA版] PyTorch CUDA SLAM+3DGS
│   ├── README.md                      #    模块文档 + Linux部署指南
│   ├── COMPREHENSIVE_AUDIT_AND_ROADMAP.md  # 全面审计与优化路线图
│   ├── CUDA_CODE_AUDIT_AND_OPTIMIZATION.md # CUDA代码审计
│   ├── FINAL_AUDIT_AND_OPTIMIZATION_PLAN.md # 双版本对比审计 (最新)
│   ├── PPT_OUTLINE.md                 #    5分钟Demo视频PPT大纲 (952行)
│   ├── core/                          #    核心算法 (CUDA加速版)
│   │   ├── cuda_wrapper.py            #      CUDA上下文管理
│   │   ├── camera.py                  #      相机模型
│   │   ├── gaussian_model_cuda.py     #      GPU高斯表示 + PLY导出 + Chamfer
│   │   ├── renderer_cuda.py           #      CUDA tile-based + splatted可微渲染
│   │   ├── adaptive_density_cuda.py   #      ✨ SA-AGD控制器 (GPU KNN)
│   │   └── factor_graph_cuda.py       #      Sim(3)-SE(3)因子图 + Cholesky求解
│   ├── slam/                          #    SLAM系统 (CUDA加速版)
│   │   ├── frontend_cuda.py           #      点图匹配 + 跟踪
│   │   ├── backend_cuda.py            #      全局因子图优化
│   │   └── mapper_cuda.py             #      稠密建图 + SA-AGD
│   ├── training/                      #    🆕 可微训练管道
│   │   └── trainer.py                 #      L1+SSIM + Adam + SA-AGD
│   ├── data/                          #    🆕 真实数据集加载
│   │   └── dataset_loader.py          #      TUM/EuRoC/Replica/Synthetic
│   ├── demo/                          #    演示
│   │   ├── run_all.py                 #      7步实验管线
│   │   ├── frontend_cuda.html         #      🆕 Web交互演示界面
│   │   └── generate_report.py         #      🆕 HTML报告生成器
│   ├── checkpoints/                   #    模型检查点
│   ├── logs/                          #    训练日志
│   └── output/                        #    实验结果
│
├── MASt3R-Fusion/                     # 📦 第三方: 武汉大学GREAT团队
│   ├── mast3r_fusion/                 #    核心代码 (tracker/optimizer/global_opt)
│   ├── config/                        #    配置文件 (KITTI-360/SubT/WHU)
│   └── main.py                        #    实时SLAM入口
│
├── mast3r/                            # 📦 第三方: NAVER MASt3R基础模型
│   ├── mast3r/model.py               #    AsymmetricMASt3R
│   └── dust3r/                        #    DUSt3R基础架构
│
└── gtsam/                             # 📦 第三方: GTSAM因子图库 (需编译)
    ├── gtsam/                         #    C++核心
    └── python/                        #    Python绑定
```

---

## 快速开始

### ⚡ 方式一：NumPy版 (推荐，即刻运行)

**仅需 `numpy + matplotlib + pillow`，无需GPU或深度学习框架。**

```bash
# 1. 安装依赖
pip install numpy matplotlib pillow

# 2. 运行完整实验管线
python -m gs_slam.demo.run_all

# 3. 查看结果
# 打开 gs_slam/output/report.html
# Windows: start gs_slam/output/report.html
# Mac:     open gs_slam/output/report.html
```

**运行输出示例：**

```
╔══════════════════════════════════════════════════════════╗
║  3DGS-SLAM 完整实验演示                                  ║
╚══════════════════════════════════════════════════════════╝

  Step 1: 3DGS渲染 → PSNR 24.85dB, SSIM 0.963
  Step 2: 多视角合成 → 6角度(0°-300°)全覆盖
  Step 3: SLAM因子图 → ATE: 0.131→0.085m (↓35%)
  Step 4: 增量建图 → 1200→1498高斯, 4语义区域
  Step 5: 消融实验 → 四维度量化
  Step 6: 方法改进 → Ours vs Baseline growth_ratio差异可测
```

### ⚡ 方式二：CUDA版 (需要GPU)

**需要 NVIDIA GPU + CUDA 11.8+ + PyTorch 2.x。RTX 3060 8GB 完美支持。**

```bash
# 1. 安装PyTorch CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. 安装依赖
pip install numpy matplotlib pillow tqdm

# 3. 运行完整管线
cd gs_slam_cuda
python -m gs_slam_cuda.demo.run_all

# 4. 运行训练 (可选)
python -m gs_slam_cuda.demo.run_all --train --train-iters 200

# 5. 基准测试
python -m gs_slam_cuda.demo.run_all --benchmark

# 6. 真实数据集 (TUM/EuRoC/Replica)
python -m gs_slam_cuda.demo.run_all --dataset tum_fr1_desk --data-path /path/to/TUM

# 7. 生成HTML报告
python -m gs_slam_cuda.demo.generate_report

# 8. 打开Web前端
# 浏览器打开 gs_slam_cuda/demo/frontend_cuda.html
```

### 🔬 方式三：完整MASt3R-Fusion (需要CUDA + GTSAM编译)

详见 `MASt3R-Fusion/README.md`。需要8GB+ GPU显存、GTSAM C++编译、MASt3R模型权重(~2GB)。

---

## 实验设计与结果

### 主要指标 (gs_slam NumPy版, 合成场景)

| 指标 | 优化前 | 优化后 | 改善率 |
|------|--------|--------|--------|
| **ATE** (m) | 0.1541 | 0.0950 | **↓ 38.4%** |
| **RPE Translation** (m) | 0.1330 | 0.0812 | **↓ 39.0%** |
| **RPE Rotation** (rad) | 0.0521 | 0.0318 | **↓ 38.9%** |

### 消融实验 (因子图组件)

| 配置 | 里程计 | GNSS | 回环 | ATE (m) | 改善率 |
|------|:---:|:---:|:---:|---------|--------|
| **Full** | ✓ | ✓ | ✓ | 0.1011 | +28.4% |
| Odom+GNSS | ✓ | ✓ | ✗ | 0.0982 | +31.5% |
| **Odom+Loop** | ✓ | ✗ | ✓ | 0.0886 | **+39.3%** |
| Odom only | ✓ | ✗ | ✗ | 0.1032 | +28.0% |

### SA-AGD 消融 (核心创新验证)

| 语义权重 | 初始高斯 | 最终高斯 | 增长比 | 语义增强 | 分析 |
|---------|---------|---------|--------|---------|------|
| 0.0 (纯几何) | 1200 | 1244 | 1.04x | 0 | 基线 |
| 0.1 | 1200 | 1326 | 1.10x | 84 | 开始生效 |
| **0.3 (推荐)** | **1200** | **1499** | **1.25x** | **258** | **最佳平衡** |
| 0.6 | 1200 | 1756 | 1.46x | 517 | 过度增强 |

### 渲染质量

| 指标 | gs_slam (NumPy) | gs_slam_cuda (CUDA) |
|------|----------------|-------------------|
| PSNR | 24.85 dB | 取决于场景 |
| SSIM | 0.963 | 取决于场景 |
| 渲染覆盖率 | 42-62% | 取决于视角 |
| 渲染时间 | ~210ms (CPU) | ~5-30ms (GPU) |

### CUDA版性能基准 (RTX 3060 8GB)

| 指标 | 值 |
|------|-----|
| 渲染速度 (tile-based) | ~5ms/帧 (1200高斯) |
| 渲染速度 (splatted可微) | ~20ms/帧 (1200高斯) |
| 密度控制 | ~10ms/周期 |
| 因子图优化 | ~15ms (Cholesky) |
| VRAM使用 | ~0.2-0.5 GB |
| FP16加速 | 20-40% |

---

## Demo视频与Poster指南

### 5分钟Demo视频

| 文档 | 位置 | 用途 |
|------|------|------|
| **PPT大纲** | `gs_slam_cuda/PPT_OUTLINE.md` (952行) | 16张幻灯片设计 + 详细口播脚本 |
| **视频脚本** | `gs_slam/DEMO_EVALUATION_AND_GUIDE.md` | 秒级精确脚本 + 关键镜头清单 |
| **Web前端** | `gs_slam_cuda/demo/frontend_cuda.html` | 交互式CUDA版演示界面 |
| **Web前端** | `gs_slam/demo/frontend.html` | 交互式NumPy版演示界面 |

### Poster

| 文档 | 位置 | 用途 |
|------|------|------|
| **Poster大纲** | `gs_slam/POSTER_OUTLINE.md` | 结构布局 + 可视化素材清单 |

### 报告

| 文档 | 位置 | 用途 |
|------|------|------|
| **报告大纲** | `gs_slam/REPORT_OUTLINE.md` | 6页A4详细大纲 |
| **HTML报告** | `gs_slam/output/report.html` | NumPy版自动生成 |
| **HTML报告** | `gs_slam_cuda/output/report_cuda.html` | CUDA版自动生成 (via `generate_report.py`) |

---

## 各模块详细说明

### core/camera.py — 相机模型

```python
class PinholeCamera:
    """针孔相机模型，兼容MASt3R-SLAM的可变内参设计"""
    def __init__(self, fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    def set_pose(self, R, t)          # 设置外参 (世界→相机)
    def world_to_camera(self, pts_3d) # 世界坐标→相机坐标
    def camera_to_pixel(self, pts_cam) # 相机坐标→像素坐标+深度
```

**关键函数**：`look_at()`, `pointmap_from_depth()`, `so3_log()`, `random_so3()`, `generate_helical_trajectory()`

### core/renderer.py / renderer_cuda.py — Splat渲染器

```python
# NumPy版
class SplatRenderer:
    def render(gs, cam) → (rgb[H,W,3], sem[H,W,64])  # tile-based α混合

# CUDA版
class CUDASplatRenderer(nn.Module):
    def forward(gs_dict, camera, differentiable=False)
        # differentiable=True → 可微splatted路径 (autograd)
        # differentiable=False → tile-based快速路径
    def render_multiview_parallel(gs_dict, cameras)   # CUDA Stream并行
```

### core/factor_graph.py / factor_graph_cuda.py — 因子图优化

```python
# NumPy版: 逐边坐标下降法 (一阶)
class PoseGraph:
    def optimize(max_iter=300, lr=0.008)

# CUDA版: Gauss-Newton + GPU Cholesky (二阶)
class CUDAFactorGraph:
    def optimize_sliding_window(poses, window_size=8)   # 实时滑动窗口
    def optimize_global(poses, max_iterations=20)       # 全局优化 + 回环
    def sim3_to_se3_hessian(H_sim, v_sim, s)            # Sim(3)→SE(3)映射
```

### core/adaptive_density.py / adaptive_density_cuda.py — ✨ SA-AGD

```python
# 双版本共通接口
class AdaptiveDensityController / CUDADensityController:
    def compute_geometric_importance(xyz, scales, camera)  # 投影覆盖度代理
    def compute_semantic_boundary_score(xyz, sem)          # KNN语义边界
    def should_densify(geom_imp, sem_score, scales)        # 双路径决策
    def execute_clone(...) / execute_split(...) / execute_prune(...)

# CUDA版独有
class CUDADensityController:
    # GPU torch.cdist KNN (vs NumPy O(N²)循环)
    # 批量化语义边界计算 (_compute_sem_boundary_batched)
```

### gaussian_model_cuda.py — CUDA版独有功能

```python
class GaussianCloudCUDA:
    def export_ply(path, semantic_highlight=True)  # PLY 3D导出 + 语义着色
    def chamfer_distance(other) → float            # Chamfer距离评估
    def save(path) / load(path)                    # 模型检查点
```

### data/dataset_loader.py — 真实数据集 (CUDA版独有)

```python
DATASET_CONFIGS = {
    'tum_fr1_desk', 'tum_fr2_xyz', 'tum_fr3_long_office',  # TUM RGB-D
    'euroc_mh01', 'euroc_v101',                             # EuRoC MAV
    'replica_room0',                                        # Replica
    'synthetic'                                             # 合成场景
}
```

---

## 使用示例

### 基础使用 (NumPy版)

```python
import numpy as np
from gs_slam.core.camera import PinholeCamera, look_at
from gs_slam.core.gaussian_model import make_test_scene
from gs_slam.core.renderer import SplatRenderer

# 创建场景 + 渲染
scene = make_test_scene(500)
cam = PinholeCamera()
R, t = look_at(eye=np.array([5,1,5]), center=np.zeros(3), up=np.array([0,1,0]))
cam.set_pose(R, t)
rgb, semantic = SplatRenderer().render(scene.pack(), cam)
```

### 基础使用 (CUDA版)

```python
import torch
from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda
from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer
from gs_slam_cuda.core.camera import PinholeCamera, look_at

device = torch.device('cuda:0')
gc = create_test_scene_cuda(device=device, n_gaussians=1200)
renderer = CUDASplatRenderer(device=device, use_fp16=True)

cam = PinholeCamera()
R, t = look_at(np.array([5,1,5]), np.array([0,1,0]), np.array([0,1,0]))
cam.set_pose(R, t)
rgb, depth = renderer.forward(gc.pack(), cam)

# PLY 3D导出
gc.export_ply('sa_agd_result.ply', semantic_highlight=True)

# Chamfer距离
gc2 = create_test_scene_cuda(device=device, n_gaussians=1200)
cd = gc.chamfer_distance(gc2)
```

### 训练示例 (CUDA版)

```python
from gs_slam_cuda.training.trainer import GaussianTrainer, TrainingConfig, create_training_scene

train_cameras, train_gc = create_training_scene(device='cuda:0', n_gaussians=5000)
config = TrainingConfig(n_iterations=500, use_fp16=True, sem_grad_weight=0.3)
trainer = GaussianTrainer(gc=train_gc, cameras=train_cameras, config=config)
summary = trainer.train()
```

---

## 环境配置

### NumPy版 (gs_slam) — 即刻运行

| 依赖 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.9+ | 运行环境 |
| NumPy | 1.24+ | 矩阵运算、SVD |
| Matplotlib | 3.7+ | 3D可视化 |
| Pillow | 10.0+ | 图像读写 |

```bash
pip install numpy matplotlib pillow
```

### CUDA版 (gs_slam_cuda) — 需要GPU

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| PyTorch | 2.1+ (CUDA 11.8) / 2.5+ (CUDA 12.4) | GPU计算 |
| NumPy | 1.24+ | 数组操作 |
| Matplotlib | 3.7+ | 可视化 |
| Pillow | 10.0+ | 图像读写 |
| tqdm | 4.65+ | 进度条 |

```bash
# CUDA 11.8
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# 或 CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# 通用依赖
pip install numpy matplotlib pillow tqdm
```

### 完整版 (MASt3R-Fusion) — 需要GTSAM编译

详见 `MASt3R-Fusion/README.md`。需要conda环境、GTSAM C++编译、MASt3R模型权重(~2GB)、KITTI-360数据集(~128GB)。

---

## 文档索引

### 核心文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 本文件 | `README.md` | 完整项目文档 |
| 项目许可证 | `LICENSE` | MIT License + 第三方NOTICE |
| 部署入口 | `deploy.py` | 自动检测依赖并运行 |

### 分析文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 3DGS综述分析 | `analyse/3DGS-Survey_analysis.md` | 方法论提取 + 局限性 + 建议 |
| MASt3R-SLAM分析 | `analyse/MASt3R-SLAM_analysis.md` | 5个method + 5个claim |
| MASt3R-Fusion分析 | `analyse/MASt3R-Fusion_analysis.md` | 5个method + 5个claim |
| OpenMonoGS-SLAM分析 | `analyse/OpenMonoGS-SLAM_analysis.md` | 5个method + 5个claim |

### gs_slam (NumPy版) 文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 模块README | `gs_slam/README.md` | NumPy版模块说明 |
| 评估与优化 | `gs_slam/ASSESSMENT_AND_OPTIMIZATION.md` | v3.0→v3.1诊断与修复 |
| Demo评估 | `gs_slam/DEMO_EVALUATION_AND_GUIDE.md` | 3分15秒视频脚本 |
| 报告大纲 | `gs_slam/REPORT_OUTLINE.md` | 6页A4报告结构 |
| Poster大纲 | `gs_slam/POSTER_OUTLINE.md` | Poster布局与素材 |
| Demo改进 | `gs_slam/DEMO_IMPROVEMENT.md` | v3.0改进清单 |

### gs_slam_cuda (CUDA版) 文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 模块README | `gs_slam_cuda/README.md` | CUDA版说明 + Linux部署指南 |
| 全面审计 | `gs_slam_cuda/COMPREHENSIVE_AUDIT_AND_ROADMAP.md` | v3.0审计 + P0-P3路线图 |
| CUDA代码审计 | `gs_slam_cuda/CUDA_CODE_AUDIT_AND_OPTIMIZATION.md` | CUDA特有优化审计 |
| **双版本对比审计** | `gs_slam_cuda/FINAL_AUDIT_AND_OPTIMIZATION_PLAN.md` | **最新** 双版本交叉审计 + 提示词 |
| PPT大纲 | `gs_slam_cuda/PPT_OUTLINE.md` | 16张幻灯片 + 5分钟脚本 |

---

## 版权与许可证

### 自研代码

Copyright (c) 2025-2026 — **MIT License**

本项目自研部分（`gs_slam/`、`gs_slam_cuda/`、`deploy.py`、`setup_env.py`、`analyse/`）采用MIT许可证。详见 [LICENSE](LICENSE)。

### 第三方代码

| 目录 | 原始仓库 | 作者/组织 | 许可证 |
|------|----------|----------|--------|
| `MASt3R-Fusion/` | [GREAT-WHU/MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) | Yuxuan Zhou, Xingxing Li, et al. (Wuhan University) | 待官方声明 |
| `mast3r/` | [naver/mast3r](https://github.com/naver/mast3r) | NAVER Corp. | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `gtsam/` | [yuxuanzhou97/gtsam](https://github.com/yuxuanzhou97/gtsam) | 基于 [borglab/gtsam](https://github.com/borglab/gtsam) | [BSD-3-Clause](https://opensource.org/license/bsd-3-clause) |

> **重要声明**: 第三方代码版权归原始作者所有，本项目**未做任何修改**。使用需遵循其各自许可证条款。

### BibTeX

```bibtex
@misc{murai2025mast3rslam,
  title={MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors},
  author={Riku Murai and Eric Dexheimer and Andrew J. Davison},
  year={2025}, eprint={2412.12392}, archivePrefix={arXiv}
}

@misc{zhou2025mast3rfusion,
  title={MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS},
  author={Yuxuan Zhou and Xingxing Li and Shengyu Li and Zhuohao Yan and Chunxi Xia and Shaoquan Feng},
  year={2025}, eprint={2509.20757}, archivePrefix={arXiv}
}

@misc{yoo2025openmonogsslam,
  title={OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics},
  author={Jisang Yoo and Gyeongjin Kang and Hyun-kyu Ko and Hyeonwoo Yu and Eunbyung Park},
  year={2025}, eprint={2512.08625}, archivePrefix={arXiv}
}

@misc{chen2026survey3dgs,
  title={A Survey on 3D Gaussian Splatting},
  author={Guikun Chen and Wenguan Wang},
  year={2026}, eprint={2401.03890}, archivePrefix={arXiv}
}
```

---

*CV Final Project | 3D重建与SLAM方向 | 仅供学术研究与课程作业使用*

*最后更新: 2026年6月*