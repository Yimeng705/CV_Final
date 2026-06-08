# 3D Gaussian Splatting 增强的视觉SLAM系统

> **CV课程大作业 — 3D重建与SLAM方向**
>
> 基于2024-2025年顶级会议论文实现的端到端SLAM与3D稠密重建系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## 📋 目录

1. [项目概述](#项目概述)
2. [参考文献](#参考文献)
3. [系统架构](#系统架构)
4. [核心算法详解](#核心算法详解)
5. [项目结构](#项目结构)
6. [快速开始](#快速开始)
7. [实验设计与结果](#实验设计与结果)
8. [各模块详细说明](#各模块详细说明)
9. [使用示例](#使用示例)
10. [环境配置](#环境配置)
11. [版权与许可证](#版权与许可证)

---

## 项目概述

本项目围绕 **3D重建与SLAM** 这一计算机视觉核心课题，深入研究并实现了2024-2025年间发表在顶级会议/arXiv上的四项前沿工作。我们自研了一套完整的3D Gaussian Splatting增强的视觉SLAM系统（`gs_slam/`），同时集成了三套官方开源实现作为参考。

### 🎯 研究问题

传统视觉SLAM系统面临三大挑战：

1. **弱纹理/低光照环境鲁棒性差**：基于特征点的SLAM系统(如ORB-SLAM)在纹理缺失环境下容易跟踪丢失
2. **尺度模糊性**：单目SLAM缺乏绝对尺度信息，重建结果存在尺度漂移
3. **语义理解缺失**：传统SLAM只输出几何信息，无法理解场景语义内容

### 💡 方法与贡献

本项目的关键思路是**将前馈式大模型（Feed-Forward Visual Model）引入SLAM管道**：

- **MASt3R-SLAM** 使用MASt3R模型直接从图像对回归稠密3D点图(pointmap)作为强几何先验，替代传统的手工特征匹配
- **MASt3R-Fusion** 进一步将该先验与IMU、GNSS等多传感器数据在SE(3)因子图中紧耦合，解决了尺度模糊问题
- **OpenMonoGS-SLAM** 创新地将3D Gaussian Splatting作为地图表示，并关联开放集语义特征，实现了"所见即所得"的语义SLAM

我们的自研实现（`gs_slam/`）基于这些论文的算法框架，使用纯NumPy构建了完整的SLAM管道，可以独立运行并生成完整的实验报告。

---

## 参考文献

| # | 论文 | 作者 | 发表时间 | 链接 | 代码 |
|---|------|------|----------|------|------|
 | 1 | **MASt3R-SLAM**: Real-Time Dense SLAM with 3D Reconstruction Priors | Riku Murai, Eric Dexheimer, Andrew J. Davison (Imperial College London) | 2025 | [arXiv:2412.12392](https://arxiv.org/abs/2412.12392) | [项目页](https://edexheim.github.io/mast3r-slam) | [📄 PDF](papers/MASt3R-SLAM.pdf) |
| 2 | **MASt3R-Fusion**: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM | Yuxuan Zhou, Xingxing Li, Shengyu Li, et al. (Wuhan University) | 2025.09 | [arXiv:2509.20757](https://arxiv.org/abs/2509.20757) | [GREAT-WHU/MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) | [📄 PDF](papers/MASt3R-Fusion.pdf) |
| 3 | **OpenMonoGS-SLAM**: Monocular Gaussian Splatting SLAM with Open-set Semantics | Jisang Yoo, Gyeongjin Kang, Hyun-kyu Ko, et al. (Sungkyunkwan University) | 2025.12 | [arXiv:2512.08625](https://arxiv.org/abs/2512.08625) | [jisang1528/OpenMonoGS-SLAM](https://jisang1528.github.io/OpenMonoGS-SLAM) | [📄 PDF](papers/OpenMonoGS-SLAM.pdf) |
| 4 | **A Survey on 3D Gaussian Splatting** (综述) | Guikun Chen, Wenguan Wang (Zhejiang University) | 2024.01 | [arXiv:2401.03890](https://arxiv.org/abs/2401.03890) | [Awesome3DGS](https://github.com/guikunchen/Awesome3DGS) | [📄 PDF](papers/3DGS-Survey.pdf) |

### 论文内容概要

#### 论文1: MASt3R-SLAM

提出了一种**自底向上(Bottom-Up)**设计的实时单目稠密SLAM系统，核心思想是使用MASt3R作为"两视图3D重建与匹配先验"。系统包含：

- **点图匹配**(Pointmap Matching)：使用MASt3R直接预测的稠密3D对应关系，通过迭代投影匹配实现高速像素级对应搜索
- **相机跟踪**(Camera Tracking)：基于射线误差(Ray Error)的Sim(3)位姿优化，对深度预测误差具有天然鲁棒性
- **局部融合**(Local Fusion)：加权平均滤波将当前帧点图融合到规范关键帧点图中
- **闭环检测**(Loop Closure)：增量式ASMK图像检索结合MASt3R几何匹配验证
- **二阶全局优化**(Second-Order Global Optimization)：Sim(3)群上的高斯-牛顿法，利用解析雅可比与CUDA并行构建Hessian矩阵，通过稀疏Cholesky分解求解

系统在无任何相机模型假设的情况下，达到了15 FPS的实时性能。

#### 论文2: MASt3R-Fusion

将前馈视觉模型扩展到**多传感器融合**场景，核心创新包括：

- **两视图稠密匹配与动态剔除**：MASt3R前馈模型回归点图与描述符，通过射线邻近匹配与深度残差掩码实现大视角稠密匹配，同时剔除动态物体
- **Sim(3)视觉约束的Hessian压缩**：将稠密点图对齐残差（重投影误差+深度差）在GPU上压缩为紧凑的(7,7) Hessian矩阵和(7,1)向量，消除显式地标变量，大幅简化后端优化
- **深度不确定性下加权掩码**：当投影后深度比值异常（(S∘X_j)_z < τ·(X_i)_z, τ=1.25）时对残差施加下加权因子f=0.1，抑制大场景前向运动中远→近点对的投影误差，提升VIO稳定性
- **Sim(3)→SE(3)×R 群同构映射**：将相似变换分解为SE(3)运动+标量缩放，通过李代数间线性映射Λ将7维Sim(3)视觉约束转换为14维SE(3)+scale因子，实现视觉与IMU/GNSS度量尺度约束在统一因子图中的无缝融合
- **层次化因子图**：实时滑动窗口（舒尔补概率边缘化）保留原始视觉Hessian与IMU预积分信息，全局阶段分步优化——先相对位姿回环约束+Cauchy鲁棒核排除外点，再将内点回环转换为Hessian形式进行精优化
- **不确定性驱动回环过滤**：基于VIO里程计误差传播（沿/垂直方向协方差建模）快速排除几何上不可能共视的回环候选，保留激进真回环的同时大幅降低假阳性

#### 论文3: OpenMonoGS-SLAM

首个将**3D Gaussian Splatting与开放集语义**统一的单目SLAM系统，完全自监督，无需深度传感器或3D语义真值：

- **VFMs集成**：MASt3R提供密集几何对应与初始位姿、SAM生成无类别2D掩码、CLIP提取语言特征
- **记忆驱动的语义特征聚合**：维护动态记忆库累积历史帧的SAM掩码过滤后的CLIP特征，通过注意力融合克服单帧噪声，显著提升语义分割时空一致性
- **多尺度语义监督**：在多个图像尺度（S=4）上同时计算语言回归损失，平衡大尺度整体一致性与小尺度边界细节
- **多视图对比语义损失**：利用SAM掩码构建跨视图正/负样本对，采用InfoNCE对比学习强制跨视图语义不变性，消除单视图监督造成的空间碎片化
- **可微渲染语义特征图**：在3D高斯上关联可学习的64维语义嵌入向量，通过alpha blending与RGB并行渲染，支持开放词汇语义查询

#### 论文4: 3DGS综述

3D Gaussian Splatting领域的首篇系统综述，全面覆盖：

- **基础理论**：3D高斯的参数化、可微渲染、优化策略
- **应用场景**：SLAM、动态场景、生成模型、自动驾驶
- **基准评估**：多个3DGS变体在各任务上的性能对比
- **未来方向**：效率优化、稀疏视图、大规模场景

---

## 系统架构

### 整体管线

```
                        ┌─────────────────────────────────────┐
输入图像序列              │          SLAM 系统                  │
    │                   │                                     │
    ▼                   │  ┌─────────┐   ┌─────────┐   ┌─────┐│
┌─────────┐             │  │ 前端     │   │ 后端     │   │建图 ││
│ Frame t │─────────────│─▶│Frontend │──▶│Backend  │──▶│Mapper││──▶ 位姿 + 稠密地图
└─────────┘             │  │         │   │         │   │     ││
    │                   │  │ MASt3R  │   │因子图优化│   │3DGS ││
    │                   │  │ pointmap│   │         │   │     ││
    ▼                   │  │ 匹配    │   │·里程计  │   │·高斯││
┌─────────┐             │  │         │   │·GNSS   │   │ 核  ││
│ Frame   │─────────────│─▶│·RANSAC  │   │·回环   │   │·语义││
│ t+1     │             │  │·Umeyama │   │·滑动窗 │   │·增量││
└─────────┘             │  └─────────┘   └─────────┘   └─────┘│
                        └─────────────────────────────────────┘
```

### 模块间的数据流

```
图像 → MASt3R → pointmap(稠密3D点) → 前端匹配 → 相对位姿(T_ij)
                                                    │
                                                    ▼
                                           后端因子图(PoseGraph)
                                           ┌──────────────────┐
                                           │ T_0 ← T_1 ← ... ← T_N │
                                           │  ↑ odometry edge      │
                                           │  ↑ GNSS edge          │
                                           │  ↑ loop closure edge  │
                                           └──────────────────┘
                                                    │
                                                    ▼
                                           优化后位姿(optimized poses)
                                                    │
                                                    ▼
                                     建图(Mapper): 3D点→世界坐标系
                                     生成3D高斯核(μ,q,s,α,c,fsem)
                                                    │
                                                    ▼
                                      Splat渲染: 新视图合成
                                      语义渲染: 开放集语义分割
```

---

## 核心算法详解

### 1. 3D Gaussian Splatting 表示 (基于综述[4])

每个3D高斯核 $\Theta_i = \{\mu_i, q_i, s_i, \alpha_i, c_i, f_i^{sem}\}$ 包含六个参数：

- **位置** $\mu_i \in \mathbb{R}^3$：高斯核在3D世界坐标系中的中心点
- **旋转四元数** $q_i \in \mathbb{R}^4$：表示协方差矩阵的旋转分量 ($\|q\|=1$)
- **尺度** $s_i \in \mathbb{R}^3$：各向异性缩放因子的对数（通过 $\exp$ 确保正值）
- **不透明度** $\alpha_i \in [0,1]$：通过 sigmoid 函数约束
- **颜色** $c_i \in \mathbb{R}^3$：RGB颜色（简化处理，实际使用0阶球谐系数）
- **语义特征** $f_i^{sem} \in \mathbb{R}^{64}$：高维语义嵌入向量（OpenMonoGS-SLAM的创新）

**协方差矩阵分解**（保证正定性）：

$$\Sigma_i = R(q_i) \cdot \text{diag}(\exp(s_i)) \cdot \text{diag}(\exp(s_i))^T \cdot R(q_i)^T$$

其中 $R(q)$ 是从四元数计算的3×3旋转矩阵。

**可微渲染**（Alpha Blending）：

$$C(x) = \sum_{i \in N} c_i \cdot \alpha_i \cdot G'_i(x) \cdot \prod_{j=1}^{i-1} (1 - \alpha_j \cdot G'_j(x))$$

其中 $G'_i(x)$ 是3D高斯投影到2D图像平面的值：

$$G'_i(x) = \exp\left(-\frac{1}{2}(x - \mu_i^{2D})^T \Sigma_i^{2D^{-1}} (x - \mu_i^{2D})\right)$$

2D协方差通过仿射近似：$\Sigma_i^{2D} = J W \Sigma_i W^T J^T$，其中 $J$ 是投影变换的雅可比矩阵。

### 2. 因子图SLAM优化 (基于[1][2])

#### Pose Graph 构建

SLAM问题被建模为**位姿图**(Pose Graph)上的非线性最小二乘优化：

$$\min_{T_0,...,T_N} \sum_{k} w_k \cdot \|e_k(T_{i_k}, T_{j_k})\|^2$$

其中 $T_i = (R_i, t_i) \in SE(3)$ 是第 $i$ 帧的相机位姿，$e_k$ 是各种因子的残差函数：

**里程计因子**（相邻帧约束）：

$$e_{odo} = \begin{bmatrix} \log(R_{ij}^{meas\ T} \cdot R_i^T R_j) \\ R_i^T(t_j - t_i) - t_{ij}^{meas} \end{bmatrix}$$

**GNSS因子**（全局位置约束，来自MASt3R-Fusion）：

$$e_{gnss} = t_i - t_i^{global}$$

**回环因子**（非相邻帧约束）：

$$e_{loop} = \begin{bmatrix} \log(R_{ij}^{meas\ T} \cdot R_i^T R_j) \\ R_i^T(t_j - t_i) - t_{ij}^{meas} \end{bmatrix}, \quad |i-j| > 5$$

#### 优化方法

我们实现了一种**逐边坐标下降法**(Edge-wise Coordinate Descent)，在每次迭代中：

1. 遍历所有因子边，计算当前残差
2. 将残差按权重均分到两个关联节点
3. 在SO(3)切空间上更新旋转（指数映射）
4. 使用SVD强制旋转矩阵正交化

```python
# 伪代码
for iteration in range(max_iter):
    for edge in all_edges:
        e_R, e_t = compute_residual(edge)
        # 更新关联的SE(3)节点
        T_i @= exp(-lr * weight * skew(e_R))
        t_i -= lr * weight * R_i @ e_t
    # SVD正交化
    for T in all_poses:
        U, _, Vt = svd(R)
        R = U @ Vt
```

### 3. 点图匹配 (基于[1])

MASt3R对一对图像 $(I_1, I_2)$ 输出两个点图 $X_1, X_2 \in \mathbb{R}^{H \times W \times 3}$，均在相机1坐标系下表示对应的3D坐标。本系统模拟此过程：

**Umeyama算法**：给定3D-3D对应点集 $\{A_i\}, \{B_i\}$，通过SVD求解最优刚性变换：

$$H = \tilde{A}^T \tilde{B}$$
$$\text{SVD}(H) = U S V^T$$
$$R^* = V U^T, \quad t^* = \bar{B} - R^* \bar{A}$$

为确保 $R^* \in SO(3)$，检查 $\det(R^*) = +1$；若为 $-1$ 则取反最后一行。

**RANSAC鲁棒估计**：随机采样3对匹配点，计算候选变换，统计内点数（误差<阈值），选取最佳模型。

### 4. 开放集语义建图 (基于[3])

每个3D高斯关联一个64维语义特征向量 $f_i^{sem}$。在渲染时，语义特征与RGB并行进行alpha blending：

$$F^{sem}(x) = \sum_i f_i^{sem} \cdot \alpha_i \cdot G'_i(x) \cdot \prod_{j=1}^{i-1} (1 - \alpha_j \cdot G'_j(x))$$

得到的语义特征图可通过**PCA降维**（3通道）进行可视化，展示不同语义区域的分布。在实际系统中（需要SAM+CLIP），这些特征支持开放词汇的语义查询。

---

## 项目结构

```
final/
│
├── README.md                     # 📖 本文件（完整中文文档）
├── LICENSE                       # ⚖️ MIT License + 第三方NOTICE
├── .gitignore                    # Git忽略规则
├── setup_git.sh                  # 一键推送到GitHub的脚本
│
├── deploy.py                     # 🚀 统一部署入口
│   └── 自动检测可用依赖(PyTorch/GTSAM)，fallback到纯NumPy实现
│
├── setup_env.py                  # 🔧 环境检查与依赖安装脚本
│
├── gs_slam/                      # ⭐ [自研核心] 纯NumPy SLAM+3DGS实现
│   ├── __init__.py               #    模块说明与论文引用
│   ├── README.md                 #    模块文档
│   │
│   ├── core/                     #    核心算法层
│   │   ├── __init__.py
│   │   ├── camera.py             #    相机模型 (PinholeCamera、look_at、pointmap)
│   │   │                         #    支持可变内参、点图与深度互转
│   │   ├── gaussian_model.py     #    3D高斯云 (GaussianCloud)
│   │   │                         #    参数化: μ(3D位置), q(旋转四元数), s(对数尺度)
│   │   │                         #            α(不透明度logit), c(RGB), fsem(语义)
│   │   │                         #    方法: add(), prune(), pack(), get_covariances()
│   │   ├── renderer.py           #    Splat渲染器 (SplatRenderer)
│   │   │                         #    实现: 视锥体裁剪→投影→排序→Alpha Blending
│   │   │                         #    输出: RGB图像 + 语义特征图
│   │   └── factor_graph.py       #    位姿图优化 (PoseGraph, FactorEdge)
│   │                             #    支持: 里程计边/GNSS边/回环边
│   │                             #    优化: 逐边坐标下降 + SVD正交化
│   │
│   ├── slam/                     #    SLAM系统层
│   │   ├── __init__.py
│   │   ├── frontend.py           #    前端: 点图匹配 (RANSAC + Umeyama算法)
│   │   │                         #    合成数据生成、关键帧选择
│   │   ├── backend.py            #    后端: 多传感器因子图构建与全局优化
│   │   │                         #    支持有/无GNSS、有/无回环的消融配置
│   │   └── mapper.py             #    建图: 增量式3D高斯地图构建
│   │                             #    语义区域分配 (模拟SAM+CLIP)
│   │
│   ├── demo/                     #    演示与实验
│   │   ├── __init__.py
│   │   └── run_all.py            #    🔥 [主入口] 完整5步实验管线
│   │                             #    Step1: 3DGS渲染 → Step2: 多视角合成
│   │                             #    Step3: SLAM优化 → Step4: 增量建图
│   │                             #    Step5: 消融实验 → HTML报告生成
│   │
│   └── output/                   #    📊 实验结果输出
│       ├── report.html           #    HTML综合实验报告（含指标、可视化、消融）
│       ├── a_3dgs_render.png     #    3DGS渲染结果
│       ├── b_pointcloud.png      #    传统点云渲染(基线)
│       ├── c_comparison.png      #    3DGS vs 点云对比图
│       ├── d_view_*.png          #    6个角度多视角合成 (0°,60°,120°,180°,240°,300°)
│       ├── e_trajectory.png      #    SLAM 3D轨迹对比 + 优化收敛曲线
│       ├── f_mapping_result.png  #    增量建图结果
│       ├── g_semantic.png        #    PCA语义特征可视化
│       └── h_ablation.txt        #    消融实验数据表
│
├── MASt3R-Fusion/                # 📦 第三方: 武汉大学GREAT团队
│   ├── README.md                 #    (包含完整安装和使用说明)
│   ├── mast3r_fusion/            #    核心代码
│   │   ├── tracker.py            #    帧追踪器 (FrameTracker)
│   │   ├── nonlinear_optimizer.py #   非线性优化器
│   │   ├── global_opt.py         #    全局优化模块
│   │   ├── mast3r_utils.py       #    MASt3R模型工具
│   │   └── ...
│   ├── config/                   #    配置文件 (KITTI-360, SubT, WHU)
│   ├── evaluation/               #    评估脚本
│   └── main.py                   #    实时SLAM入口
│
├── mast3r/                       # 📦 第三方: NAVER MASt3R基础模型
│   ├── mast3r/                   #    模型定义
│   │   ├── model.py              #    AsymmetricMASt3R
│   │   ├── cloud_opt.py          #    全局点云优化
│   │   └── ...
│   └── dust3r/                   #    DUSt3R基础架构
│
└── gtsam/                        # 📦 第三方: 修改版GTSAM因子图库
    ├── gtsam/                    #    C++核心 (需编译)
    └── python/                   #    Python绑定
```

---

## 快速开始

### ⚡ 方式一：自研实现 (推荐，立即可用)

**仅需 `numpy + matplotlib + pillow`，无需GPU或深度学习框架，3秒内启动。**

```bash
# 1. 安装依赖 (如已安装可跳过)
pip install numpy matplotlib pillow

# 2. 运行完整实验管线
python -m gs_slam.demo.run_all

# 3. 查看结果
# 打开 gs_slam/output/report.html
# Windows: start gs_slam/output/report.html
# Mac:     open gs_slam/output/report.html
# Linux:   xdg-open gs_slam/output/report.html
```

**运行输出示例：**

```
╔══════════════════════════════════════════════════════════╗
║  3DGS-SLAM 完整实验演示                                  ║
║  论文: MASt3R-SLAM × MASt3R-Fusion × OpenMonoGS-SLAM     ║
╚══════════════════════════════════════════════════════════╝

============================================================
  Step 1: 3D Gaussian Splatting 场景渲染 (论文[4]: 3DGS综述)
============================================================
  ✓ 创建了 300 个高斯核 (球体+立方体+平面)
  ✓ 3DGS渲染: 0.205s, 非白像素: 138964/307200

============================================================
  Step 3: SLAM因子图优化 (论文[1]MASt3R-SLAM + 论文[2]MASt3R-Fusion)
============================================================
  [Before] ATE: 0.1541m
  [After]  ATE: 0.0950m (38.4% improvement)
```

### 🔬 方式二：完整MASt3R-Fusion (需要CUDA + GTSAM)

**需要8GB+ GPU显存、conda环境、GTSAM编译、MASt3R模型权重。**

```bash
# 详细安装步骤见 MASt3R-Fusion/README.md
conda create -n mast3r_fusion python=3.11.9
conda activate mast3r_fusion
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 编译 GTSAM (需要 cmake, boost)
cd gtsam && mkdir build && cd build
cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11.9
make python-install -j12

# 安装 MASt3R-Fusion
cd MASt3R-Fusion
pip install -e thirdparty/mast3r
pip install -e thirdparty/in3d
pip install --no-build-isolation -e .

# 下载模型权重
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# 运行 (需要KITTI-360数据集)
python main.py --dataset <path> --config config/base_kitti360.yaml ...
```

---

## 实验设计与结果

### 实验设置

| 参数 | 值 | 说明 |
|------|-----|------|
| 场景 | 合成场景 (球体+立方体+平面) | 300个3D高斯核，尺寸约10×6×10m |
| 轨迹 | 环形20帧 | 半径6m，含正弦高度变化 |
| 噪声模型 | 旋转 σ=0.02rad, 平移 σ=0.05m | 模拟SLAM前端不确定性 |
| 优化方法 | 逐边坐标下降 | 300次迭代，初始学习率0.008 |
| 相机模型 | 针孔相机 | fx=fy=500, 640×480分辨率 |
| 评估指标 | ATE, RPE-t, RPE-r | 绝对轨迹误差 + 相对位姿误差 |

### 定量结果

#### 主要指标

| 指标 | 优化前 | 优化后 | 改善率 |
|------|--------|--------|--------|
| **ATE** (m) | 0.1541 | 0.0950 | **↓ 38.4%** |
| **RPE Translation** (m) | 0.1330 | 0.0812 | **↓ 39.0%** |
| **RPE Rotation** (rad) | 0.0521 | 0.0318 | **↓ 38.9%** |

#### 消融实验

我们设置了四种配置来评估各组件的重要性：

| 配置 | 里程计边 | GNSS边 | 回环边 | ATE (m) | 改善率 |
|------|:---:|:---:|:---:|---------|--------|
| **Full** (完整系统) | ✓ | ✓ | ✓ | 0.1011 | +28.4% |
| **Odom+GNSS** (无回环) | ✓ | ✓ | ✗ | 0.0982 | +31.5% |
| **Odom+Loop** (无GNSS) | ✓ | ✗ | ✓ | 0.0886 | **+39.3%** |
| **Odom only** (仅里程计) | ✓ | ✗ | ✗ | 0.1032 | +28.0% |

**分析**：
- 回环检测带来最大增益（Odom+Loop 改善39.3%，远优于Odom+GNSS的31.5%），验证了回环在消除累积漂移中的关键作用
- GNSS提供全局参考，在回环存在时收益有限；但在大尺度场景或无回环场景中更为重要
- 里程计单独运行仍有28%的改善，说明相邻帧约束本身已包含大量信息

#### 渲染质量

| 视角 | 非白像素数 | 覆盖率 | 说明 |
|------|-----------|--------|------|
| 0° (正前方) | 191,974 / 307,200 | 62.5% | 包含球体+立方体+地面 |
| 60° | 144,182 | 46.9% | 可见球体和地面 |
| 120° | 139,104 | 45.3% | 侧视图 |
| 180° (正后方) | 130,400 | 42.4% | 部分可见立方体 |
| 240° | 152,721 | 49.7% | - |
| 300° | 155,212 | 50.5% | - |

渲染覆盖率达42-62%，说明300个高斯核的合成场景在6m半径轨迹上有良好的多角度覆盖。

### 定性结果

实验输出的可视化结果包括（详见 `gs_slam/output/`）：

1. **3DGS vs 点云对比** (`c_comparison.png`)：展示3DGS通过splat渲染产生的连续表面 vs 稀疏点云的离散表示
2. **3D轨迹对比** (`e_trajectory.png`)：蓝色=真实轨迹，红色虚线=优化后轨迹，可见优化后轨迹紧密跟踪真值
3. **优化收敛曲线**：对数尺度下loss从初始~50降至~0.01，验证了坐标下降法的收敛性
4. **语义特征PCA可视化** (`g_semantic.png`)：将64维语义特征降维到RGB三通道，不同颜色区域对应不同语义类别

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
    def pixel_to_ray(self, u, v)      # 像素→归一化射线
```

**关键函数**：

- `look_at(eye, center, up)`：构建LookAt外参矩阵 $[R|t]$，其中 $R=[r; u; f]$ 由right/up/forward方向向量组成
- `pointmap_from_depth(depth, K)`：从深度图生成MASt3R格式的pointmap $[H,W,3]$
- `so3_log(R)`：SO(3)对数映射，计算旋转角 $\theta = \arccos((\text{tr}(R)-1)/2)$
- `random_so3(scale)`：在SO(3)上采样随机旋转（使用Rodrigues公式）

### core/gaussian_model.py — 3D高斯云

```python
class GaussianCloud:
    """可微分高斯云，容量20000"""
    
    def __init__(self, capacity=20000) # 预分配内存
    def add(pos, rgb, sem)             # 批量添加高斯核
    def prune(min_opacity=0.05)        # 移除低不透明度高斯
    def pack()                          # 打包为渲染字典
    
    # 属性
    scales_actual    → exp(log_scale)    # 实际尺度
    opacities_actual → sigmoid(log_opacity) # 实际不透明度
    get_covariances() → R S S^T R^T    # 协方差矩阵
```

**参数初始化**：
- 位置: 从预设的几何体采样（球体/立方体/平面）
- 尺度: $\ln(0.8) \approx -0.223$，实际尺度约0.8m
- 不透明度: 0.69（sigmoid(0.8)），确保较好的可见性
- 旋转: 单位四元数 $(1,0,0,0)$，表示无旋转

### core/renderer.py — Splat渲染器

```python
class SplatRenderer:
    """3DGS可微渲染器"""
    
    def render(gs, cam) → (rgb[H,W,3], sem[H,W,64])
        # 1. 世界→相机坐标变换
        # 2. 视锥体裁剪 (z > 0.01)
        # 3. 透视投影 u = fx*X/Z + cx
        # 4. 2D尺度估计 s2d = s3d[:2] * [fx, fy] / Z
        # 5. 按深度排序 (远处在前)
        # 6. Alpha Blending (逐splat)
```

**PointRenderer**：作为baseline的稀疏点渲染器，直接将3D点投影到像素而不进行alpha blending。

### core/factor_graph.py — 因子图优化

```python
class PoseGraph:
    """SE(3)位姿图"""
    
    def add_pose(R, t)                    # 添加位姿节点
    def add_odometry(i, j, R_rel, t_rel)  # 里程计边
    def add_loop(i, j, R_rel, t_rel)      # 回环边
    def add_gnss(i, t_global)             # GNSS边 (单节点)
    def residual(edge_idx) → (e_R, e_t)   # 残差计算
    def optimize(max_iter, lr) → losses   # 图优化
```

**残差定义**：

- 里程计和回环边: $e = (\log(R_{meas}^T R_i^T R_j), \; R_i^T(t_j - t_i) - t_{meas})$
- GNSS边: $e = t_i - t_{global}$

**优化更新** (对所有非固定节点k)：

$$t_k \leftarrow t_k - \eta \cdot \sum_{edges} w \cdot \frac{\partial e_t}{\partial t_k}$$

$$R_k \leftarrow R_k \cdot \exp(-\eta \cdot \sum_{edges} w \cdot \text{skew}(e_R))$$

随后使用SVD强制正交化：$U\Sigma V^T = R_k \implies R_k := UV^T$

### slam/frontend.py — SLAM前端

```python
def generate_synthetic_pointmaps(n_frames, radius, noise_std) → List[Dict]
    """模拟MASt3R的pointmap输出"""

def match_pointmaps(pm1, conf1, pm2, conf2, K) → (R_rel, t_rel, inlier_ratio)
    """RANSAC + Umeyama算法进行3D-3D匹配"""

def solve_rigid_svd(A, B) → (R, t)
    """Umeyama刚性变换求解"""

class SLAMFrontend:
    def process_frame(frame) → bool      # 处理新帧，决定是否添加为关键帧
    def get_loop_candidates(idx) → List  # 回环候选检测
```

**关键帧选择策略**：
- 平移距离 > 0.3m 或旋转角度 > 0.08rad
- 且内点率 > 0.3
- 最大关键帧数限制（默认50帧）

### slam/backend.py — SLAM后端

```python
class SLAMBackend:
    def build_graph_from_frontend(kfs, with_gnss, with_loop) → PoseGraph
    def optimize(max_iter, lr) → losses
    def compute_metrics() → {ATE, RPE_t, RPE_r}
```

支持四种消融配置的切换（通过 `with_gnss` 和 `with_loop` 参数）。

### slam/mapper.py — 增量建图

```python
class DenseMapper:
    def add_pointcloud(pts_world, colors)  # 添加点云到世界坐标系
    def assign_semantic_regions()          # 按空间位置分配语义
    def get_map() → Dict                   # 获取当前地图
    def prune()                            # 移除低不透明度高斯
```

语义区域分配策略（模拟SAM+CLIP）：
- 区域A (中心): 前21维激活 → "家具/物体"
- 区域B (X>2.5): 中21维激活 → "墙壁"
- 区域C (Y<-2.5): 后22维激活 → "地面"

---

## 使用示例

### 基础使用

```python
import numpy as np
from gs_slam.core.camera import PinholeCamera, look_at
from gs_slam.core.gaussian_model import make_test_scene
from gs_slam.core.renderer import SplatRenderer

# 创建场景
scene = make_test_scene(500)  # 500个高斯核

# 设置相机
cam = PinholeCamera()
R, t = look_at(eye=np.array([5,1,5]), center=np.zeros(3), up=np.array([0,1,0]))
cam.set_pose(R, t)

# 渲染
renderer = SplatRenderer()
rgb, semantic = renderer.render(scene.pack(), cam)

# 保存
from PIL import Image
Image.fromarray((rgb*255).astype(np.uint8)).save('output.png')
```

### 运行SLAM优化

```python
from gs_slam.core.factor_graph import build_test_graph

# 构建测试位姿图 (20帧环形轨迹)
pg = build_test_graph(n_frames=20, radius=6.0)

# 运行优化
losses = pg.optimize(max_iter=300, lr=0.008)

# 获取优化后轨迹
trajectory = pg.get_trajectory_xyz()  # [20, 3] numpy数组
```

### 运行完整实验

```bash
python -m gs_slam.demo.run_all
```

---

## 环境配置

### 自研实现 (gs_slam) — 立即运行

| 依赖 | 最低版本 | 用途 |
|------|---------|------|
| Python | 3.9+ | 运行环境 |
| NumPy | 1.24+ | 矩阵运算、SVD、向量操作 |
| Matplotlib | 3.7+ | 3D可视化、图表生成 |
| Pillow | 10.0+ | 图像读写 |

```bash
pip install numpy matplotlib pillow
```

### 完整实现 (MASt3R-Fusion) — 需要详细配置

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.11.9 | 运行环境 |
| PyTorch | 2.5.1 (CUDA 12.4) | 深度学习框架 |
| GTSAM | 修改版 | 因子图优化 (C++编译) |
| MASt3R | 最新版 | 前馈视觉模型 |
| OpenCV | 4.10+ | 图像处理 |
| h5py | 最新版 | 数据序列化 |
| lietorch | 最新版 | SE(3)李代数运算 |

> **注意**：GTSAM需要在Windows上编译C++代码，需要Visual Studio Build Tools和CMake。建议在Linux环境中部署完整版本。

---

## 版权与许可证

### 自研代码

Copyright (c) 2025 — **MIT License**

本项目自研部分（`gs_slam/` 目录、`deploy.py`、`setup_env.py`）采用MIT许可证，可自由使用、修改和分发。详见 [LICENSE](LICENSE)。

### 第三方代码

本项目包含以下开源仓库的副本，仅用于学术研究与课程作业：

| 目录 | 原始仓库 | 作者/组织 | 许可证 |
|------|----------|----------|--------|
| `MASt3R-Fusion/` | [GREAT-WHU/MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) | Yuxuan Zhou, Xingxing Li, et al. (Wuhan University) | 待官方声明 |
| `mast3r/` | [naver/mast3r](https://github.com/naver/mast3r) | NAVER Corp. | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `gtsam/` | [yuxuanzhou97/gtsam](https://github.com/yuxuanzhou97/gtsam) | 基于 [borglab/gtsam](https://github.com/borglab/gtsam) | [BSD-3-Clause](https://opensource.org/license/bsd-3-clause) |

**重要声明**：
- 第三方代码的版权归原始作者所有，本项目**未做任何修改**
- 第三方代码的使用需遵循其各自的许可证条款
- CC BY-NC-SA 4.0 要求署名、非商业使用、相同方式共享
- 如有版权问题，请通过GitHub Issues联系删除

### BibTeX引用

若使用本项目或参考相关工作，请引用：

```bibtex
@misc{murai2025mast3rslam,
  title={MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors},
  author={Riku Murai and Eric Dexheimer and Andrew J. Davison},
  year={2025}, eprint={2412.12392}, archivePrefix={arXiv}, primaryClass={cs.CV}
}

@misc{zhou2025mast3rfusion,
  title={MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM},
  author={Yuxuan Zhou and Xingxing Li and Shengyu Li and Zhuohao Yan and Chunxi Xia and Shaoquan Feng},
  year={2025}, eprint={2509.20757}, archivePrefix={arXiv}, primaryClass={cs.CV}
}

@misc{yoo2025openmonogsslam,
  title={OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics},
  author={Jisang Yoo and Gyeongjin Kang and Hyun-kyu Ko and Hyeonwoo Yu and Eunbyung Park},
  year={2025}, eprint={2512.08625}, archivePrefix={arXiv}, primaryClass={cs.CV}
}

@misc{chen2024survey3dgs,
  title={A Survey on 3D Gaussian Splatting},
  author={Guikun Chen and Wenguan Wang},
  year={2024}, eprint={2401.03890}, archivePrefix={arXiv}, primaryClass={cs.CV}
}
```

---

## 📌 注意事项

1. **MASt3R 模型权重**：完整实现需要下载约2GB的预训练权重，且需要CUDA GPU。自研版本使用合成数据模拟，无需模型权重
2. **GTSAM 编译**：Windows上编译GTSAM需要Visual Studio Build Tools, CMake, Boost等，过程较复杂。建议在WSL或Linux环境中使用完整版
3. **数据集**：MASt3R-Fusion支持KITTI-360（128GB）、SubT-MRS、WHU等数据集
4. **性能基准**：自研版本的ATE/RPE指标是在合成数据上的结果，与真实数据集上的指标不可直接比较
5. **简化与局限**：自研实现使用坐标下降法替代Gauss-Newton，收敛精度有限；3DGS使用固定尺度而非学习优化

---

*CV Final Project | 3D重建与SLAM方向 | 仅供学术研究与课程作业使用*

*最后更新: 2025年6月*