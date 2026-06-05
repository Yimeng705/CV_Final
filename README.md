# 3D Gaussian Splatting 增强的视觉SLAM系统

> **CV课程大作业 — 3D重建与SLAM方向**

基于2024-2025年顶级会议论文实现的端到端SLAM与3D重建系统。

---

## 📚 参考文献

| # | 论文 | 作者 | 年份 | 链接 |
|---|------|------|------|------|
| 1 | **MASt3R-SLAM**: Real-Time Monocular Dense SLAM from MASt3R | Riku Murai, Eric Dexheimer, Andrew J. Davison | 2024 | [arXiv:2412.12392](https://arxiv.org/abs/2412.12392) |
| 2 | **MASt3R-Fusion**: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM | Yuxuan Zhou, Xingxing Li, Shengyu Li, et al. | 2025 | [arXiv:2509.20757](https://arxiv.org/abs/2509.20757) |
| 3 | **OpenMonoGS-SLAM**: Monocular Gaussian Splatting SLAM with Open-set Semantics | Jisang Yoo, Gyeongjin Kang, Hyun-kyu Ko, et al. | 2025 | [arXiv:2512.08625](https://arxiv.org/abs/2512.08625) |
| 4 | **A Survey on 3D Gaussian Splatting** (综述) | Guikun Chen, Wenguan Wang | 2024 | [arXiv:2401.03890](https://arxiv.org/abs/2401.03890) |

---

## 🏗️ 项目结构

```
final/
│
├── README.md                     # 本文件
├── LICENSE                       # MIT License
├── .gitignore
├── deploy.py                     # 统一部署入口
├── setup_env.py                  # 环境检查与依赖安装
│
├── gs_slam/                      # [自研] 纯NumPy实现
│   ├── core/
│   │   ├── camera.py             #   相机模型、LookAt外参、SO(3)李代数
│   │   ├── gaussian_model.py     #   3D高斯云参数化
│   │   ├── renderer.py           #   Splat渲染器 + Alpha Blending
│   │   └── factor_graph.py       #   位姿图优化
│   ├── slam/
│   │   ├── frontend.py           #   前端: pointmap匹配 + RANSAC
│   │   ├── backend.py            #   后端: 多传感器因子图优化
│   │   └── mapper.py             #   建图: 增量3DGS + 语义
│   ├── demo/
│   │   └── run_all.py            #   [主入口] 完整实验管线
│   └── output/
│       ├── report.html           #   HTML综合实验报告
│       └── h_ablation.txt        #   消融实验数据
│
├── MASt3R-Fusion/                # 第三方: GREAt-WHU/MASt3R-Fusion
├── mast3r/                       # 第三方: naver/mast3r
└── gtsam/                        # 第三方: yuxuanzhou97/gtsam
```

---

## 🚀 快速开始

### 方式一：自研实现 (推荐，立即可用)

**仅需 `numpy + matplotlib + pillow`，无需GPU/深度学习框架。**

```bash
pip install numpy matplotlib pillow
python -m gs_slam.demo.run_all
# 然后打开 gs_slam/output/report.html 查看完整实验报告
```

**包含的实验：**
1. ✅ 3D高斯场景渲染 (球体+立方体+平面)
2. ✅ 多视角新视图合成 (6个角度)
3. ✅ 3DGS vs 传统点云渲染对比
4. ✅ SLAM因子图优化 (ATE从0.15m降至0.095m，改善38.4%)
5. ✅ 增量3DGS建图 + 开放集语义
6. ✅ 消融实验 (4种配置)

### 方式二：完整MASt3R-Fusion (需要CUDA + GTSAM + 模型权重)

请参考 [MASt3R-Fusion官方文档](https://github.com/GREAT-WHU/MASt3R-Fusion) 完成环境配置。

---

## 📊 实验结果摘要

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| ATE (绝对轨迹误差) | 0.1541 m | 0.0950 m | **↓ 38.4%** |
| RPE-t (相对平移误差) | 0.1330 m | 0.0812 m | **↓ 39.0%** |
| 3DGS渲染覆盖 | - | 138K-192K 像素 (45-62%) | - |

### 消融实验

| 配置 | ATE (m) | 改善 |
|------|---------|------|
| Full (里程计+GNSS+回环) | 0.1011 | +28.4% |
| 里程计+GNSS (无回环) | 0.0982 | +31.5% |
| 里程计+回环 (无GNSS) | 0.0886 | +39.3% |
| 仅里程计 | 0.1032 | +28.0% |

---

## 🔧 系统架构

```
┌──────────────────────────────────────────────────────┐
│                    SLAM 系统                          │
├──────────────┬───────────────┬────────────────────────┤
│   前端       │    后端       │       建图             │
│ (Frontend)   │  (Backend)    │     (Mapper)           │
├──────────────┼───────────────┼────────────────────────┤
│ MASt3R       │ 因子图优化     │ 3D Gaussian Splatting │
│ pointmap匹配  │              │                        │
│              │ · 里程计因子   │ · μ,q,s,α,c 参数       │
│ · RANSAC     │ · GNSS因子    │ · Splat渲染            │
│ · Umeyama    │ · 回环因子    │ · 语义特征关联          │
│ · 关键帧选择  │ · 滑动窗口BA  │ · 增量更新              │
└──────────────┴───────────────┴────────────────────────┘
```

---

## ⚖️ 版权与许可证

### 自研代码 (`gs_slam/`, `deploy.py`, `setup_env.py`)

Copyright (c) 2025 — MIT License

本项目的自研部分（`gs_slam/` 目录、`deploy.py`、`setup_env.py`）采用 MIT 许可证，可自由使用、修改和分发。详见 [LICENSE](LICENSE) 文件。

### 第三方代码

本项目包含以下开源仓库的副本，仅用于学术研究与课程作业：

| 目录 | 原始仓库 | 许可证 |
|------|----------|--------|
| `MASt3R-Fusion/` | [GREAT-WHU/MASt3R-Fusion](https://github.com/GREAT-WHU/MASt3R-Fusion) | 待官方声明 |
| `mast3r/` | [naver/mast3r](https://github.com/naver/mast3r) | CC BY-NC-SA 4.0 |
| `gtsam/` | [yuxuanzhou97/gtsam](https://github.com/yuxuanzhou97/gtsam) | BSD-3-Clause |

**重要声明：**
- 第三方代码的版权归原始作者所有
- 第三方代码的使用需遵循其各自的许可证条款
- 本项目不对第三方代码做任何修改，仅作为依赖引用
- 如有版权问题，请联系删除

### 论文引用

如果您使用本项目或参考了相关工作，请引用以下论文：

```bibtex
@misc{murai2024mast3rslam,
  title={MASt3R-SLAM: Real-Time Monocular Dense SLAM from MASt3R},
  author={Riku Murai and Eric Dexheimer and Andrew J. Davison},
  year={2024}, eprint={2412.12392}, archivePrefix={arXiv}
}

@misc{zhou2025mast3rfusion,
  title={MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for SLAM},
  author={Yuxuan Zhou and Xingxing Li and Shengyu Li and Zhuohao Yan and Chunxi Xia and Shaoquan Feng},
  year={2025}, eprint={2509.20757}, archivePrefix={arXiv}
}

@misc{yoo2025openmonogsslam,
  title={OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics},
  author={Jisang Yoo and Gyeongjin Kang and Hyun-kyu Ko and Hyeonwoo Yu and Eunbyung Park},
  year={2025}, eprint={2512.08625}, archivePrefix={arXiv}
}

@misc{chen2024survey3dgs,
  title={A Survey on 3D Gaussian Splatting},
  author={Guikun Chen and Wenguan Wang},
  year={2024}, eprint={2401.03890}, archivePrefix={arXiv}
}
```

---

## 🛠️ 依赖

### 自研实现 (gs_slam)
```
numpy >= 1.24
matplotlib >= 3.7
pillow >= 10.0
```

### 完整实现 (MASt3R-Fusion)
```
PyTorch >= 2.5.1 (CUDA)
GTSAM (修改版)
lietorch / opencv-python / mast3r / h5py
```

---

*CV Final Project | 3D重建与SLAM方向 | 仅供学术研究与课程作业使用*