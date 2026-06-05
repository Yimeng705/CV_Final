# 3D Gaussian Splatting 增强的视觉SLAM系统

基于以下4篇顶级论文的实现:
- [MASt3R-SLAM](https://arxiv.org/abs/2412.12392) (Riku Murai et al., 2024)
- [MASt3R-Fusion](https://arxiv.org/abs/2509.20757) (Zhou et al., 2025)
- [OpenMonoGS-SLAM](https://arxiv.org/abs/2512.08625) (Yoo et al., 2025)
- [A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890) (Chen & Wang, 2024)

## 核心算法实现

```
gs_slam/
├── core/           # 核心算法
│   ├── camera.py           # 相机模型与投影
│   ├── gaussian_model.py   # 3DGS模型 (论文[4])
│   ├── renderer.py         # 可微渲染器
│   └── factor_graph.py     # 因子图优化 (论文[1][2])
├── slam/           # SLAM系统
│   ├── frontend.py         # 前端: 特征提取与匹配 (论文[1])
│   ├── backend.py          # 后端: 全局BA优化
│   └── mapper.py           # 建图: 增量高斯地图 (论文[3])
├── demo/           # 演示
│   └── run_all.py          # 完整实验管道
└── output/         # 实验结果输出
```

## 运行
```bash
python -m gs_slam.demo.run_all