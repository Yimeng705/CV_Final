# 3D Gaussian Splatting 增强的视觉SLAM系统 

基于以下4篇顶级论文的实现:
- [MASt3R-SLAM](https://arxiv.org/abs/2412.12392) (Murai et al., 2024, ICCV 2025)
- [MASt3R-Fusion](https://arxiv.org/abs/2509.20757) (Zhou et al., 2025, AAAI 2026)
- [OpenMonoGS-SLAM](https://arxiv.org/abs/2512.08625) (Yoo et al., 2025, CVPR 2025)
- [A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890) (Chen & Wang, 2026, TPAMI)

## 改进方法: 语义感知的自适应高斯密度控制

基于对四篇论文的深入分析，我们提出了一个方法改进:
将OpenMonoGS-SLAM的语义特征与3DGS综述的自适应密度控制相结合，
在语义边界区域增加高斯密度，提升重建细节质量。

### 核心算法实现

```
gs_slam/
├── core/               # 核心算法
│   ├── camera.py              # 相机模型与投影
│   ├── gaussian_model.py      # 3DGS模型 (论文[4])
│   ├── renderer.py            # Tile-based可微渲染器 + 深度图
│   ├── factor_graph.py        # 因子图优化 (论文[1][2])
│   └── adaptive_density.py    # [改进] 语义感知密度控制
├── slam/               # SLAM系统
│   ├── frontend.py            # 前端: 点图提取与匹配 (论文[1])
│   ├── backend.py             # 后端: 全局BA优化
│   └── mapper.py              # 建图: 增量高斯地图 + 密度控制 (论文[3]+[改进])
├── demo/               # 演示
│   ├── run_all.py             # 完整实验管线 (6步骤)
│   ├── frontend.html          # Web交互演示界面
│   └── test_imports.py        # 导入测试
└── output/             # 实验结果 (17个文件)
```

## 运行

### 完整实验管线
```bash
python -m gs_slam.demo.run_all
```

### 6个实验步骤
1. **3DGS渲染**: 300个高斯核场景渲染 + 点云/深度图对比
2. **多视角合成**: 6个角度(0°-300°)新视图合成
3. **SLAM因子图**: 20帧环形轨迹 + 里程计/GNSS/回环优化, ATE降至0.081m
4. **增量建图**: 8个关键帧的稠密3DGS建图 + 语义区域分配
5. **消融实验**: 有/无回环、有/无GNSS四种配置, ATE改善27-37%
6. **方法改进验证**: 4种语义权重的密度控制对比, 增长比17.65x→18.43x

### 交互演示
```bash
# 运行实验后, 用浏览器打开:
gs_slam/demo/frontend.html