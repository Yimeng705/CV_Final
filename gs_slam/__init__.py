"""
3D Gaussian Splatting 增强的视觉SLAM系统

基于论文:
- [1] MASt3R-SLAM (Murai et al., 2024): 基于MASt3R的实时单目稠密SLAM
  - 两视图3D重建先验 + 点图匹配 + 二阶全局优化
  - GitHub: https://edexheim.github.io/mast3r-slam
- [2] MASt3R-Fusion (Zhou et al., 2025): 多传感器融合SLAM
  - Sim(3)视觉约束 + SE(3)因子图 + IMU/GNSS融合
  - GitHub: https://github.com/GREAT-WHU/MASt3R-Fusion
  - [3] OpenMonoGS-SLAM (Yoo et al., 2025): 单目3DGS-SLAM+开放集语义
  - SAM/CLIP语义 + 记忆机制
  - 项目页: https://jisang1528.github.io/OpenMonoGS-SLAM
- [4] A Survey on 3D Gaussian Splatting (Chen & Wang, 2026, TPAMI)
  - 3DGS领域首篇系统性综述
  - GitHub: https://github.com/guikunchen/Awesome3DGS

系统架构 (改进版):
- 前端: 点图提取与匹配 (仿MASt3R的2-view匹配 + RANSAC+Umeyama)
- 后端: 因子图优化 (仿MASt3R-SLAM的二阶优化 + MASt3R-Fusion的多传感器融合)
- 建图: 增量3DGS建图 (仿OpenMonoGS-SLAM的高斯地图表示)
- 改进: 语义感知的自适应密度控制 (结合3DGS综述method-003与OpenMonoGS-SLAM语义)

运行:
  python -m gs_slam.demo.run_all     # 运行完整实验管线
  # 然后打开 gs_slam/demo/frontend.html 进行交互演示
"""

__version__ = "1.0.0"