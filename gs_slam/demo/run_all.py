"""
完整实验演示
============
基于4篇论文的完整SLAM+3DGS系统实验

运行: python -m gs_slam.demo.run_all
输出: gs_slam/output/ 目录下的全部结果
"""

import sys
import os
import time
import numpy as np

# 路径设置
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ_ROOT)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from gs_slam.core.camera import PinholeCamera, look_at, so3_log
from gs_slam.core.gaussian_model import GaussianCloud, make_test_scene
from gs_slam.core.renderer import SplatRenderer, PointRenderer
from gs_slam.core.factor_graph import PoseGraph, build_test_graph
from gs_slam.slam.frontend import generate_synthetic_pointmaps, SLAMFrontend
from gs_slam.slam.backend import SLAMBackend
from gs_slam.slam.mapper import DenseMapper

np.random.seed(42)

def header(s):
    print(f"\n{'='*60}\n  {s}\n{'='*60}")

# ================ Step 1: 3DGS场景渲染 ================
def step1_render():
    header("Step 1: 3D Gaussian Splatting 场景渲染 (论文[4]: 3DGS综述)")
    gc = make_test_scene(300)
    print(f"  ✓ 创建了 {len(gc)} 个高斯核 (球体+立方体+平面)")
    
    renderer = SplatRenderer()
    pt_renderer = PointRenderer()
    
    cam = PinholeCamera()
    eye = np.array([5.0, 1.5, 5.0], dtype=np.float32)
    R, t = look_at(eye, np.zeros(3), np.array([0.,1.,0.]))
    cam.set_pose(R, t)
    
    gs_data = gc.pack()
    t0 = time.time()
    rgb_gs, sem = renderer.render(gs_data, cam)
    t_gs = time.time() - t0
    print(f"  ✓ 3DGS渲染: {t_gs:.3f}s, 非白像素: {(rgb_gs.sum(-1)<2.9).sum()}/{480*640}")
    
    rgb_pt = pt_renderer.render(gs_data, cam)
    
    Image.fromarray((np.clip(rgb_gs,0,1)*255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, 'a_3dgs_render.png'))
    Image.fromarray(rgb_pt).save(os.path.join(OUT_DIR, 'b_pointcloud.png'))
    
    # 对比图
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].imshow((np.clip(rgb_gs,0,1)*255).astype(np.uint8)); axes[0].set_title('3DGS (Ours)'); axes[0].axis('off')
    axes[1].imshow(rgb_pt); axes[1].set_title('Point Cloud'); axes[1].axis('off')
    plt.suptitle('Rendering Comparison: 3DGS vs Point Cloud', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'c_comparison.png'), dpi=120); plt.close()
    print("  [OK] 3DGS渲染对比图已保存")
    
    return gc, renderer

# ================ Step 2: 多视角渲染 ================
def step2_multiview(gc, renderer):
    header("Step 2: 多视角新视图合成 (论文[4])")
    for deg in [0, 60, 120, 180, 240, 300]:
        a = deg/180*np.pi
        eye = np.array([6*np.cos(a), 1.0, 6*np.sin(a)], dtype=np.float32)
        R, t = look_at(eye, np.zeros(3), np.array([0.,1.,0.]))
        cam = PinholeCamera(); cam.set_pose(R, t)
        rgb, _ = renderer.render(gc.pack(), cam)
        img = (np.clip(rgb,0,1)*255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(OUT_DIR, f'd_view_{deg:03d}.png'))
        nz = (rgb.sum(-1)<2.9).sum()
        print(f"  ✓ {deg:3d}°: 非白像素={nz}/{480*640}")

# ================ Step 3: SLAM因子图优化 ================
def step3_slam():
    header("Step 3: SLAM因子图优化 (论文[1]MASt3R-SLAM + 论文[2]MASt3R-Fusion)")
    
    # 使用前端生成合成pointmaps
    kfs = generate_synthetic_pointmaps(n_frames=20, radius=6.0, noise_std=0.03)
    print(f"  ✓ 生成了 {len(kfs)} 帧合成点图 (模拟MASt3R输出)")
    
    # 后端构建因子图并优化
    backend = SLAMBackend()
    pg = backend.build_graph_from_frontend(kfs, with_gnss=True, with_loop=True)
    
    # 优化前ATE
    before = backend._compute_ate(pg.poses, backend.clean_poses)
    print(f"  [Before] ATE: {before:.4f}m")
    
    # 运行优化
    t0 = time.time()
    losses = backend.optimize(max_iter=300, lr=0.008)
    t_opt = time.time() - t0
    
    after = backend._compute_ate(pg.poses, backend.clean_poses)
    improvement = (before-after)/before*100
    print(f"  [After]  ATE: {after:.4f}m ({improvement:.1f}% improvement)")
    print(f"  ✓ 优化完成: {t_opt:.2f}s, {len(losses)}迭代")
    
    # 轨迹可视化
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(121, projection='3d')
    gt_xyz = np.array([t.flatten() for _,t in backend.clean_poses])
    est_xyz = np.array([t.flatten() for _,t in pg.poses])
    ax.plot(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], 'b-', lw=2, label='GT')
    ax.plot(est_xyz[:,0], est_xyz[:,1], est_xyz[:,2], 'r--', lw=2, label='Optimized')
    ax.scatter(*gt_xyz[0], c='g', s=100, label='Start')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Trajectory Comparison'); ax.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(losses, 'b-', lw=1); ax2.set_yscale('log')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Total Loss')
    ax2.set_title('Factor Graph Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'e_trajectory.png'), dpi=120); plt.close()
    print("  [OK] 轨迹对比图已保存")
    
    metrics = backend.compute_metrics()
    return kfs, pg, metrics, losses, improvement

# ================ Step 4: 增量建图 ================
def step4_mapping(kfs, pg):
    header("Step 4: 增量3DGS建图 (论文[3]OpenMonoGS-SLAM)")
    
    mapper = DenseMapper(5000)
    
    # 使用优化后的位姿将点云注册到世界坐标系
    for i, kf in enumerate(kfs[:5]):  # 使用前5个关键帧
        R_opt, t_opt = pg.poses[i]
        R_gt = kf['R_gt']
        
        # 从pointmap提取有效3D点 (相机坐标系)
        pm = kf['pointmap']
        conf = kf['confidence']
        valid = (conf > 0.5) & (pm[:,:,2] > 0.01)
        pts_cam = pm[valid][:200]  # 每帧最多200点
        
        # 变换到世界坐标系
        pts_world = (R_opt.T @ pts_cam.T - R_opt.T @ t_opt).T
        colors = np.tile(np.random.rand(3), (len(pts_world), 1)).astype(np.float32)
        
        mapper.add_pointcloud(pts_world.astype(np.float32), colors)
    
    print(f"  ✓ 建图完成: {mapper.size()} 个高斯核")
    
    # 语义分配
    sem = mapper.assign_semantic_regions()
    
    # 渲染建图结果
    renderer = SplatRenderer()
    cam = PinholeCamera()
    R, t = look_at(np.array([3.,2.,4.]), np.array([0.,1.,0.]), np.array([0.,1.,0.]))
    cam.set_pose(R, t)
    
    rgb, sem_map = renderer.render(mapper.get_map(), cam)
    
    # 语义可视化 (简单PCA)
    H, W = sem_map.shape[:2]
    sem_flat = sem_map.reshape(-1, 64)
    nonzero = sem_flat.sum(-1) > 0.01
    if nonzero.sum() > 10:
        centered = sem_flat[nonzero] - sem_flat[nonzero].mean(0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca3 = centered @ Vt[:3].T
        pca3 = (pca3 - pca3.min(0)) / (pca3.max(0) - pca3.min(0) + 1e-8)
        sem_viz = np.zeros((H*W, 3)); sem_viz[nonzero] = pca3; sem_viz = sem_viz.reshape(H,W,3)
    else:
        sem_viz = np.zeros((H,W,3))
    
    rgb_u8 = (np.clip(rgb,0,1)*255).astype(np.uint8)
    Image.fromarray(rgb_u8).save(os.path.join(OUT_DIR,'f_mapping_result.png'))
    
    # 语义图
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(rgb_u8); axes[0].set_title('Reconstructed RGB'); axes[0].axis('off')
    axes[1].imshow(sem_viz); axes[1].set_title('Semantic Features (PCA)'); axes[1].axis('off')
    overlay = (rgb_u8.astype(float)*0.6 + sem_viz*255*0.4).clip(0,255).astype(np.uint8)
    axes[2].imshow(overlay); axes[2].set_title('RGB + Semantics'); axes[2].axis('off')
    plt.suptitle('OpenMonoGS-SLAM: Open-set Semantic Mapping', fontweight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,'g_semantic.png'), dpi=120); plt.close()
    print("  [OK] 语义建图结果已保存")
    
    return mapper

# ================ Step 5: 方法对比实验 ================
def step5_ablation(kfs, pg):
    header("Step 5: 消融实验 (有/无回环, 有/无GNSS)")
    
    configs = [
        ("Full (Odometry+GNSS+Loop)", True, True),
        ("Odom+GNSS (w/o Loop)", True, False),
        ("Odom+Loop (w/o GNSS)", False, True),
        ("Odometry only", False, False),
    ]
    
    results = []
    for name, gnss, loop in configs:
        backend = SLAMBackend()
        pg_abl = backend.build_graph_from_frontend(kfs, with_gnss=gnss, with_loop=loop)
        before = backend._compute_ate(pg_abl.poses, backend.clean_poses)
        backend.optimize(max_iter=300, lr=0.008)
        after = backend._compute_ate(pg_abl.poses, backend.clean_poses)
        imp = (before-after)/before*100
        results.append((name, before, after, imp))
        print(f"  {name:30s}: ATE {before:.4f}->{after:.4f}m ({imp:+.1f}%)")
    
    # 保存消融结果
    with open(os.path.join(OUT_DIR, 'h_ablation.txt'), 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("="*60 + "\n")
        f.write(f"{'Configuration':<30} {'Before':>8} {'After':>8} {'Δ%':>8}\n")
        f.write("-"*60 + "\n")
        for name, b, a, imp in results:
            f.write(f"{name:<30} {b:8.4f} {a:8.4f} {imp:+7.1f}%\n")
    print("  [OK] 消融实验结果已保存")
    return results

# ================ 综合报告 ================
def generate_report(gc, metrics, improvement, results):
    header("生成HTML综合报告")
    
    metrics_str = "\n".join([f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" 
                              for k,v in metrics.items()])
    
    ablation_rows = ""
    for name, b, a, imp in results:
        ablation_rows += f"<tr><td>{name}</td><td>{b:.4f}</td><td>{a:.4f}</td><td style='color:{'green' if imp>0 else 'red'}'>{imp:+.1f}%</td></tr>"
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><title>3DGS-SLAM 实验报告</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial,sans-serif;background:#f0f2f5;color:#333}}
.c{{max-width:1200px;margin:0 auto;padding:20px}}
.hd{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:40px 20px;text-align:center;border-radius:12px;margin-bottom:30px}}
.hd h1{{font-size:2em}}
.card{{background:#fff;border-radius:12px;padding:24px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
.card h2{{color:#667eea;margin-bottom:16px;border-bottom:2px solid #667eea20;padding-bottom:8px}}
.metrics{{display:flex;gap:20px;flex-wrap:wrap}}
.mb{{flex:1;min-width:180px;background:linear-gradient(135deg,#667eea10,#764ba210);border-radius:10px;padding:20px;text-align:center}}
.mb .v{{font-size:2em;font-weight:bold;color:#667eea}}
.mb .l{{color:#666;margin-top:4px}}
table{{width:100%;border-collapse:collapse;margin:10px 0}}
th,td{{padding:10px 14px;text-align:center;border-bottom:1px solid #eee}}
th{{background:#667eea10;color:#667eea;font-weight:600}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:16px}}
.gallery img{{width:100%;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
.gallery .caption{{text-align:center;color:#666;margin-top:6px;font-size:.9em}}
.paper{{background:#f8f9fa;border-left:4px solid #667eea;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0}}
.footer{{text-align:center;color:#999;padding:20px;margin-top:30px;border-top:1px solid #eee}}
</style></head>
<body><div class="c">
<div class="hd">
<h1>3D Gaussian Splatting 增强的视觉SLAM系统</h1>
<p>基于 MASt3R-SLAM + MASt3R-Fusion + OpenMonoGS-SLAM + 3DGS综述 的实验</p>
</div>

<div class="card">
<h2>📊 实验指标</h2>
<div class="metrics">
<div class="mb"><div class="v">{metrics['ATE']:.4f}</div><div class="l">ATE (m)</div></div>
<div class="mb"><div class="v">{metrics['RPE_t']:.4f}</div><div class="l">RPE-t (m)</div></div>
<div class="mb"><div class="v">{metrics['RPE_r']:.4f}</div><div class="l">RPE-r (rad)</div></div>
<div class="mb"><div class="v">{gc._N}</div><div class="l">3D高斯核</div></div>
</div>
</div>

<div class="card">
<h2>🧪 消融实验</h2>
<table>
<tr><th>配置</th><th>优化前ATE</th><th>优化后ATE</th><th>改善</th></tr>
{ablation_rows}
</table>
</div>

<div class="card">
<h2>🖼️ 可视化</h2>
<div class="gallery">
<div><img src="a_3dgs_render.png"><div class="caption">图1: 3DGS渲染结果</div></div>
<div><img src="c_comparison.png"><div class="caption">图2: 3DGS vs 点云对比</div></div>
<div><img src="e_trajectory.png"><div class="caption">图3: SLAM轨迹对比与优化</div></div>
<div><img src="f_mapping_result.png"><div class="caption">图4: 增量建图结果</div></div>
<div><img src="g_semantic.png"><div class="caption">图5: 语义特征可视化</div></div>
</div>
</div>

<div class="card">
<h2>📚 参考文献</h2>
<div class="paper"><strong>[1] MASt3R-SLAM</strong> (Murai et al., 2024) - 基于MASt3R的实时单目稠密SLAM<br>
代码: https://edexheim.github.io/mast3r-slam</div>
<div class="paper"><strong>[2] MASt3R-Fusion</strong> (Zhou et al., 2025) - 前馈视觉+IMU/GNSS多传感器融合<br>
代码: https://github.com/GREAT-WHU/MASt3R-Fusion</div>
<div class="paper"><strong>[3] OpenMonoGS-SLAM</strong> (Yoo et al., 2025) - 单目3DGS+开放集语义SLAM<br>
代码: https://jisang1528.github.io/OpenMonoGS-SLAM</div>
<div class="paper"><strong>[4] 3DGS综述</strong> (Chen & Wang, 2024) - 3D Gaussian Splatting系统综述<br>
代码: https://github.com/guikunchen/Awesome3DGS</div>
</div>

<div class="card">
<h2>💡 实现要点</h2>
<ul style="padding-left:20px;line-height:2">
<li><strong>前端:</strong> 模拟MASt3R的pointmap输出, 使用RANSAC+Umeyama进行3D-3D匹配</li>
<li><strong>后端:</strong> 位姿图优化, 支持里程计/GNSS/回环多种因子</li>
<li><strong>建图:</strong> 增量3DGS建图, 关联语义特征向量</li>
<li><strong>渲染:</strong> 基于tile的splat渲染, 支持RGB+语义联合输出</li>
</ul>
</div>

<div class="footer">CV Final Project | 3D重建与SLAM方向 | 基于2024-2025年顶会论文</div>
</div></body></html>"""
    
    with open(os.path.join(OUT_DIR, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n  [OK] 综合报告: {os.path.join(OUT_DIR, 'report.html')}")

# ================ 主函数 ================
def main():
    print("╔" + "═"*58 + "╗")
    print("║  3DGS-SLAM 完整实验演示                                  ║")
    print("║  论文: MASt3R-SLAM × MASt3R-Fusion × OpenMonoGS-SLAM     ║")
    print("╚" + "═"*58 + "╝")
    
    # Step 1: 渲染
    gc, renderer = step1_render()
    
    # Step 2: 多视角
    step2_multiview(gc, renderer)
    
    # Step 3: SLAM
    kfs, pg, metrics, losses, improvement = step3_slam()
    
    # Step 4: 建图
    mapper = step4_mapping(kfs, pg)
    
    # Step 5: 消融
    results = step5_ablation(kfs, pg)
    
    # 报告
    generate_report(gc, metrics, improvement, results)
    
    # 总结
    print("\n" + "="*60)
    print(f"  所有实验完成! 输出目录: {OUT_DIR}")
    print("="*60)
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:30s} {sz/1024:8.1f} KB")
    print(f"\n  👉 打开 {os.path.join(OUT_DIR,'report.html')} 查看完整报告")


if __name__ == '__main__':
    main()