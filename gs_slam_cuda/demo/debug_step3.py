"""
Test exactly what step3 of run_all.py does with helical trajectory.
Output to C:\temp\debug_step3_output.txt
"""
import sys, os
import torch
import numpy as np

OUTPUT = r"C:\temp\debug_step3_output.txt"
lines = []

def pr(s=""):
    text = str(s)
    print(text)
    lines.append(text)

try:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if base not in sys.path:
        sys.path.insert(0, base)
    
    from gs_slam_cuda.core.camera import PinholeCamera, look_at, generate_helical_trajectory
    from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda
    from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer
    
    device = torch.device('cuda:0')
    
    # Exactly replicate step2 + step3 flow
    pr("=== Replicating run_all.py step2+step3 ===")
    
    gc = create_test_scene_cuda(device=device, n_gaussians=1200)
    pr(f"Created scene: N={len(gc)}")
    
    poses = generate_helical_trajectory(n_poses=50, radius=10.0, height_range=(-2.0, 4.0))
    pr(f"Helical poses: {len(poses)}")
    
    # Check first pose
    R0, t0 = poses[0]
    pr(f"Pose[0]: R det={np.linalg.det(R0):.4f}")
    pr(f"  t = {t0.flatten()}")
    
    renderer = CUDASplatRenderer(image_height=480, image_width=640, use_fp16=False, device=device)
    gs_dict = gc.pack()
    
    pr(f"\nPacked: xyz={gs_dict['xyz'].shape}, cov={gs_dict['cov'].shape}")
    pr(f"  cov trace: {gs_dict['cov'].diagonal(dim1=1,dim2=2).sum(dim=1).mean().item():.4f}")
    
    n_views = 6
    for i in range(n_views):
        R, t = poses[i]
        cam = PinholeCamera()
        cam.set_pose(R, t)
        
        pr(f"\n--- View {i} ---")
        pr(f"  cam.fx={cam.fx}, cam.cx={cam.cx}")
        
        # Manual projection check
        xyz_gpu = gs_dict['xyz'].to(device)
        R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
        t_t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(3, 1)
        pts_cam = (R_t @ xyz_gpu.T + t_t).T
        depth_cam = pts_cam[:, 2]
        valid = depth_cam > 0.01
        pr(f"  valid depth: {valid.sum().item()}/{len(gc)}")
        pr(f"  Z range: {depth_cam.min().item():.1f} ~ {depth_cam.max().item():.1f}")
        
        u_cam = cam.fx * pts_cam[:, 0] / depth_cam.clamp(min=0.01) + cam.cx
        v_cam = cam.fy * pts_cam[:, 1] / depth_cam.clamp(min=0.01) + cam.cy
        in_img = valid & (u_cam >= 0) & (u_cam < 640) & (v_cam >= 0) & (v_cam < 480)
        pr(f"  in image: {in_img.sum().item()}/{valid.sum().item()}")
        
        # Check cov2d
        cov3d_gpu = gs_dict['cov'].to(device)
        cov2d = renderer.project_covariance_2d_full(cov3d_gpu, pts_cam, cam.fx, cam.fy)
        radius = renderer.get_radius_from_cov2d(cov2d)
        pr(f"  cov2d diag mean: xx={cov2d[:,0,0].mean().item():.1f}, yy={cov2d[:,1,1].mean().item():.1f}")
        pr(f"  radius: {radius.min().item():.1f}~{radius.max().item():.1f}, r>0.5: {(radius>0.5).sum().item()}")
        
        # CHECK: tile-based render path
        tile_size = 16
        tiles_W = (640 + 15) // 16
        tiles_H = (480 + 15) // 16
        tile_x0 = ((u_cam - radius) / tile_size).long().clamp(0, tiles_W - 1)
        tile_x1 = ((u_cam + radius) / tile_size).long().clamp(0, tiles_W - 1)
        tile_y0 = ((v_cam - radius) / tile_size).long().clamp(0, tiles_H - 1)
        tile_y1 = ((v_cam + radius) / tile_size).long().clamp(0, tiles_H - 1)
        n_tiles_per_gs = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
        valid_tile = valid & (n_tiles_per_gs > 0) & (radius > 0.5)
        total_tile_slots = int(n_tiles_per_gs.sum().item())
        pr(f"  valid_tile count: {valid_tile.sum().item()}")
        pr(f"  total_tile_slots: {total_tile_slots}")
        
        # Actually render
        rgb, depth = renderer.forward(gs_dict, cam)
        coverage = (depth < float('inf')).sum().item() / (480 * 640) * 100
        rgb_np = rgb.cpu().numpy()
        pr(f"  RGB min={rgb_np.min():.4f}, max={rgb_np.max():.4f}, mean={rgb_np.mean():.4f}")
        pr(f"  Coverage: {coverage:.2f}%")
        
        if coverage < 1.0:
            pr(f"  ** WARNING: Low coverage!")
    
    pr("\n### DONE ###")
    
except Exception as e:
    import traceback
    pr(f"ERROR: {e}")
    pr(traceback.format_exc())

finally:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nOutput saved to {OUTPUT}")