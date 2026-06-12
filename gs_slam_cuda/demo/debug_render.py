"""
Diagnostic script to find root cause of blank rendering (coverage=0.0%).
Output saved to C:\temp\debug_render_output.txt (avoids special-char path issues).
"""
import sys, os
import torch
import numpy as np

OUTPUT = r"C:\temp\debug_render_output.txt"
lines = []

def pr(s=""):
    text = str(s)
    print(text)
    lines.append(text)

try:
    # Use absolute import from the project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if base not in sys.path:
        sys.path.insert(0, base)
    
    from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda, GaussianCloudCUDA
    from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer
    from gs_slam_cuda.core.camera import PinholeCamera, look_at
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pr(f"Device: {device}")
    pr(f"CUDA available: {torch.cuda.is_available()}")
    pr("=" * 60)
    
    # Test 1: Create scene & inspect packed data
    pr("\n### TEST 1: Packed Data Inspection ###")
    gc = create_test_scene_cuda(device=device, n_gaussians=400)
    gs = gc.pack()
    pr(f"N={len(gc)}")
    
    for key in ['xyz', 'rgb', 'opacity', 'cov', 'scale', 'rot']:
        v = gs.get(key)
        if v is not None:
            pr(f"  {key}: shape={v.shape}, dtype={v.dtype}, "
                  f"min={v.min().item():.4f}, max={v.max().item():.4f}")
        else:
            pr(f"  {key}: None")
    
    cov = gs['cov']
    trace = cov.diagonal(dim1=1, dim2=2).sum(dim=1)
    pr(f"  cov trace: min={trace.min().item():.6f}, max={trace.max().item():.6f}, mean={trace.mean().item():.6f}")
    op = gs['opacity']
    pr(f"  opacity: min={op.min().item():.6f}, max={op.max().item():.6f}")
    
    # Test 2: Camera & Projection
    pr("\n### TEST 2: Projection & Radius ###")
    R, t = look_at(np.array([8., 5., 10.]), np.array([0., 1., 0.]))
    cam = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam.set_pose(R, t)
    pr(f"Camera pos: (8,5,10), target: (0,1,0)")
    pr(f"Det(R)={np.linalg.det(R):.4f}")
    
    renderer = CUDASplatRenderer(device=device, image_height=480, image_width=640)
    
    xyz = gs['xyz'].to(device)
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    t_t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(3, 1)
    pts_cam = (R_t @ xyz.T + t_t).T
    depth = pts_cam[:, 2]
    valid = depth > 0.01
    
    pr(f"  pts_cam Z: min={depth.min().item():.2f}, max={depth.max().item():.2f}")
    pr(f"  valid (z>0.01): {valid.sum().item()}/{xyz.shape[0]}")
    
    u = 500.0 * pts_cam[:, 0] / depth.clamp(min=0.01) + 320.0
    v = 500.0 * pts_cam[:, 1] / depth.clamp(min=0.01) + 240.0
    pr(f"  u range: {u.min().item():.1f} ~ {u.max().item():.1f}")
    pr(f"  v range: {v.min().item():.1f} ~ {v.max().item():.1f}")
    
    in_img = valid & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
    pr(f"  in image bounds: {in_img.sum().item()}/{valid.sum().item()}")
    
    # Test 3: Covariance projection & radius
    pr("\n### TEST 3: Covariance 2D & Radius ###")
    cov3d = gs['cov'].to(device)
    cov2d = renderer.project_covariance_2d_full(cov3d, pts_cam, 500.0, 500.0)
    radius = renderer.get_radius_from_cov2d(cov2d)
    
    pr(f"  cov2d shape: {cov2d.shape}")
    pr(f"  cov2d[0]: {cov2d[0].cpu().numpy().tolist()}")
    pr(f"  cov2d diagonal: xx={cov2d[:,0,0].mean().item():.6f}, yy={cov2d[:,1,1].mean().item():.6f}")
    pr(f"  radius: min={radius.min().item():.4f}, max={radius.max().item():.4f}, mean={radius.mean().item():.4f}")
    pr(f"  radius > 0.5 count: {(radius > 0.5).sum().item()}/{radius.shape[0]}")
    pr(f"  radius > 1.0 count: {(radius > 1.0).sum().item()}/{radius.shape[0]}")
    
    # Test 4: Check valid_tile conditions
    pr("\n### TEST 4: Tile-based render conditions ###")
    tile_size = 16
    tiles_W = (640 + 15) // 16
    tiles_H = (480 + 15) // 16
    tile_x0 = ((u - radius) / tile_size).long().clamp(0, tiles_W - 1)
    tile_x1 = ((u + radius) / tile_size).long().clamp(0, tiles_W - 1)
    tile_y0 = ((v - radius) / tile_size).long().clamp(0, tiles_H - 1)
    tile_y1 = ((v + radius) / tile_size).long().clamp(0, tiles_H - 1)
    n_tiles_per_gs = (tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)
    
    pr(f"  n_tiles_per_gs: min={n_tiles_per_gs.min().item()}, max={n_tiles_per_gs.max().item()}, "
          f"sum={n_tiles_per_gs.sum().item()}")
    pr(f"  n_tiles_per_gs > 0: {(n_tiles_per_gs > 0).sum().item()}/{xyz.shape[0]}")
    pr(f"  valid: {valid.sum().item()}")
    pr(f"  valid & (n_tiles>0): {(valid & (n_tiles_per_gs > 0)).sum().item()}")
    pr(f"  valid & (n_tiles>0) & (radius>0.5): {(valid & (n_tiles_per_gs > 0) & (radius > 0.5)).sum().item()}")
    
    # Test 5: Single Gaussian at camera front
    pr("\n### TEST 5: Single Gaussian at camera front ###")
    xyz_test = torch.tensor([[0.0, 1.0, 5.0]], device=device)
    rgb_test = torch.tensor([[0.9, 0.3, 0.3]], device=device)
    op_test = torch.tensor([[1.0]], device=device)
    cov_test = torch.eye(3, device=device).unsqueeze(0) * 1.0
    
    R_simple = np.eye(3, dtype=np.float32)
    t_simple = np.zeros((3, 1), dtype=np.float32)
    cam_simple = PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam_simple.set_pose(R_simple, t_simple)
    
    gs_test = {'xyz': xyz_test, 'rgb': rgb_test, 'opacity': op_test, 'cov': cov_test}
    
    with torch.no_grad():
        pts_test = (torch.as_tensor(R_simple, device=device) @ xyz_test.T + torch.as_tensor(t_simple, device=device).reshape(3,1)).T
        pr(f"  Test point in camera: {pts_test[0].cpu().numpy()}")
        pr(f"  Expected uv: ({500*0/5 + 320:.1f}, {500*0/5 + 240:.1f}) = (320, 240)")
        
        img, depth = renderer.forward(gs_test, cam_simple)
        pr(f"  Render shape: {img.shape}")
        pr(f"  RGB min/max: {img.min().item():.4f}, {img.max().item():.4f}")
        pr(f"  Coverage: {(depth < float('inf')).sum().item()}/{480*640}")
    
    # Test 6: Force _render_tile_based with single Gaussian
    pr("\n### TEST 6: Force _render_tile_based with single Gaussian ###")
    with torch.no_grad():
        img_tile, depth_tile = renderer._render_tile_based(
            gs_test['xyz'], gs_test['rgb'], gs_test['opacity'], gs_test['cov'], cam_simple
        )
        pr(f"  Tile-based RGB min/max: {img_tile.min().item():.4f}, {img_tile.max().item():.4f}")
        pr(f"  Tile-based coverage: {(depth_tile < float('inf')).sum().item()}/{480*640}")
    
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