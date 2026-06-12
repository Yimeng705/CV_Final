"""
Test FP16 vs FP32 rendering to identify if precision is the issue.
Output to C:\temp\debug_fp16_output.txt
"""
import sys, os
import torch
import numpy as np

OUTPUT = r"C:\temp\debug_fp16_output.txt"
lines = []

def pr(s=""):
    text = str(s)
    print(text)
    lines.append(text)

# Build path without quotes
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if base not in sys.path:
    sys.path.insert(0, base)

try:
    from gs_slam_cuda.core.camera import PinholeCamera, generate_helical_trajectory
    from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda
    from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer

    device = torch.device('cuda:0')

    gc = create_test_scene_cuda(device=device, n_gaussians=1200)
    poses = generate_helical_trajectory(n_poses=50, radius=10.0, height_range=(-2.0, 4.0))

    pr(f"Scene: N={len(gc)}")
    pr(f"Poses: {len(poses)}")

    # Test with FP16=True (how run_all.py calls it) vs FP16=False
    for use_fp16 in [True, False]:
        pr(f"\n{'='*60}")
        pr(f"Testing with use_fp16={use_fp16}")
        pr(f"{'='*60}")

        renderer = CUDASplatRenderer(image_height=480, image_width=640, use_fp16=use_fp16, device=device)
        gs_dict = gc.pack()

        n_views = min(6, len(poses))
        for i in range(n_views):
            R, t = poses[i]
            cam = PinholeCamera()
            cam.set_pose(R, t)

            with torch.no_grad():
                rgb, depth = renderer.forward(gs_dict, cam)

            coverage = (depth < float('inf')).sum().item() / (480 * 640) * 100
            rgb_np = rgb.cpu().numpy()
            pr(f"  View {i}: min={rgb_np.min():.4f} max={rgb_np.max():.4f} mean={rgb_np.mean():.4f} coverage={coverage:.2f}%")

            if coverage < 1.0:
                # Diagnose why
                xyz_g = gs_dict['xyz'].to(device)
                R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
                t_t = torch.as_tensor(t, device=device, dtype=torch.float32).reshape(3, 1)
                pts_cam = (R_t @ xyz_g.T + t_t).T
                depth_cam = pts_cam[:, 2]
                valid = depth_cam > 0.01

                # Check internal radius values
                cov3d_g = gs_dict['cov'].to(device)
                cov2d = renderer.project_covariance_2d_full(cov3d_g, pts_cam, cam.fx, cam.fy)
                radius = renderer.get_radius_from_cov2d(cov2d)
                u_cam = cam.fx * pts_cam[:, 0] / depth_cam.clamp(min=0.01) + cam.cx
                v_cam = cam.fy * pts_cam[:, 1] / depth_cam.clamp(min=0.01) + cam.cy

                # Tile filter
                tile_size = 16
                tiles_W = (640 + 15) // 16
                tiles_H = (480 + 15) // 16
                tx0 = ((u_cam - radius) / tile_size).long().clamp(0, tiles_W - 1)
                tx1 = ((u_cam + radius) / tile_size).long().clamp(0, tiles_W - 1)
                ty0 = ((v_cam - radius) / tile_size).long().clamp(0, tiles_H - 1)
                ty1 = ((v_cam + radius) / tile_size).long().clamp(0, tiles_H - 1)
                n_tiles_per_gs = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
                vt = valid & (n_tiles_per_gs > 0) & (radius > 0.5)
                tts = int(n_tiles_per_gs.sum().item())

                pr(f"    valid={valid.sum().item()}, radius min={radius.min().item():.2f} max={radius.max().item():.2f}")
                pr(f"    radius NaN count: {radius.isnan().sum().item()}")
                pr(f"    valid_tile count={vt.sum().item()}, total_tile_slots={tts}")
                pr(f"    cov2d[0]: {cov2d[0].cpu().numpy().tolist()}")

                # Check if this is tile-based or splatted
                # Directly call tile-based to check
                img_tile, d_tile = renderer._render_tile_based(
                    gs_dict['xyz'].to(device),
                    gs_dict['rgb'].to(device),
                    gs_dict['opacity'].to(device),
                    gs_dict['cov'].to(device),
                    cam
                )
                c_tile = (d_tile < float('inf')).sum().item() / (480 * 640) * 100
                pr(f"    Direct tile-based coverage: {c_tile:.2f}%")

                if c_tile < 1.0:
                    pr(f"    *** TILE-BASED PATH ALSO BLANK! Investigating...")
                    # Check total_tile_slots specifically
                    pr(f"    total_tile_slots={tts} (if 0, falls through to white image)")

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