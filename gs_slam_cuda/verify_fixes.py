"""Verify all fixes without running full pipeline"""
import sys, os
sys.path.insert(0, r"d:\Myhomework\j3down'\cv\final")

log_lines = []

def log(s):
    log_lines.append(s)
    print(s)

try:
    import torch
    import numpy as np

    log(f"PyTorch {torch.__version__} CUDA {torch.version.cuda}")
    device = torch.device('cuda:0')

    # Test A: Renderer - single Gaussian (no FP16 autocast)
    log("\n--- Test A: Single Gaussian FP16 (verify no NaN) ---")
    from gs_slam_cuda.core.camera import PinholeCamera
    from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer
    xyz = torch.tensor([[0., 1., 5.]], device=device)
    rgb = torch.tensor([[0.9, 0.3, 0.3]], device=device)
    op = torch.tensor([[1.0]], device=device)
    cov = torch.eye(3, device=device).unsqueeze(0) * 1.0
    gs = {'xyz': xyz, 'rgb': rgb, 'opacity': op, 'cov': cov}
    cam = PinholeCamera()
    cam.set_pose(np.eye(3), np.zeros((3, 1)))
    r = CUDASplatRenderer(use_fp16=True, device=device)
    img, depth = r.forward(gs, cam)
    arr = img.cpu().numpy()
    cov_pct = (depth < 9e9).sum().item() / (480 * 640) * 100
    has_nan = np.isnan(arr).any()
    log(f"  coverage={cov_pct:.1f}% NaN={has_nan}")
    assert cov_pct > 0, "FAIL A1: blank image"
    assert not has_nan, "FAIL A2: NaN output"
    log("  PASS")

    # Test B: compute_rpe with empty GT list
    log("\n--- Test B: compute_rpe empty GT guard ---")
    from gs_slam_cuda.core.factor_graph_cuda import compute_rpe
    from gs_slam_cuda.core.camera import CameraPose

    # Create one dummy pose
    p0 = CameraPose(R=np.eye(3), t=np.zeros((3, 1)))

    # Empty GT list
    rpe_t, rpe_r = compute_rpe([p0], [])
    log(f"  empty GT: rpe_t={rpe_t}, rpe_r={rpe_r}")
    assert rpe_t == 0.0 and rpe_r == 0.0, "FAIL B1: should return 0 for empty GT"

    # Single GT (only 1 element)
    rpe_t2, rpe_r2 = compute_rpe([p0], [p0])
    log(f"  single GT: rpe_t={rpe_t2}, rpe_r={rpe_r2}")
    assert rpe_t2 == 0.0 and rpe_r2 == 0.0, "FAIL B2: should return 0 for single GT"

    # Valid GT (2 elements)
    p1 = CameraPose(R=np.eye(3), t=np.ones((3, 1)))
    rpe_t3, rpe_r3 = compute_rpe([p0, p1], [p0, p1])
    log(f"  valid GT: rpe_t={rpe_t3:.4f}, rpe_r={rpe_r3:.4f}")
    log("  PASS")

    # Test C: evaluate_trajectory with empty GT
    log("\n--- Test C: backend.evaluate_trajectory empty GT ---")
    from gs_slam_cuda.slam.backend_cuda import CUDABackend
    backend = CUDABackend(device=device)
    # Add 2 keyframes
    backend.add_keyframe(CameraPose(R=np.eye(3), t=np.zeros((3, 1))),
                         CameraPose(R=np.eye(3), t=np.zeros((3, 1))), 1.0)
    backend.add_keyframe(p1, p0, 0.9)
    metrics = backend.evaluate_trajectory([])
    log(f"  metrics={metrics}")
    assert 'ate_rmse' in metrics, "FAIL C1: missing ate_rmse"
    log("  PASS")

    # Test D: Multi-Gaussian FP16 rendering with autocast disabled for non-diff
    log("\n--- Test D: create_test_scene 1200 Gaussians FP16 ---")
    from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda
    from gs_slam_cuda.core.camera import look_at
    gc = create_test_scene_cuda(device=device, n_gaussians=1200)
    r4 = CUDASplatRenderer(use_fp16=True, device=device)
    gs4 = gc.pack()
    R, t = look_at(np.array([8., 5., 10.]), np.array([0., 1., 0.]))
    cam4 = PinholeCamera()
    cam4.set_pose(R, t)
    img4, depth4 = r4.forward(gs4, cam4)
    arr4 = img4.cpu().numpy()
    cov4 = (depth4 < 9e9).sum().item() / (480 * 640) * 100
    nan4 = np.isnan(arr4).any()
    log(f"  coverage={cov4:.1f}% NaN={nan4}")
    assert cov4 > 50, "FAIL D1: coverage < 50%"
    assert not nan4, "FAIL D2: NaN output"
    log("  PASS")

    log("\n" + "=" * 50)
    log("ALL FIXES VERIFIED SUCCESSFULLY")
    log("=" * 50)

except Exception as e:
    log(f"\nERROR: {e}")
    import traceback
    log(traceback.format_exc())

# Write output file
out_file = r"d:\Myhomework\j3down'\cv\final\gs_slam_cuda\verify_out.txt"
with open(out_file, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"\nLog written to {out_file}")