"""
Verify the FP16 fix: render with use_fp16=True, check coverage is > 0.
Output to C:\temp\verify_fix_output.txt
"""
import sys, os, torch, numpy as np

OUTPUT = r"C:\temp\verify_fix_output.txt"
lines = []
def pr(s=""): lines.append(str(s)); print(s)

try:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if base not in sys.path:
        sys.path.insert(0, base)
    
    from gs_slam_cuda.core.camera import PinholeCamera, generate_helical_trajectory
    from gs_slam_cuda.core.gaussian_model_cuda import create_test_scene_cuda
    from gs_slam_cuda.core.renderer_cuda import CUDASplatRenderer
    
    device = torch.device('cuda:0')
    gc = create_test_scene_cuda(device=device, n_gaussians=1200)
    poses = generate_helical_trajectory(n_poses=50, radius=10.0, height_range=(-2.0, 4.0))
    
    pr(f"Scene: N={len(gc)}, Poses: {len(poses)}")
    
    # Test FP16=True (the previous problematic configuration)
    renderer = CUDASplatRenderer(image_height=480, image_width=640, use_fp16=True, device=device)
    gs_dict = gc.pack()
    
    all_ok = True
    for i in range(6):
        R, t = poses[i]
        cam = PinholeCamera()
        cam.set_pose(R, t)
        
        with torch.no_grad():
            rgb, depth = renderer.forward(gs_dict, cam)
        
        coverage = (depth < float('inf')).sum().item() / (480 * 640) * 100
        arr = rgb.cpu().numpy()
        has_content = arr.max() - arr.min() > 0.01
        
        status = "OK" if (coverage > 0 and has_content) else "FAIL"
        pr(f"  View {i}: coverage={coverage:.1f}% min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f} [{status}]")
        
        if coverage < 0.01:
            all_ok = False
    
    pr(f"\nOverall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    pr("DONE")

except Exception as e:
    import traceback
    pr(f"ERROR: {e}")
    pr(traceback.format_exc())

finally:
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nOutput saved to {OUTPUT}")