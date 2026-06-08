"""Quick test of all imports"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Testing imports...")

from gs_slam.core.camera import PinholeCamera, look_at
print("  [OK] camera")

from gs_slam.core.gaussian_model import GaussianCloud, make_test_scene
print("  [OK] gaussian_model")

from gs_slam.core.renderer import SplatRenderer, PointRenderer
print("  [OK] renderer")

from gs_slam.core.factor_graph import PoseGraph
print("  [OK] factor_graph")

from gs_slam.core.adaptive_density import AdaptiveDensityController, run_adaptive_densification_cycle
print("  [OK] adaptive_density")

from gs_slam.slam.frontend import generate_synthetic_pointmaps
print("  [OK] frontend")

from gs_slam.slam.backend import SLAMBackend
print("  [OK] backend")

from gs_slam.slam.mapper import DenseMapper
print("  [OK] mapper")

# Quick functional test
print("\nFunctional test...")
gc = make_test_scene(100)
renderer = SplatRenderer()
cam = PinholeCamera()
R, t = look_at([5,1,5], [0,0,0], [0,1,0])
cam.set_pose(R, t)
rgb, sem, depth = renderer.render(gc.pack(), cam)
print(f"  [OK] Render: {rgb.shape}, depth range: {depth.min():.2f}-{depth.max():.2f}")

# Test adaptive density
ctrl = AdaptiveDensityController()
gs_data = gc.pack()
gs_data = run_adaptive_densification_cycle(gs_data, ctrl, n_iterations=2)
stats = ctrl.get_stats()
print(f"  [OK] Adaptive density: {stats}")

print("\nAll tests passed!")