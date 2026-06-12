#!/usr/bin/env python3
"""
gs_slam_cuda Deployment Script
================================
Complete deployment entry point for the optimized gs_slam_cuda pipeline.

Target: Linux + RTX 3060 8GB

Prerequisites:
1. conda create -n gs_slam python=3.11.9 && conda activate gs_slam
2. pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
3. pip install numpy pillow tqdm

Usage:
  python deploy.py                   # Full pipeline (CUDA)
  python deploy.py --cpu             # CPU fallback
  python deploy.py --demo            # Demo mode (quick synthetic)
  python deploy.py --train           # Include training
  python deploy.py --benchmark        # Performance benchmark
  python deploy.py --dataset tum_fr1_desk --data-path /path/to/TUM
"""

import sys
import os
import argparse
import warnings

ROOT = os.path.dirname(os.path.abspath(__file__))


def check_dependencies():
    """Check and report available dependencies."""
    deps = {}
    
    # Check PyTorch
    try:
        import torch
        deps['torch'] = torch.__version__
        deps['cuda'] = torch.cuda.is_available()
        if deps['cuda']:
            deps['cuda_version'] = torch.version.cuda
            deps['device'] = torch.cuda.get_device_name(0)
    except ImportError:
        deps['torch'] = None
        deps['cuda'] = False
    
    # Check PIL
    try:
        from PIL import Image
        deps['pil'] = True
    except ImportError:
        deps['pil'] = False
    
    # Check MASt3R
    try:
        import mast3r
        deps['mast3r'] = True
    except ImportError:
        deps['mast3r'] = False
    
    # Check GTSAM
    try:
        import gtsam
        deps['gtsam'] = True
    except ImportError:
        deps['gtsam'] = False
    
    return deps


def print_banner():
    """Print deployment banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  gs_slam_cuda  v3.0.0                        ║
║          SA-AGD SLAM with CUDA Optimization                  ║
║       Target: Linux + RTX 3060 8GB                           ║
║       Framework: PyTorch 2.x + CUDA 12.x                     ║
╚══════════════════════════════════════════════════════════════╝
    """)


def main():
    parser = argparse.ArgumentParser(
        description='gs_slam_cuda: SA-AGD 3DGS SLAM Deployment'
    )
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode')
    parser.add_argument('--cuda', action='store_true',
                        help='Force CUDA mode')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Enable FP16 mixed precision')
    parser.add_argument('--no-fp16', action='store_false', dest='fp16',
                        help='Disable FP16')
    parser.add_argument('--demo', action='store_true',
                        help='Quick demo mode (synthetic only)')
    parser.add_argument('--train', action='store_true',
                        help='Include training pipeline')
    parser.add_argument('--train-iters', type=int, default=200,
                        help='Training iterations')
    parser.add_argument('--benchmark', action='store_true',
                        help='Performance benchmark only')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'tum_fr1_desk', 'tum_fr2_xyz',
                                 'tum_fr3_long_office', 'euroc_mh01',
                                 'euroc_v101', 'replica_room0'],
                        help='Dataset to use')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to dataset root directory')
    parser.add_argument('--max-frames', type=int, default=500,
                        help='Maximum frames to load')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies
    deps = check_dependencies()
    print(f"  PyTorch:  {deps.get('torch', 'NOT FOUND')}")
    print(f"  CUDA:     {deps.get('cuda', False)}")
    if deps.get('cuda'):
        print(f"  Device:   {deps.get('device', 'unknown')}")
        print(f"  CUDA:     {deps.get('cuda_version', 'unknown')}")
    print(f"  PIL:      {deps.get('pil', False)}")
    print(f"  MASt3R:   {deps.get('mast3r', False)}")
    print(f"  GTSAM:    {deps.get('gtsam', False)}")
    
    if not deps.get('torch'):
        print("\n[ERROR] PyTorch is required!")
        print("  Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    # Import and run gs_slam_cuda demo
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'gs_slam_cuda'))
    
    # Override sys.argv for the demo script
    demo_args = ['run_all.py']
    
    if args.cpu:
        demo_args.append('--cuda')  # will fallback to CPU
    elif args.cuda:
        demo_args.append('--cuda')
    
    if args.fp16:
        demo_args.append('--fp16')
    else:
        demo_args.append('--no-fp16')
    
    if args.benchmark:
        demo_args.append('--benchmark')
    else:
        demo_args.append('--dataset')
        demo_args.append(args.dataset)
    
    if args.data_path:
        demo_args.append('--data-path')
        demo_args.append(args.data_path)
    
    if args.max_frames:
        demo_args.append('--max-frames')
        demo_args.append(str(args.max_frames))
    
    if args.train:
        demo_args.append('--train')
        demo_args.append('--train-iters')
        demo_args.append(str(args.train_iters))
    
    if args.output:
        demo_args.append('--output')
        demo_args.append(args.output)
    
    if args.demo:
        demo_args.append('--all')
    
    # Save original argv and set new ones
    orig_argv = sys.argv
    sys.argv = demo_args
    
    try:
        from gs_slam_cuda.demo.run_all import main as demo_main
        demo_main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        sys.argv = orig_argv
    
    print("\n[INFO] Deployment completed successfully!")


if __name__ == '__main__':
    main()