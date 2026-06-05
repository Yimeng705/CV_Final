"""
MASt3R-Fusion + gs_slam 环境设置脚本
检查依赖并安装
"""
import subprocess
import sys

def run(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=r'd:\Myhomework\j3down\'\cv\final')
    if r.stdout: print(r.stdout[-500:])
    if r.stderr and 'WARNING' not in r.stderr: print('ERR:', r.stderr[-200:])
    return r

# 1. Check PyTorch
print("=== Checking PyTorch ===")
try:
    import torch
    print(f"Torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("Torch not found, installing...")
    run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")

# 2. Install mast3r
print("\n=== Installing MASt3R ===")
run("pip install -e thirdparty/mast3r", cwd='MASt3R-Fusion')
# Fallback: install directly from local copy
run("pip install -e mast3r")

# 3. Install other deps  
print("\n=== Installing other deps ===")
for pkg in ['opencv-python','h5py','pyparsing','lietorch','scipy','matplotlib','pillow']:
    try:
        __import__(pkg.replace('-','_'))
        print(f"  {pkg}: OK")
    except:
        print(f"  Installing {pkg}...")
        run(f"pip install {pkg}")

print("\n=== Setup complete ===")