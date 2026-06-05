"""
MASt3R-Fusion + GS-SLAM 部署入口
=================================
基于真实代码仓库的部署脚本。

前置条件 (按顺序执行):
1. conda create -n mast3r_fusion python=3.11.9 && conda activate mast3r_fusion
2. pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
3. cd gtsam && mkdir build && cd build
   cmake .. -DGTSAM_BUILD_PYTHON=1 -DGTSAM_PYTHON_VERSION=3.11.9
   make python-install -j12
4. cd MASt3R-Fusion && pip install -e thirdparty/mast3r && pip install -e thirdparty/in3d && pip install --no-build-isolation -e .
5. 下载MASt3R模型权重到 MASt3R-Fusion/checkpoints/

如果上述前置条件不满足，将使用gs_slam的纯NumPy实现作为fallback。
"""

import sys
import os
import warnings

# 获取项目根目录
ROOT = os.path.dirname(os.path.abspath(__file__))

def try_import_mast3r_fusion():
    """尝试使用真实的MASt3R-Fusion"""
    try:
        sys.path.insert(0, os.path.join(ROOT, 'MASt3R-Fusion'))
        import mast3r_fusion
        print("[INFO] MASt3R-Fusion 导入成功! 使用完整CUDA管线。")
        return True
    except ImportError as e:
        print(f"[WARN] MASt3R-Fusion 导入失败: {e}")
        print("[INFO] 使用 gs_slam 纯NumPy fallback实现")
        return False

def try_import_mast3r():
    """检查MASt3R模型是否可用"""
    try:
        import mast3r
        from mast3r.model import AsymmetricMASt3R
        print("[INFO] MASt3R 模型可用")
        return True
    except ImportError:
        print("[WARN] MASt3R 未安装，将使用模拟pointmap")
        return False

def run_full_pipeline():
    """根据可用依赖选择运行方式"""
    have_mast3r_fusion = try_import_mast3r_fusion()
    have_mast3r = try_import_mast3r()
    
    if have_mast3r_fusion and have_mast3r:
        print("\n" + "="*60)
        print("  运行完整MASt3R-Fusion管线 (CUDA)")
        print("="*60)
        run_mast3r_fusion()
    else:
        print("\n" + "="*60)
        print("  运行 gs_slam 实现 (纯NumPy)")
        print("="*60)
        run_gs_slam()

def run_mast3r_fusion():
    """使用真实MASt3R-Fusion管线"""
    print("[INFO] MASt3R-Fusion 管线待实现")
    print("[INFO] 请参考 MASt3R-Fusion/README.md 中的使用说明")
    print("[INFO] 关键命令:")
    print("  python MASt3R-Fusion/main.py --dataset <path> --config <config> ...")

def run_gs_slam():
    """使用gs_slam实现"""
    sys.path.insert(0, ROOT)
    from gs_slam.demo.run_all import main
    main()

if __name__ == '__main__':
    run_full_pipeline()