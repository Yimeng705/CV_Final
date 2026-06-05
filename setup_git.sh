#!/bin/bash
# ============================================================
# 初始化Git仓库并上传到GitHub
# 使用前请替换 YOUR_USERNAME 和 YOUR_REPO_NAME
# ============================================================

set -e

GITHUB_USERNAME="YOUR_USERNAME"
REPO_NAME="3DGS-SLAM"

echo "=== Initializing Git repository ==="

# 清理可能的锁文件
rm -f .git/index.lock

# 初始化 (如果尚未)
if [ ! -d .git ]; then
    git init
fi

# 添加所有文件
git add -A

# 首次提交
git commit -m "Initial commit: 3DGS-SLAM system

- gs_slam/: Pure NumPy implementation of 3DGS+SLAM
- MASt3R-Fusion/: Third-party code from GREAT-WHU
- mast3r/: Third-party MASt3R model from NAVER
- gtsam/: Third-party GTSAM factor graph library
- deploy.py: Unified deployment entry point
- setup_env.py: Environment setup script

Based on:
- MASt3R-SLAM (Murai et al., 2024)
- MASt3R-Fusion (Zhou et al., 2025)
- OpenMonoGS-SLAM (Yoo et al., 2025)
- A Survey on 3D Gaussian Splatting (Chen & Wang, 2024)"

echo ""
echo "=== Pushing to GitHub ==="
echo "If you haven't created the repo yet, go to:"
echo "  https://github.com/new"
echo "  Name: $REPO_NAME"
echo "  Do NOT add README, .gitignore, or LICENSE (we already have them)"
echo ""

# 添加远程仓库
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git" 2>/dev/null || \
    git remote set-url origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# 推送
git branch -M main
git push -u origin main

echo ""
echo "=== Done! ==="
echo "Your repository is at: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"