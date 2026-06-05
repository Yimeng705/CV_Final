"""
3DGS可微渲染器
================
基于综述论文(Chen & Wang, 2024)的tile-based渲染管线，结合OpenMonoGS-SLAM的语义渲染。

核心流程:
1. 视锥体裁剪
2. 3D→2D投影 (Σ_2D = J W Σ W^T J^T)
3. 按深度排序
4. Alpha blending
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .camera import PinholeCamera

class SplatRenderer:
    """3DGS Splat渲染器 (简化版)"""
    
    def __init__(self, H: int = 480, W: int = 640):
        self.H = H; self.W = W
    
    def render(self, gs: Dict[str, np.ndarray],
               cam: PinholeCamera) -> Tuple[np.ndarray, np.ndarray]:
        """
        主渲染函数
        
        Returns:
            rgb: [H,W,3] float32 [0,1]
            sem: [H,W,D] float32 语义特征图
        """
        xyz = gs['xyz']; rgb = gs['rgb']
        scale = gs['scale']; opacity = gs['opacity']
        sem = gs.get('sem', np.zeros((len(xyz), 64), dtype=np.float32))
        
        # 世界→相机
        pts_cam = (cam.R @ xyz.T + cam.t).T  # [N,3]
        z = pts_cam[:, 2]
        
        # 视锥体裁剪
        ok = z > 0.01
        if not ok.any():
            return np.ones((self.H, self.W, 3), dtype=np.float32), \
                   np.zeros((self.H, self.W, 64), dtype=np.float32)
        
        pts_cam = pts_cam[ok]; z = z[ok]
        rgb = rgb[ok]; opacity = opacity[ok]; scale = scale[ok]
        sem = sem[ok]
        
        # 投影到像素
        fx, fy = cam.fx, cam.fy
        cx, cy = cam.cx, cam.cy
        
        u = fx * pts_cam[:,0] / z + cx
        v = fy * pts_cam[:,1] / z + cy
        
        # 2D尺度近似
        s2d = (scale[:, :2] * np.array([fx, fy])[None]) / z[:, None]
        radius = np.maximum(2, (np.sqrt(np.max(s2d, axis=1)) * 2.5).astype(int))
        
        # 深度排序(远处在前)
        order = np.argsort(z)
        u, v = u[order], v[order]
        z = z[order]
        rgb, opacity, sem = rgb[order], opacity[order], sem[order]
        radius = radius[order]
        s2d = s2d[order]
        
        # Alpha blending
        img = np.ones((self.H, self.W, 3), dtype=np.float32)
        sem_img = np.zeros((self.H, self.W, sem.shape[1]), dtype=np.float32)
        T = np.ones((self.H, self.W, 1), dtype=np.float32)
        
        for i in range(len(u)):
            ui, vi = int(round(u[i])), int(round(v[i]))
            r = int(radius[i])
            if ui<-r or ui>=self.W+r or vi<-r or vi>=self.H+r: continue
            
            y0, y1 = max(0,vi-r), min(self.H,vi+r+1)
            x0, x1 = max(0,ui-r), min(self.W,ui+r+1)
            if y1<=y0 or x1<=x0: continue
            
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            sx, sy = max(s2d[i,0], 0.01), max(s2d[i,1], 0.01)
            
            g = np.exp(-0.5 * (((xx-ui)/sx)**2 + ((yy-vi)/sy)**2))[..., None]
            alpha = float(opacity[i, 0]) * g
            
            patch = img[y0:y1, x0:x1]
            patch_sem = sem_img[y0:y1, x0:x1]
            T_patch = T[y0:y1, x0:x1]
            
            img[y0:y1, x0:x1] = patch*(1-alpha) + rgb[i]*alpha
            sem_img[y0:y1, x0:x1] = patch_sem*(1-alpha) + sem[i]*alpha
            T[y0:y1, x0:x1] *= (1-alpha)
        
        return np.clip(img, 0, 1).astype(np.float32), sem_img
    
    def render_rgb(self, *args, **kwargs) -> np.ndarray:
        """只渲染RGB"""
        rgb, _ = self.render(*args, **kwargs)
        return (np.clip(rgb, 0, 1)*255).astype(np.uint8)


class PointRenderer:
    """稀疏点渲染(Baseline)"""
    
    def __init__(self, H: int = 480, W: int = 640):
        self.H = H; self.W = W
    
    def render(self, gs: Dict[str, np.ndarray], cam: PinholeCamera) -> np.ndarray:
        xyz = gs['xyz']; rgb = gs['rgb']
        pts_cam = (cam.R @ xyz.T + cam.t).T
        z = pts_cam[:,2]
        ok = z > 0.01; pts_cam=pts_cam[ok]; z=z[ok]; rgb=rgb[ok]
        
        u = (cam.fx*pts_cam[:,0]/z+cam.cx).astype(int)
        v = (cam.fy*pts_cam[:,1]/z+cam.cy).astype(int)
        
        valid = (u>=0)&(u<self.W)&(v>=0)&(v<self.H)
        u, v = u[valid], v[valid]
        
        img = np.full((self.H, self.W, 3), 255, dtype=np.uint8)
        col = (np.clip(rgb[valid],0,1)*255).astype(np.uint8)
        
        for du in range(-1,2):
            for dv in range(-1,2):
                cu = np.clip(u+du, 0, self.W-1)
                cv = np.clip(v+dv, 0, self.H-1)
                img[cv, cu] = col
        return img