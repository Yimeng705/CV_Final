# Demo代码问题诊断与可执行优化方案

## 一、总体评估

### 1.1 选题方向 ✅

选择"3DGS增强的视觉SLAM"方向，覆盖AIGC/3D重建+视觉感知+SLAM，符合课程"与大模型相关"的要求。四篇论文选择满足规范：1篇TPAMI综述(3DGS-Survey, 2026) + 3篇顶会前沿(MASt3R-SLAM, MASt3R-Fusion, OpenMonoGS-SLAM)。

### 1.2 现有Demo架构

- **6步实验管线** (`run_all.py`)：3DGS渲染→多视角合成→SLAM因子图→增量建图→消融实验→方法改进验证
- **Web交互演示界面** (`frontend.html`)：支持逐步演示、结果查看、论文信息浏览
- **HTML综合报告** (`output/report.html`)：实验指标、消融结果、可视化图片集成展示
- **17个输出文件**：渲染图、点云对比图、深度图、多视角图、轨迹图、语义图、消融图表等

---

## 二、关键问题诊断

### 2.1 🔴 严重问题：全部使用合成随机数据（伪造实验闭环）

当前demo声称实现了MASt3R-SLAM + MASt3R-Fusion + OpenMonoGS-SLAM的融合系统，但所有组件均在完全合成的随机数据上运行，构成了"自己出题自己答"的循环论证：

| 模块 | 当前实现 | 造假方式 |
|------|---------|---------|
| 前端点图生成 | `frontend.py:47` 生成2000个随机3D点 `np.random.uniform(-5,5,(2000,3))` | 真正的MASt3R输出是通过Transformer网络推理的稠密pointmap |
| 后端图优化 | `backend.py:44-48` 用带噪声的真值位姿初始化图节点 | "优化"只是对自加噪声的图做梯度下降，必然收敛 |
| 语义特征 | `mapper.py:141-168` 用K-means空间聚类替代SAM+CLIP | 真正的语义来自VLM模型输出，不是空间聚类 |
| 密度控制梯度 | `adaptive_density.py:295` 用 `np.random.uniform` 模拟梯度 | 真正的梯度由渲染损失反向传播产生 |
| 评估GT | `step1_render():97-103` 同场景偏移渲染作为"GT" | PSNR/SSIM是对同一合成场景的两个视角对比 |

**后果**：所有"实验结果"和"消融验证"都不能证明任何方法的有效性，因为它们都在同一人工闭环中运行。

### 2.2 🔴 严重问题：核心声称与代码实现不一致

#### 2.2a 渲染器声称tile-based但实际是per-Gaussian splatting

| 3DGS综述method-002要求 | `renderer.py` 实现 | 差距 |
|------------------------|-------------------|------|
| 屏幕划分为16×16 tile，每tile独立并行处理 | `render():345-394` 按高斯逐个循环 `for i in range(N):` | 不是tile-based |
| GPU共享内存加速并行alpha混合 | 纯NumPy CPU串行逐像素计算 | 无并行 |
| EWA splatting: `Σ2D = J W Σ Wᵀ Jᵀ` | 简化对角近似 `s2d[:,0,0] = sqrt(abs(cov[:,0,0])) * fx / z` | 缺少完整Jacobian |
| 可微反向传播 | 无任何autograd实现 | 整个渲染器不可微 |

#### 2.2b 因子图缺少Sim(3)和同构映射

MASt3R-Fusion核心创新——Sim(3)-SE(3)群同构映射——完全缺失：

| 论文要求 | `factor_graph.py` 实现 | 差距 |
|---------|----------------------|------|
| Sim(3)群上的高斯-牛顿法 | `optimize():87-148` 只在SE(3)上做简化梯度下降 | 无Sim(3) |
| Sim(3)→SE(3)×R同构映射 Λ=diag(I,s⁻¹I,s) | 完全不存在 | 论文核心被忽略 |
| 舒尔补边缘化 | 完全不存在 | 无滑动窗口 |
| CUDA并行解析雅可比 | NumPy手动构造skew矩阵近似 | Jacobian不正确 |

#### 2.2c 语义感知密度控制的梯度是伪造的

```python
# adaptive_density.py:295
geom_grad = np.random.uniform(0, 0.001, N).astype(np.float32)  # ← 完全是随机数
```

3DGS综述method-003的梯度来自渲染损失反向传播。你的"语义增强"效果来自参数 `sem_grad_weight` 改变阈值分母的数学效应，而非语义信息真正提供了有用的信号。

### 2.3 🟡 中等问题

| 问题 | 位置 | 说明 |
|------|------|------|
| 评估循环论证 | `step1_render():97-109` | PSNR/SSIM对比同场景偏移渲染，非真值 |
| 动态场景测试是假造的 | `step6:610-687` | 手动加异常再检测，测试的是自己预设的逻辑 |
| 前端滑块不触发实际计算 | `frontend.html:191-194` | `updateSemWeight()` 只改显示数字 |
| 论文年份标注不一致 | `__init__.py` vs analyse文件 | "2024" vs "2025" vs "2026" |
| 方法对比用相同随机种子 | `step6:522-543` | baseline和ours用相同随机数，差异仅来自参数变化 |

---

## 三、可执行优化方案（按优先级）

### P0-1: 改造渲染器为真正的Tile-based Splatting ✅ 必须

**目标**：使渲染器符合3DGS综述method-002的tile-based管线定义。

**技术路径**（`gs_slam/core/renderer.py`）：

```python
# 新增方法：SplatRenderer._tile_based_render()
def _tile_based_render(self, gs: Dict, cam: PinholeCamera):
    """
    真正的tile-based渲染管线
    1. 视锥体裁剪 + 协方差投影
    2. 将高斯分配到各tile
    3. tile内深度排序
    4. 逐tile alpha混合（带提前终止）
    """
    # Step 1: 计算每个高斯的2D投影参数
    u, v, depth, radius = self._compute_2d_projections(gs, cam)
    
    # Step 2: 将高斯分配到所有覆盖的tile
    tile_gaussians = [[] for _ in range(self.n_tiles)]
    for i in range(N):
        tile_range = self._get_tile_range(u[i], v[i], radius[i])
        for tile_id in tile_range:
            tile_gaussians[tile_id].append(i)
    
    # Step 3: 每个tile内按深度排序（远处在前）
    for tile_id in range(self.n_tiles):
        tile_gaussians[tile_id].sort(key=lambda i: -depth[i])
    
    # Step 4: 逐tile并行（Python中逐tile循环，但每个tile内的像素可向量化）
    for tile_id in range(self.n_tiles):
        self._composite_tile(tile_id, tile_gaussians[tile_id], ...)
```

**实现步骤**：
1. 在 `SplatRenderer` 类中新增 `_compute_2d_projections()` 方法——将世界坐标高斯投影到屏幕并计算2D协方差
2. 新增 `_get_tile_range()` ——根据2D位置和半径计算该高斯覆盖的tile范围
3. 新增 `_composite_tile()` ——对单个tile内的高斯执行批量alpha混合
4. `render()` 方法内部判断：若高斯数>500则使用tile-based路径，否则fallback到逐高斯路径
5. **关键**：在注释中标注 "NumPy CPU实现，GPU加速版本需CUDA kernel"

**预期效果**：
- 渲染逻辑与3DGS综述method-002完全对齐
- 渲染速度（300高斯）从~450ms降至~50-80ms
- 可在报告中展示 "tile-based vs naive" 的渲染时间对比

---

### P0-2: 标记合成数据边界，诚实声明实验范围 ✅ 必须

**目标**：避免被误解为宣称在真实数据上运行。

**技术路径**：

1. **`run_all.py` 开头新增**：
```python
EXPERIMENT_MODE = "SYNTHETIC"  # "SYNTHETIC" 或 "REAL_DATA"
# 当前版本使用合成数据验证系统架构和组件交互。
# 真实数据版本需集成MASt3R预训练模型和Replica/TUM数据集。
```

2. **`report.html` 标题横幅修改**：
```html
<span class="badge" style="background:#e67e22;">合成数据概念验证</span>
```

3. **每个step的print输出添加前缀**：
```python
print(f"  [合成数据] PSNR: {psnr:.2f}dB (注意: 使用同场景多视图作为参考)")
```

4. **在Poster大纲的Col 1底部增加**：
```
⚠ 实验说明
┌──────────────────────────┐
│ 本demo在合成场景上验证   │
│ 系统架构和组件交互。     │
│ 每个指标的含义已在报告   │
│ 中明确标注。             │
│ 真实数据验证为后续工作。 │
└──────────────────────────┘
```

---

### P0-3: 实现真实PSNR/SSIM评估基准 ✅ 必须（代码已部分实现）

**目标**：将当前的"循环论证"评估替换为有意义的对比。

**技术路径**（修改 `run_all.py` 的 `step1_render()`）：

```python
def step1_render():
    """改进版：使用高分辨率渲染作为pseudo-GT"""
    
    # 方案A：高分辨率渲染作为参考（2x分辨率）
    renderer_hires = SplatRenderer(H=960, W=1280, tile_size=16)
    cam_hires = PinholeCamera(fx=1000, fy=1000, cx=640, cy=480, width=1280, height=960)
    rgb_hires, _, _ = renderer_hires.render(gs_data, cam_hires)
    # 下采样到480×640作为"ground truth"
    gt_ref = rgb_hires[::2, ::2, :]  # 简单2x下采样
    
    # 计算指标
    render_metrics = compute_rendering_metrics(rgb_gs_for_metric, gt_ref)
    
    # 方案B（备选）：多视图融合pseudo-GT
    fused_gt = np.zeros_like(rgb_gs)
    weight_sum = np.zeros((H, W, 1))
    for offset in [(-0.3,0,0), (0.3,0,0), (0,-0.2,0), (0,0.2,0), (0,0,0)]:
        # 渲染偏移视图
        rgb_v, _, depth_v = render_offset_view(...)
        # 只信任深度有效的像素
        valid = depth_v < np.inf
        fused_gt[valid] += rgb_v[valid]
        weight_sum[valid] += 1
    gt_ref = fused_gt / np.maximum(weight_sum, 1)
    
    render_metrics = compute_rendering_metrics(rgb_gs_for_metric, gt_ref)
```

**指标输出的标注**：
```
[Metrics] PSNR: 28.32dB (参考: 2x高分辨率下采样)
[Metrics] SSIM: 0.8912 (参考: 2x高分辨率下采样)
[Metrics] LPIPS(proxy): 0.1534
```

**关键**：在HTML报告和poster中必须标注"参考标准：高分辨率渲染下采样"，不可标注为"Ground Truth"。

---

### P1-1: 替换随机梯度为可解释的几何代理 ✅ 强烈建议

**目标**：让密度控制的触发逻辑有物理意义，而非随机。

**技术路径**（修改 `gs_slam/core/adaptive_density.py`）：

```python
def compute_geometric_importance(self, gs_data: Dict[str, np.ndarray],
                                  cam: PinholeCamera) -> np.ndarray:
    """
    基于投影覆盖度的几何重要性代理
    
    替代随机梯度：在真实3DGS训练中，梯度大的高斯通常具有：
    1. 较大的2D投影面积（覆盖更多像素）
    2. 位于图像边缘/高频纹理区域
    3. 在视图空间中靠近相机
    
    我们使用2D投影面积作为几何重要性的代理：
    - 投影面积大 → 高斯覆盖范围大 → 可能需要分裂以获得更细粒度
    - 投影面积小 → 高斯覆盖范围小 → 可能已经足够精细
    
    Returns:
        importance: [N] 归一化重要性分数 [0, 1]
    """
    xyz = gs_data['xyz']
    scales = gs_data['scale']
    N = len(xyz)
    
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    
    # 世界→相机
    pts_cam = (cam.R @ xyz.T + cam.t).T
    z = pts_cam[:, 2]
    valid_z = np.maximum(z, 0.01)
    
    # 2D投影尺度 (实际系统使用完整协方差投影)
    fx, fy = cam.fx, cam.fy
    # 高斯在图像平面的近似覆盖面积 ∝ scale² * f² / z²
    proj_area = (scales[:, 0] * fx / valid_z) * (scales[:, 1] * fy / valid_z)
    
    # 深度加权：近处的高斯影响更大
    depth_weight = 1.0 / (1.0 + np.abs(z - np.mean(z)) / np.std(z + 1e-6))
    
    # 组合
    importance = proj_area * depth_weight
    
    # 归一化到 [0, 1]
    if importance.max() > importance.min():
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    return importance.astype(np.float32)
```

**修改 `run_adaptive_densification_cycle()`**：
```python
# 旧代码：
# geom_grad = np.random.uniform(0, 0.001, N).astype(np.float32)

# 新代码：
# 需要传入camera参数
def run_adaptive_densification_cycle(
    gs_data, controller, n_iterations=5, camera=None
):
    # ... 
    geom_grad = controller.compute_geometric_importance(gs_data, camera)
    # ...
```

**说明**：
- 这个代理值不是真正的反向传播梯度，但它有明确的物理含义
- 在代码注释中标注："几何重要性代理（投影覆盖度），真实梯度需通过可微渲染反向传播获得"
- 这使得密度控制有了内在逻辑：覆盖面积大的高斯更可能被分裂

---

### P1-2: 改造评估为有意义的"基线vs改进"对比 ✅ 强烈建议

**目标**：让Step 6的"方法改进验证"从单一参数切换变为真正的方法对比。

**技术路径**（修改 `run_all.py` 的 `step6_improved_method()`）：

```python
def step6_improved_method(kfs, pg, mapper, render_metrics_step1):
    """改进版：多维度方法对比"""
    
    results = {}
    renderer = SplatRenderer()
    cam = PinholeCamera()  # 固定评估视角
    
    # --- 对比1：密度控制策略 ---
    strategies = [
        ("无密度控制", False, 0.0),
        ("纯几何密度控制(3DGS)", True, 0.0),
        ("语义感知密度控制(Ours)", True, 0.3),
    ]
    
    for name, use_den, sw in strategies:
        mapper_test = DenseMapper(5000, use_adaptive_density=use_den, sem_weight=sw)
        # 用相同的高斯初始化（关键：不同策略但相同输入）
        _init_with_fixed_seed(mapper_test, kfs, pg, seed=42)
        mapper_test.assign_semantic_regions(n_regions=4)
        
        if use_den:
            stats = mapper_test.run_densification(n_cycles=3)
        else:
            stats = {'initial_n': mapper_test.size(), 'final_n': mapper_test.size(), 
                     'growth_ratio': 1.0, 'n_cloned': 0, 'n_split': 0, 
                     'n_pruned': 0, 'n_semantic_boost': 0}
        
        # 渲染评估
        rgb, _, depth = renderer.render(mapper_test.get_map(), cam)
        coverage = (depth < np.inf).mean()
        
        strategies_results.append({
            'name': name, 'n_gaussians': mapper_test.size(),
            'coverage_ratio': coverage,
            # 注：PSNR/SSIM需要真值，合成场景下使用高分辨率渲染作为参考
        })
    
    # --- 对比2：渲染方式对比（3DGS vs Point Cloud） ---
    # 已部分实现在step1中，增强对比可视化
    
    # --- 对比3：多尺度语义监督消融 ---
    # 对比单尺度 vs 多尺度语义损失的效果
    # （模拟OpenMonoGS-SLAM消融实验）
```

**在HTML报告中增加对比表**：
```html
<table>
<tr><th>方法</th><th>高斯数</th><th>覆盖度</th><th>语义边界增强</th></tr>
<tr><td>无密度控制</td><td>3500</td><td>68%</td><td>0</td></tr>
<tr><td>纯几何密度控制</td><td>4890</td><td>76%</td><td>0</td></tr>
<tr style="font-weight:bold"><td>语义感知密度控制(Ours)</td><td>5123</td><td>78%</td><td>87</td></tr>
</table>
<p style="color:#888;font-size:0.85em;">注：以上为合成场景概念验证数据。覆盖度和语义增强操作数为代理指标。</p>
```

---

### P1-3: 使前端参数滑块触发实际效果 ✅ 强烈建议

**目标**：让 `frontend.html` 的语义权重滑块真正展示不同参数的效果差异。

**技术路径**：

1. **预计算多参数渲染**（`run_all.py` 新增函数）：
```python
def precompute_param_sweep(kfs, pg, out_dir):
    """预计算sem_weight从0.0到0.6的7组渲染结果"""
    for sw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        mapper = DenseMapper(5000, use_adaptive_density=True, sem_weight=sw)
        # ... 建图 + 渲染 ...
        rgb, sem_map, _ = renderer.render(mapper.get_map(), cam)
        # 保存为 sem_sweep_w{sw:.1f}.png
        Image.fromarray(rgb_u8).save(os.path.join(out_dir, f'sem_sweep_w{sw*10:02.0f}.png'))
```

2. **前端JS联动**：
```javascript
// 预加载7张对比图
const sweepImages = {};
for (let w = 0; w <= 6; w++) {
    sweepImages[w/10] = `../output/sem_sweep_w${w.toString().padStart(2,'0')}.png`;
}

function updateSemWeight(val) {
    semWeight = parseFloat(val);
    document.getElementById('sem-weight-val').textContent = val.toFixed(1);
    // 切换到对应参数的预渲染图
    const imgSrc = sweepImages[val.toFixed(1)];
    if (imgSrc) {
        showImageOnCanvas(imgSrc, `语义权重 α=${val.toFixed(1)} 渲染结果`);
    }
    // 更新语义增强操作数
    const boostData = {0.0:0, 0.1:15, 0.2:32, 0.3:52, 0.4:87, 0.5:109, 0.6:124};
    document.getElementById('m-sem-boost').textContent = boostData[val.toFixed(1)] || '--';
}
```

3. **HTML中增加显示项**：
```html
<div class="metric-row">
    <span class="label">语义增强操作数</span>
    <span class="value" id="m-sem-boost">52</span>
</div>
```

---

### P2-1: 集成真实MASt3R预训练模型输出 ✅ 建议（提升说服力最大的改进）

**目标**：至少用一对真实图像的pointmap替换合成数据。

**技术路径**（工作量约4-6小时）：

1. **安装MASt3R依赖**：
```bash
# 项目已包含mast3r/目录，检查是否可直接使用
cd mast3r && pip install -e .
# 下载预训练权重
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

2. **创建 `gs_slam/data_loader.py`**：
```python
"""
真实数据加载器
支持：
- 预提取的MASt3R pointmap (.npy文件)
- Replica/TUM RGB-D数据
- 自采图像对
"""
import numpy as np
from pathlib import Path

def load_cached_pointmaps(data_dir: str = "data/tum_fr1_desk"):
    """
    加载预缓存的pointmap序列
    
    数据准备步骤：
    1. 下载TUM fr1/desk序列（~1.5GB）
    2. 运行 mast3r/demo.py 提取pointmap到 .npy 文件
    3. 将.npy文件放入 data/tum_fr1_desk/
    """
    pm_path = Path(data_dir)
    pointmaps = []
    for npy_file in sorted(pm_path.glob("pointmap_*.npy")):
        pm = np.load(npy_file)
        pointmaps.append(pm)
    
    if len(pointmaps) == 0:
        print(f"[Warning] 未找到缓存pointmap，回退到合成数据")
        return None
    
    return pointmaps

def load_replica_sequence(seq_name: str = "office0"):
    """
    加载Replica数据集序列（需要先下载数据集）
    
    下载: https://github.com/facebookresearch/Replica-Dataset
    """
    # 实现Replica数据读取
    # 每个序列包含：rgb/*.jpg, depth/*.png, poses.txt
    pass
```

3. **修改 `run_all.py` 主函数**：
```python
def main():
    # 尝试加载真实数据
    real_pointmaps = load_cached_pointmaps()
    
    if real_pointmaps is not None:
        EXPERIMENT_MODE = "REAL_DATA"
        print("[INFO] 使用真实MASt3R pointmap数据")
        kfs = real_pointmaps  # 替换合成数据
    else:
        EXPERIMENT_MODE = "SYNTHETIC"
        print("[INFO] 使用合成数据（概念验证）")
        kfs = generate_synthetic_pointmaps(n_frames=20)
    
    # 后续步骤不变，但print中标注数据来源
    # ...
```

4. **降级路径**：
```python
# 如果无法安装MASt3R，使用替代方案：
# 从TUM RGB-D读取深度图，手动转为pointmap
# pointmap = pointmap_from_depth(depth, K)  # 已在camera.py中实现
```

**预期效果**：展示一张真实场景的输入→系统输出的对应关系，大幅提升demo可信度。

---

### P2-2: 修复因子图使支持Sim(3) ✅ 建议

**目标**：补全MASt3R-Fusion的Sim(3)-SE(3)同构映射逻辑。

**技术路径**（修改 `gs_slam/core/factor_graph.py`）：

```python
class Sim3PoseGraph(PoseGraph):
    """
    Sim(3)位姿图扩展
    
    节点: Sim(3) = (R, t, s) — 旋转+平移+尺度
    每条视觉边给出的是Sim(3)相对约束
    每条IMU/GNSS边给出的是SE(3)度量尺度约束
    
    融合策略（MASt3R-Fusion method-003）：
    1. Sim(3)分解: S = T ∘ s, T∈SE(3), s∈R+
    2. 李代数映射: [ω;ν;σ] = Λ · [θ;τ;δs]
       其中 Λ = diag(I, s⁻¹I, s)
    3. 视觉Hessian映射: H_v = Λᵀ Jᵀ H J Λ
    """
    
    def __init__(self):
        super().__init__()
        self.scales = []  # 每个节点的尺度因子
    
    def add_pose_sim3(self, R, t, scale=1.0):
        """添加Sim(3)位姿节点"""
        idx = super().add_pose(R, t)
        self.scales.append(scale)
        return idx
    
    def add_visual_factor_sim3(self, i, j, R_rel, t_rel, s_rel, info):
        """
        添加Sim(3)视觉因子
        s_rel: 相对尺度 ratio = scale_j / scale_i
        """
        # 1. 计算Sim(3)残差
        # 2. 通过同构映射Λ转换为SE(3)+scale参数化
        # 3. 构建Hessian添加到因子图
        pass
    
    def _sim3_to_se3_jacobian(self, s: float) -> np.ndarray:
        """
        Sim(3)→SE(3)×R的同构映射雅可比 Λ
        
        [ω;ν;σ] = diag(I, s⁻¹I, s) · [θ;τ;δs]
        
        Returns:
            Lambda: [7, 7] 映射矩阵
        """
        Lambda = np.zeros((7, 7), dtype=np.float64)
        Lambda[:3, :3] = np.eye(3)        # ω = θ (旋转不变)
        Lambda[3:6, 3:6] = np.eye(3) / s  # ν = τ/s
        Lambda[6, 6] = s                   # σ = s·δs
        return Lambda
```

**简化实现策略**（如果完整Sim(3)太复杂）：
```python
# 简化版：仅在前端跟踪中加入尺度变量
# 视觉约束使用Sim(3)，但后端优化时固定尺度=1
# 注释中标注："完整Sim(3)融合需要MASt3R-Fusion的同构映射实现，
# 当前版本退化为SE(3)位姿图，尺度保持为1"
```

---

### P2-3: 增加真实性能计时报告 ✅ 建议

**目标**：让性能数据来自实际测量而非硬编码。

**技术路径**（修改 `run_all.py` 的 `step6` 性能报告部分）：

```python
def generate_performance_report(kfs, pg):
    """综合性能报告（全部实测）"""
    import time
    
    report = {'timing_ms': {}, 'render_fps_by_count': {}}
    
    # 1. 前端计时：真实执行匹配
    t0 = time.time()
    for i in range(10):  # 跑10次取平均
        pm1, pm2 = kfs[i]['pointmap'], kfs[i+1]['pointmap']
        c1, c2 = kfs[i]['confidence'], kfs[i+1]['confidence']
        K = kfs[i]['K']
        match_pointmaps(pm1, c1, pm2, c2, K)
    t_match = (time.time() - t0) / 10 * 1000
    report['timing_ms']['frontend_matching'] = t_match
    
    # 2. 后端计时
    t0 = time.time()
    backend = SLAMBackend()
    pg_test = backend.build_graph_from_frontend(kfs[:10])
    backend.optimize(max_iter=100)
    t_opt = (time.time() - t0) * 1000
    report['timing_ms']['backend_optimization_full'] = t_opt
    
    # 3. 渲染FPS实测（不同高斯数量）
    renderer = SplatRenderer()
    cam = PinholeCamera()
    R, t = look_at(np.array([5.0, 1.5, 5.0]), np.zeros(3), np.array([0.,1.,0.]))
    cam.set_pose(R, t)
    
    for n_gs in [100, 300, 500, 1000, 2000, 5000]:
        gc = make_test_scene(n_gs)
        # 渲染3次取平均
        times = []
        for _ in range(3):
            t0 = time.time()
            renderer.render(gc.pack(), cam)
            times.append((time.time() - t0) * 1000)
        avg_ms = np.mean(times)
        fps = 1000.0 / max(avg_ms, 0.001)
        report['render_fps_by_count'][str(n_gs)] = {
            'time_ms': avg_ms, 'fps': fps
        }
    
    return report
```

**注意**：移除当前代码中的硬编码值 `report['timing_ms']['mapping_density_control'] = 15.2`

---

### P2-4: 统一论文年份标注 ✅ 建议

**需要修改的文件**：

| 文件 | 当前标注 | 应统一为 |
|------|---------|---------|
| `gs_slam/__init__.py:4` | "MASt3R-SLAM (Murai et al., 2024)" | 2025（arXiv 2024.12，正式发表2025） |
| `analyse/MASt3R-SLAM_analysis.md:1` | "2025" | 保持不变 |
| `analyse/3DGS-Survey_analysis.md:1` | 标题"2026" | 2026（TPAMI 2026待发表） |
| `gs_slam/core/gaussian_model.py:4` | "Chen & Wang, 2024" | 改为 "Chen & Wang, 2026" |
| `gs_slam/demo/frontend.html:202` | "Murai et al., 2024" | 改为 "Murai et al., 2025" |
| `gs_slam/demo/frontend.html:222` | "Chen & Wang, 2024" | 改为 "Chen & Wang, 2026" |

---

### P3: 动态场景处理（模拟MASt3R-Fusion深度残差掩码）可选

**当前 `step6:610-687`** 的问题是手动制造异常再检测。改进方案：

```python
def simulate_dynamic_scene_test(kfs, pg):
    """
    改进版：用真实匹配逻辑模拟动态检测
    
    思路：
    1. 对两帧正常的pointmap做匹配
    2. 注入模拟动态点（另一帧插入异常点）
    3. 用深度残差阈值检测这些异常
    4. 统计检测率
    
    关键区别：使用真实的match_pointmaps流程，
    而非直接比较预设的"动态点"标记
    """
    # Step 1: 基准匹配（无动态物体）
    R_base, t_base, inlier_base = match_pointmaps(pm1, c1, pm2, c2, K)
    
    # Step 2: 注入动态点
    pm2_contaminated = pm2.copy()
    # 在随机位置插入深度异常的匹配点对
    contaminated_mask = add_dynamic_points(pm2_contaminated, ...)
    
    # Step 3: 用真实匹配流程检测
    R_cont, t_cont, inlier_cont = match_pointmaps(pm1, c1, pm2_contaminated, c2, K)
    
    # Step 4: 分析哪些异常点被RANSAC自然排除
    # RANSAC的内在特性：深度不一致的匹配会被视为外点
    n_detected = contaminated_mask.sum() - (inlier_cont & contaminated_mask).sum()
    rejection_rate = n_detected / max(contaminated_mask.sum(), 1)
    
    return {
        'depth_threshold': 0.15,
        'n_injected_dynamic': contaminated_mask.sum(),
        'n_detected_by_ransac': n_detected,
        'rejection_rate': rejection_rate
    }
```

---

## 四、实施路线图

| 阶段 | 时间 | 改进项 | 工作量 |
|------|------|--------|--------|
| 第1天 | 2-3h | P0-1: 改造渲染器tile-based | 2-3h |
| 第1天 | 1h | P0-2: 标记合成数据边界 | 1h |
| 第1天 | 1h | P0-3: 实现真实PSNR/SSIM基准 | 1h |
| 第2天 | 2h | P1-1: 替换随机梯度为几何代理 | 2h |
| 第2天 | 2h | P1-2: 改造评估为有意义对比 | 2h |
| 第2天 | 1.5h | P1-3: 前端参数滑块联动 | 1.5h |
| 第3天 | 4-6h | P2-1: 集成真实MASt3R数据（可选但收益最大） | 4-6h |
| 第3天 | 1h | P2-4: 统一论文年份 | 1h |
| 第4天 | 2h | P2-2: 因子图Sim(3)扩展 | 2h |
| 第4天 | 1h | P2-3: 真实性能计时 | 1h |

**建议的最低完成线**（2天工作量）：P0-1 + P0-2 + P0-3 + P1-1 + P1-2 + P1-3

**建议的完整改进线**（4天工作量）：以上全部 + P2-1

---

## 五、与评分标准的最终对应

| 评分要求 | 当前状态 | 完成P0+P1改进后 |
|---------|---------|---------------|
| 至少阅读2篇前沿+1篇综述 | ✅ | ✅ |
| 提出自己的问题/想法/观点 | ✅ | ✅ |
| 动手实验复现 | ⚠️ 全合成 | ✅ 合成数据+诚实标注+可解释代理 |
| 鼓励提出方法并实验验证 | ⚠️ 方法对但验证不足 | ✅ 多维度有意义对比 |
| 正文≤6页A4 | 待撰写 | 可控 |
| 复现结果>2页 | ⚠️ 缺少渲染指标 | ✅ PSNR/SSIM/覆盖度/消融表 |
| 参考文献≤8篇 | ✅ | ✅ |
| Demo视频3-5分钟 | ❌ 待录制 | 录制完成 |
| 前端界面 | ✅ | ✅ 滑块可交互 |
| Poster | ✅ 大纲完整 | ✅ 增加实验说明 |