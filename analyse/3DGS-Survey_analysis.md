# 论文分析报告: A Survey on 3D Gaussian Splatting

- **分析时间**: 2026-06-08T03:25:08.943418
- **作者**: Guikun Chen (The State Key Lab of Brain-Machine Intelligence, Zhejiang University, China), Wenguan Wang (The State Key Lab of Brain-Machine Intelligence, Zhejiang University, China)
- **年份**: 2026
- **DOI**: [10.1145/nnnnnnn.nnnnnnn](https://doi.org/10.1145/nnnnnnn.nnnnnnn)

## 摘要
3D Gaussian splatting (GS) has emerged as a transformative technique in radiance fields. Unlike mainstream implicit neural models, 3D GS uses millions of learnable 3D Gaussians for an explicit scene representation. Paired with a differentiable rendering algorithm, this approach achieves real-time rendering and unprecedented editability, making it a potential game-changer for 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.

## 方法论

### method-001 3D Gaussian Splatting Pipeline [系统设计]

将场景显式表示为大量可学习的3D高斯体，这些高斯体携带位置、协方差、颜色和不透明度等属性。通过一个基于点的可微渲染器，将高斯体投影到图像平面并执行从前到后的alpha混合，实现快速的新视角合成。整个管线采用从运动恢复结构稀疏点云初始化，并利用梯度下降与自适应密度控制进行端到端优化。

**创新点**:
- 用显式3D高斯替代隐式神经场，实现了实时渲染与高度可编辑性
- 将可微渲染与基于点的分块并行算法结合，保持高保真度的同时极大提升渲染速度
- 提出自适应密度控制策略，使高斯数量与场景复杂度动态匹配

**输入**: 多视点 RGB 图像及其对应的相机位姿与稀疏点云
**输出**: 可实时渲染的可编辑3D高斯场景表示，同时可输出任意新视角的高质量图像

**步骤**:
从稀疏SfM点云初始化3D高斯的中心位置，并设置初始颜色、尺度、旋转和不透明度。
对每个训练视图，将3D高斯通过投影变换映射到屏幕空间，形成2D高斯核。
按深度排序所有2D高斯，并利用基于块的并行光栅化器逐像素进行alpha混合计算颜色。
计算渲染图像与真实图像的加权L1与SSIM组合损失，反向传播梯度更新高斯的所有属性。
在优化过程中定期执行自适应密度控制：对梯度大的高斯进行克隆或分裂，并移除不透明度低的高斯。

**关键公式**:

$$
\mathbf{C} = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i
$$

$$
T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

### method-002 Tile-based Differentiable Rasterizer [模型架构]

该可微光栅化器将屏幕划分为多个瓦片，每个瓦片内的像素并行处理经过深度排序的2D高斯。通过从前到后的alpha混合计算每个像素的颜色，并允许梯度从损失函数回传至所有高斯参数，包括位置、协方差、颜色和不透明度。这种设计充分利用GPU的大规模并行能力，避免了传统射线行进的冗余计算。

**创新点**:
- 首次将有序alpha混合与可微流水线结合用于3D高斯表示
- 采用基于瓦片的并行策略和提前终止机制，大幅提高渲染效率
- 直接反向传播梯度至高斯几何与外观属性，支持端到端学习

**输入**: 当前视图下所有3D高斯体的位置、协方差、颜色、不透明度，以及相机参数
**输出**: 渲染出的RGB图像，同时为梯度下降提供所有高斯参数的梯度

**步骤**:
将图像空间划分为固定大小的瓦片，并为每个瓦片分配需要渲染的高斯列表。
将所有3D高斯投影到屏幕后，按深度快速排序形成全局有序高斯序列。
每个瓦片内的像素按顺序遍历分配的高斯，计算当前高斯的透明度贡献并累加颜色。
当像素累积的不透明度超过阈值或遍历完所有高斯时，提前终止该像素的计算。
在反向传播时，根据混合公式将梯度分配到参与计算的高斯的各个属性。

**关键公式**:

$$
\mathbf{C}(\mathbf{x}) = \sum_{i=1}^{N} \mathbf{c}_i \, \alpha_i(\mathbf{x}) \prod_{j=1}^{i-1} (1 - \alpha_j(\mathbf{x}))
$$

$$
\alpha_i(\mathbf{x}) = o_i \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_i')^T \boldsymbol{\Sigma}_i'^{-1} (\mathbf{x} - \boldsymbol{\mu}_i')\right)
$$

### method-003 Adaptive Density Control [训练策略]

在优化过程中，通过监测每个高斯体接收到的梯度大小来动态调整场景中的高斯密度。对于视野空间中累积梯度超过阈值的小高斯进行克隆（以覆盖细节），对梯度大且本身过大的高斯进行分裂（细化几何），同时移除不透明度极低或占位过大的冗余高斯。该策略使场景表示在训练过程中逐步细化，在复杂区域增加细节，在空白区域降低消耗。

**创新点**:
- 实现了完全自动化的、基于梯度的密度调节，无需人工设置区域细节级别
- 克隆与分裂策略能够从稀疏初始点云逐步增长出完整场景结构
- 与可微渲染紧密结合，密度调整直接响应损失信号，有效减少空洞和漂浮物

**输入**: 当前所有高斯体的属性及其累积的视图空间梯度
**输出**: 更新后的高斯体集合，包含新增、调整或移除的实体

**步骤**:
在优化开始后，每经过若干次迭代触发一次密度控制。
对每个高斯，计算其在视图空间中的梯度范数，作为该高斯重要性的度量。
对于梯度超过阈值且尺度较小的高斯，在当前位置克隆一份新的高斯，并共享其他属性。
对于梯度超过阈值且尺度较大的高斯，将其分裂为两个更小的高斯，并按比例分配初始位置。
移除不透明度低于阈值或世界空间尺度超过上限的无效高斯。

### method-004 Combined L1-SSIM Loss [损失函数]

采用L1像素损失与结构相似性指数（SSIM）损失的加权组合作为优化目标。L1损失直接约束每个像素的亮度差异，保证重建的准确性；SSIM损失则从亮度、对比度和结构三个方面度量图像块之间的感知相似性，有助于保留高频纹理和几何细节。这种组合既能快速收敛，又能提升视觉质量。

**创新点**:
- 将像素级精确损失与感知结构损失直接联合优化显式高斯表示
- 相比纯L1或纯SSIM损失，该组合在收敛速度和最终质量之间取得更好平衡
- λ系数可调以适应不同场景类型，简单有效

**输入**: 模型渲染的图像和对应的真实参考图像
**输出**: 标量损失值，用于驱动整个模型的参数更新

**步骤**:
对每个训练批次，使用当前的高斯模型渲染一组视点的图像。
逐像素计算渲染图像与真实图像之间的L1距离。
在滑动窗口上计算两幅图像的SSIM值，并转换为损失（1 - SSIM）。
将L1损失与SSIM损失按预设权重 λ 进行线性组合。
将组合损失反向传播，更新高斯参数。

**关键公式**:

$$
L = (1 - \lambda) \, L_1 + \lambda \, (1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}}))
$$

### method-005 SfM-based Gaussian Initialization [数据预处理]

利用运动恢复结构从输入多视图像中重建的稀疏3D点云作为高斯体的初始中心位置，并将其颜色初始化为对应观测的图像颜色。每个高斯的协方差则根据最近邻点的平均距离初始化为各向同性，不透明度设为常数值。这一初始化策略为后续优化提供了合理的几何起点，显著加速收敛并减少陷入局部极小的风险。

**创新点**:
- 使用现成的SfM输出作为显式先验，避免随机初始化导致的收敛困难
- 将结构信息直接编码到离散基元中，继承了点云表示的优势
- 支持直接处理真实世界无序图像集，无需额外深度或形状先验

**输入**: 一组从不同视角拍摄的场景RGB图像
**输出**: 一组初步具有几何和颜色属性的3D高斯体，以及估计的相机内外参数

**步骤**:
对输入的多视图像集运行运动恢复结构算法，得到稀疏3D点云及相机位姿。
提取每个3D点的坐标作为高斯中心，从可见视图中采样颜色作为高斯初始颜色。
基于点云密度计算每个高斯的初始尺度，并设置旋转为单位四元数。
将所有高斯的不透明度初始化为同一标准值（如0.1）。
将构建的初始高斯集直接用于后续优化和渲染步骤。

## 核心观点

### claim-001 [设计] [→ method-001, method-002]
> **通过显式3D高斯与瓦片可微光栅化替代MLP查询，实现实时高质量新视角合成。**
- **问题**: 隐式神经辐射场（NeRF）渲染速度慢，且难以直接编辑场景，限制了实时交互和内容创作等应用。
- **方法**: 采用显式3D高斯体表示场景，结合基于瓦片的可微光栅化器（tile-based differentiable rasterizer）进行前向splatting渲染。
- **机制**: 通过将场景表达为离散的3D高斯基元，并将所有高斯投影到图像空间进行从前到后的alpha混合，避免了NeRF的逐点射线采样与MLP查询；同时瓦片分块并行处理大幅利用GPU并行能力，梯度可直接回传至高斯参数，从而在保持高保真度的前提下实现实时渲染和可编辑性。
- **结果**: 在多个任务中实现了实时渲染，例如在Replica静态场景上GSSLAM达到769 FPS，渲染质量（PSNR 37.50、SSIM 0.96）超越现有NeRF方法；在其他基准上也取得高保真度重建。
- **前提**: 场景的光场可由大量各向异性3D高斯体充分近似；训练时需提供多视角图像及相应的相机位姿；GPU需支持大规模并行计算和足够的共享内存。
- **来源**: 第1节: INTRODUCTION; 第43节: Background
- **隐含假设**: 场景的几何与外观可由有限数量的3D高斯体有效表示; 高斯体在投影平面上的线性近似与alpha混合模型足够准确; 相机内参和外参已知或可精确估计; GPU硬件支持瓦片化的并行渲染和共享内存访问
> 原文: [第1节] 3D GS addressed this need by introducing an advanced, explicit scene representation that models a scene using millions of learnable 3D Gaussians in space. Unlike the implicit, coordinate-based models, 3D GS employs an explicit representation and highly parallelized workflows, facilitating more
  *置信度: 85%*

### claim-002 [设计] [→ method-003]
> **通过梯度驱动的自适应密度控制动态调整高斯体数量，增强细节并减少冗余。**
- **问题**: 固定数量的高斯体难以适应不同场景区域复杂度的变化，导致细节区域欠重建而空旷区域冗余计算。
- **方法**: 自适应密度控制策略，在优化过程中根据高斯体接收的梯度大小动态克隆、分裂或移除高斯体。
- **机制**: 缺失：原文未论述。
- **结果**: 减少了空洞和漂浮物，提升了场景重建的完整性和细节水平，使得3D GS系列方法在各类基准上优于固定基元的表示。
- **前提**: 梯度阈值和分裂/克隆策略的超参数需适当设置；优化过程中梯度信号需足够稳定以可靠指导密度调整。
- **来源**: 第5节: MLP
- **隐含假设**: 梯度的大小能有效反映高斯体所在区域的表示误差; 克隆和分裂后的高斯体能被后续优化快速调整到正确位置; 场景的几何结构允许通过增加高斯数量来逼近更多细节
> 原文: [第5.6节] An additional challenge lies in maintaining visual quality, as large-scale scenes often feature texture-less surfaces that can hamper the effectiveness of optimization such as Gaussian initialization and density control (Sec. 3.2).
  *置信度: 40%*

### claim-003 [实验] [→ method-004]
> **联合L1像素损失与SSIM结构损失优化，兼顾重建精度和视觉纹理细节。**
- **问题**: 纯L1损失收敛较慢且易忽略纹理细节，纯SSIM损失可能导致颜色偏移或不稳定，需要一种平衡的重建目标。
- **方法**: 将L1像素损失与SSIM结构相似性损失加权组合作为总损失函数。
- **机制**: 缺失：原文未论述。
- **结果**: 在各类场景重建中同时获得了高PSNR和高SSIM，例如在静态场景Gaussian-SLAM达到PSNR 38.90、SSIM 0.99，动态场景D-3DGS达到PSNR 39.51、SSIM 0.99，验证了该组合损失对收敛速度和视觉质量的有效平衡。
- **前提**: SSIM权重 λ 需根据场景特点调优；假设训练数据中颜色分布无显著异常值，且L1和SSIM梯度方向在训练中足够一致。
- **来源**: 第6节: RGB
- **隐含假设**: L1和SSIM损失在优化过程中不会产生严重冲突; λ系数可在不同场景中通过经验或启发性规则确定; 图像亮度、对比度和结构的变化可由SSIM有效捕捉
> 原文: 缺失：原文未论述组合损失的具体机制，相关效果隐含在性能比较中，如[第6.2节] Table S5中Gaussian-SLAM等使用该损失的方法取得了优异指标。
  *置信度: 35%*

### claim-004 [设计] [→ method-005]
> **利用运动恢复结构稀疏点云初始化高斯体，加速收敛并提升重建质量。**
- **问题**: 从随机初始点开始优化3D高斯体的位置和形状容易收敛缓慢或陷入局部极小，难以从无序图像集重建完整场景。
- **方法**: 利用运动恢复结构（SfM）得到的稀疏3D点云作为高斯体中心的初始值，并根据邻近点距离初始化协方差。
- **机制**: 缺失：原文未论述。
- **结果**: 提供了合理的几何起点，加快了训练收敛，使得3D GS能够可靠地从真实世界无序图像集进行高质量重建，并在各项基准上表现出色。
- **前提**: SfM能够成功恢复稀疏几何且覆盖场景主要部分；输入图像具有足够的纹理和视角重叠以支持SfM。
- **来源**: 第5节: MLP
- **隐含假设**: SfM点云与场景表面大致重合; 点云密度足以捕捉局部细节的初始化需求; 图像曝光和光照一致性足够让SfM正确匹配
> 原文: [第5.6节] ... Gaussian initialization and density control (Sec. 3.2) ...
  *置信度: 40%*

## 局限性分析

- **[实验]** (较高) 综述中的性能比较主要依赖于原论文报告的数字，并未在不同方法间进行统一超参数调优、相同计算资源的重新评估。由于各方法的训练策略、评估协议和硬件环境存在差异，直接对比可能导致部分结论不公平，无法准确反映方法间的相对优劣。
  > 建议: 在公开统一的数据集和硬件条件下重新训练和评估主要基线，或至少进行总结性元分析，注明各结果所对应的具体实验设置差异。

- **[可复现性]** (较高) 作为一篇系统综述，文中并未提供完整的实验代码、标准化的基准配置（如学习率、调度策略、训练轮次）及评估脚本，读者难以复现所引用的性能数字或验证比较分析的公平性。
  > 建议: 建议提供统一的基准代码库与配置文件，或至少以附录形式明确给出各比较方法的详细超参数和评估流程，提升综述的实用参考价值。

- **[方法]** (中等) 对3D GS关键组件的工作原理（如自适应密度控制中基于梯度累积的克隆/分裂机制、L1+SSIM组合损失在优化中的交互作用）解释不足，很多处仅作现象描述而缺少对设计动机和内在逻辑的深入剖析，削弱了综述作为教学和指导性文献的作用。
  > 建议: 增加对核心机制的原理性阐述，可结合公式推导或伪代码说明，并讨论各设计选择背后的直觉与替代方案，增强文章深度。

- **[理论]** (中等) 综述对3D GS方法的理论性质讨论薄弱。例如，未涉及显式高斯表示的表示能力上界、优化的收敛性保证、瓦片渲染引入的近似误差分析等数学基础，使文章局限于工程总结而缺乏理论指导。
  > 建议: 增加一节专门讨论3D GS的理论分析现状与开放问题，总结现有的收敛/误差界结果，并指出未来理论探索方向。

- **[泛化性]** (中等) 所讨论的应用和评估主要集中在受控的学术基准（如Replica、标准多视图数据集），对于真实世界大规模、无约束场景（如自动驾驶街景、长尾物体、剧烈光照变化）的泛化能力讨论不够，无法全面反映3D GS在复杂环境中的实际可用性。
  > 建议: 增加对更具挑战性的真实世界数据集上的实验分析，或单独讨论3D GS在域外泛化中的表现与局限，以帮助读者了解其适用边界。

- **[伦理]** (较低) 3D GS具有高保真实时渲染和强可编辑性，容易被用于生成虚假或欺骗性视觉内容（如深度伪造、假场景）。论文完全没有提及这一技术的社会影响和潜在伦理风险，也未建议任何防护或使用规范。
  > 建议: 在结论或专门章节中加入对潜在滥用风险的警示，并建议社区建立负责任使用指南或检测手段。

## 总结
本文系统综述了3D Gaussian Splatting（3D GS）渲染管线及其核心组件，包括基于瓦片的可微光栅化器、自适应密度控制、L1-SSIM联合损失以及基于运动恢复结构的初始化策略。这些组件共同构成了一种显式场景表示与快速并行渲染的新范式，有效突破了隐式神经辐射场（NeRF）实时性差、不易编辑的瓶颈。论文梳理了各技术环节的设计动机，展示了3D GS方法在静态与动态场景重建中相较于传统NeRF的性能优势。

主要关键发现表明，3D GS通过可微splatting渲染与自适应基元优化，能够在多个基准上达到实时帧率（如769 FPS）和高保真度（PSNR 37.50、SSIM 0.96）。自适应密度控制与混合损失函数进一步提升了复杂几何和纹理细节的重建质量。然而，综述的性能对比主要依赖原始论文报告的数据，缺乏统一超参数调优和等同计算资源下的重评，其结论的公平性与稳健性受到一定限制。同时，对关键机制（如密度控制的梯度响应原理、L1与SSIM的优化交互）的理论分析深度不足，对真实大规模场景的泛化性讨论亦较薄弱。

整体而言，作为2026年的早期综述，该文及时归纳了3D GS技术的核心思想与应用成效，为后续研究提供了清晰的脉络。其优势在于系统整合了分散的方法论，提升了该领域的可理解性；不足之处在于缺乏标准化的复现基准、深层理论剖析以及对伦理风险的警觉，反映出当时综述工作的普遍局限。未来工作可围绕统一评估协议、理论性质证明以及面向开放环境的泛化展开，以增强该管线的可靠性与实用价值。