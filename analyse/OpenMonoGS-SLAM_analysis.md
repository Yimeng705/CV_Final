# 论文分析报告: OpenMonoGS-SLAM: Monocular Gaussian Splatting SLAM with Open-set Semantics

- **分析时间**: 2026-06-08T02:52:56.657237
- **作者**: Jisang Yoo (Sungkyunkwan University), Gyeongjin Kang (Sungkyunkwan University), Hyun-kyu Ko (Sungkyunkwan University), Hyeonwoo Yu (Sungkyunkwan University), Eunbyung Park (Yonsei University)

## 摘要
Simultaneous Localization and Mapping (SLAM) is a foundational component in robotics, AR/VR, and autonomous systems. With the rising focus on spatial AI in recent years, combining SLAM with semantic understanding has become increasingly important for enabling intelligent perception and interaction. Recent efforts have explored this integration, but they often rely on depth sensors or closed-set semantic models, limiting their scalability and adaptability in open-world environments. In this work, we present OpenMonoGS-SLAM, the first monocular SLAM framework that unifies 3D Gaussian Splatting (3DGS) with open-set semantic understanding. To achieve our goal, we leverage recent advances in Visual Foundation Models (VFMs), including MASt3R for visual geometry and SAM and CLIP for open-vocabulary semantics. These models provide robust generalization across diverse tasks, enabling accurate monocular camera tracking and mapping, as well as a rich understanding of semantics in open-world environments. Our method operates without any depth input or 3D semantic ground truth, relying solely on self-supervised learning objectives. Furthermore, we propose a memory mechanism specifically designed to manage high-dimensional semantic features, which effectively constructs Gaussian semantic feature maps, leading to strong overall performance. Experimental results demonstrate that our approach achieves performance comparable to or surpassing existing baselines in both closed-set and open-set segmentation tasks, all without relying on supplementary sensors such as depth maps or semantic annotations.

## 方法论

### method-001 OpenMonoGS-SLAM 整体框架 [系统设计]

OpenMonoGS-SLAM 是一个单目开放词汇语义SLAM系统，它统一了3D高斯泼溅（3DGS）与视觉基础模型MASt3R、SAM和CLIP。系统利用MASt3R提供几何对应与初始位姿，SAM生成无类别的2D掩码，CLIP提取语言特征，并通过记忆机制和多尺度监督在完全自监督下联合优化场景几何与语义，实现无需深度和语义真值的相机跟踪与场景重建。

**创新点**:
- 首个将3DGS与开放集语义VFM结合的单目SLAM系统
- 完全自监督，无需任何深度传感器或3D语义真值
- 集成记忆机制稳定处理高维动态语义特征

**输入**: 单目RGB视频流
**输出**: 相机轨迹、附带颜色与开放词汇语义特征的3D高斯场景，以及2D开放词汇分割掩码

**步骤**:
对输入单目RGB序列预处理，统一缩放至512像素宽度
使用MASt3R提取帧间密集对应和初始相机位姿
使用SAM为每帧生成无类别的物体掩码
使用CLIP为每个掩码区域提取高维语言特征，并送入记忆库累积多帧信息
初始化含有可学习语义特征向量的3D高斯表示
可微渲染RGB图像与高斯语义特征图
联合优化光度损失、几何对应损失、语言回归损失与多视图对比损失
按每10帧添加映射帧，迭代30K次完成场景与语义学习

**关键公式**:

$$
L_{\text{total}} = \lambda_{\text{photo}} L_{\text{photo}} + \lambda_{\text{corr}} L_{\text{corr}} + \lambda_{\text{lang}} L_{\text{lang}}
$$

### method-002 记忆驱动的语义特征聚合 [训练策略]

针对单帧CLIP特征噪声大、跨帧不一致的问题，本方法维护一个动态记忆库，持续累积由SAM掩码过滤的历史CLIP特征。在优化高斯语义特征时，将当前帧特征与记忆库中存储的稳定特征进行注意力融合，生成一致的目标语义向量，用于语言回归损失监督，从而显著提升语义分割的时空一致性。

**创新点**:
- 首次在3DGS语义SLAM中引入动态记忆库管理开放词汇特征
- 大幅缓解单帧特征噪声，使开放集分割性能提升明显

**输入**: 由SAM掩码过滤的每帧CLIP特征向量
**输出**: 更新后的记忆库，以及用于监督的稳定目标语义特征

**步骤**:
对每帧使用SAM提取物体掩码，并在各掩码区域上用CLIP编码得到特征向量
按余弦相似度阈值 τ_m=0.9 将新特征与记忆库条目匹配，更新或插入条目
每次迭代中，为每个高斯检索记忆库中对应的目标语义特征
计算渲染高斯特征与检索到的记忆特征的均方误差（回归损失）
通过梯度回传同时优化高斯特征和记忆特征表示

### method-003 多尺度语义监督 [训练策略]

为了让模型能同时处理不同大小的物体，该方法在多个图像尺度（S=4）上计算语义损失。在不同分辨率下同时监督渲染的高斯语义特征图，使粗尺度维持整体语义一致性，细尺度保留细节和边界，避免单一尺度过合并或噪声过大的问题。

**创新点**:
- 在3DGS语义SLAM中提出多尺度监督，有效提升不同粒度物体的分割质量
- 通过粗细尺度联合优化平衡全局形状和局部细节

**输入**: 渲染的语义特征图，目标语义特征图
**输出**: 多尺度语义损失

**步骤**:
将渲染的语义特征图和对应的目标特征图分别下采样或保持为多个尺度
在每个尺度上分别计算语言回归损失和多视图对比损失
对所有尺度的损失求和，形成总的多尺度语义损失 L_lang

**关键公式**:

$$
L_{\text{lang}} = \sum_{s=1}^{S} \left( \lambda_{\text{reg}} L_{\text{reg}}^{(s)} + \lambda_{\text{contrast}} L_{\text{contrast}}^{(s)} \right)
$$

### method-004 多视图对比语义损失 [损失函数]

该损失在不同训练视图之间强化语义特征的一致性。通过从两个视图的渲染特征图中，利用SAM掩码确定相同语义区域作为正样本对、其他区域作为负样本对，采用InfoNCE对比学习拉近正样本、推远负样本，从而增强跨视图的语义不变性，减轻单视图监督的空间碎片化。

**创新点**:
- 首次在多视图3DGS场景中引入对比学习以对齐开放词汇语义特征
- 有效解决仅用单视图语义监督造成的空间碎片化问题

**输入**: 两个视图的渲染语义特征图及对应的SAM掩码
**输出**: 对比损失值

**步骤**:
从两个训练帧渲染语义特征图
利用SAM掩码定位各语义区域，提取对应位置的渲染特征向量
构造正样本对（不同视图下同一区域的特征）和负样本对（不同区域的特征）
计算InfoNCE形式的对比损失
将该损失加入总损失驱动优化

**关键公式**:

$$
L_{\text{contrast}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\text{sim}(z_i,z_i^+)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(z_i,z_j)/\tau)}
$$

### method-005 基于MASt3R的几何对应与相机跟踪 [系统设计]

系统借助MASt3R大规模预训练模型从单目图像对中提取密集几何对应，替代传统手工特征或深度传感器。通过最小化渲染坐标图与MASt3R对应点之间的重投影误差，并结合光度误差，实现鲁棒的单目相机位姿估计与场景几何重建，保障跟踪的准确性和稳定性。

**创新点**:
- 利用大规模预训练VFM提供密集几何先验，使单目SLAM摆脱深度传感器依赖
- 显著提升在室内场景下单目相机跟踪的精度与鲁棒性

**输入**: 连续的单目RGB图像帧
**输出**: 相机位姿（旋转和平移），场景几何的3D高斯表示

**步骤**:
用MASt3R处理当前帧与关键帧，输出密集像素对应和初始相对位姿
通过3DGS渲染当前估计的深度或坐标图
计算渲染坐标与MASt3R对应点的L1重投影损失 L_corr
结合光度损失联合优化相机位姿和3D高斯参数
根据MASt3R对应质量选择关键帧以确保几何覆盖

**关键公式**:

$$
L_{\text{corr}} = \frac{1}{|\mathcal{C}|}\sum_{(p,q)\in\mathcal{C}} \lVert \Pi\left( \mathbf{P}(p) \right) - q \rVert_1
$$

## 核心观点

### claim-001 [对比] [→ method-001]
> **通过统一3D高斯泼溅与视觉基础模型，实现单目开放词汇SLAM，性能媲美RGB-D方法。**
- **问题**: 现有单目视觉SLAM难以在无深度传感器下实现开放词汇语义理解，依赖闭集语义注释或深度输入。
- **方法**: 提出OpenMonoGS-SLAM框架，统一3D高斯泼溅（3DGS）与视觉基础模型MASt3R、SAM、CLIP，仅用RGB自监督联合优化几何与语义。
- **机制**: MASt3R提供密集几何对应与初始位姿，SAM生成无类别掩码，CLIP提供语言特征；通过3DGS可微渲染将几何和语义映射到高斯场，利用自监督目标（光度、几何对应、语义回归）联合优化，从而无需深度和语义真值即可从RGB端到端学习场景表示。
- **结果**: 在Replica等基准上，新视角渲染PSNR达34.47，相机跟踪ATE仅1.60 cm，开集分割mIoU达0.845，均优于或可比基于RGB-D的方法。
- **前提**: 场景与VFM训练数据分布相近，VFM输出质量高（MASt3R匹配准确、SAM分割可靠、CLIP特征判别），且场景相对静态。
- **来源**: 第1节: RGB
- **隐含假设**: 预训练VFMs在新场景具有足够泛化能力; SAM生成的掩码正确分离物体; 场景不包含剧烈运动或动态物体（该方法对动态场景不鲁棒）
> 原文: [摘要] We present OpenMonoGS-SLAM, the first monocular SLAM framework that unifies 3D Gaussian Splatting (3DGS) with open-set semantic understanding. ... Experimental results demonstrate that our approach achieves performance comparable to or surpassing existing baselines ... all without relying on su
  *置信度: 90%*

### claim-002 [实验] [→ method-002]
> **记忆驱动的特征聚合利用历史稳定语义特征，克服单帧噪声，显著提升分割时空一致性。**
- **问题**: 单帧CLIP特征噪声大且跨帧不一致，导致开放词汇语义学习不稳定。
- **方法**: 维护动态记忆库，累积历史帧的SAM掩码过滤后的CLIP特征；在优化高斯语义特征时，通过注意力机制融合当前帧特征与记忆库稳定特征，监督语言回归损失。
- **机制**: 记忆库聚合多帧CLIP嵌入提供时域一致的语义参考，过滤单帧噪声和遮挡，使监督信号更鲁棒，引导高斯特征学习到稳定的开放词汇表示。
- **结果**: 消融实验显示，去除记忆模块后mIoU下降21%，FWIoU下降10%，像素准确率下降约10%。
- **前提**: SAM掩码能有效分离前景语义区域；场景中语义对象的CLIP特征在时序上具有一致性（相对静态或缓慢变化）。
- **来源**: 第4节: RGB
- **隐含假设**: SAM输出的掩码准确覆盖语义对象; 场景中语义对象的CLIP特征在时间上具有一致性（无剧烈光照变化或视角突变导致特征偏移）
> 原文: [消融实验] We also ablate the memory attention mechanism by removing the memory bank ... Without the memory bank ('w/o memory'), mIoU drops by 21%, and both FWIoU and accuracy decrease by 10%, confirming the pivotal role of memory-based attention in semantic learning. This suggests that aggregating CLIP
  *置信度: 95%*

### claim-003 [实验] [→ method-003]
> **多尺度语义损失同时监督粗细粒度，平衡整体一致性与边界细节，改善分割质量。**
- **问题**: 单一尺度语义监督无法同时保持大尺度整体一致性和小尺度细节，导致大物体过合并或小物体碎片化。
- **方法**: 在S=4个图像尺度上计算语义损失，同时监督不同分辨率下的渲染语义特征图。
- **机制**: 粗尺度损失强制大范围语义一致性，细尺度损失保留边界和细节，多尺度联合优化使高斯特征适应不同粒度目标，平衡全局形状与局部结构。
- **结果**: 消融显示仅粗尺度或仅细尺度的mIoU、FWIoU和Accuracy均低于全模型；定性结果中全尺度模型生成更连贯且细节更好的掩码。
- **前提**: 尺度数量和范围合适（S=4），不同尺度损失权重合理，且下采样方式保留主要语义结构。
- **来源**: 第4节: RGB
- **隐含假设**: 多尺度金字塔下采样不丢失关键语义拓扑; 不同尺度的特征图能捕捉不同粒度的语义信息; CLIP特征本身具有一定尺度不变性
> 原文: [消融实验] We further ablate the proposed multi-scale strategy by training with only a single scale level ... using a single scale consistently underperforms the full model, indicating that multi-scale supervision is crucial for handling objects with diverse sizes and granularities. Qualitative results 
  *置信度: 95%*

### claim-004 [实验] [→ method-004]
> **多视图对比损失强制跨视图语义不变性，消除空间碎片化，大幅提升分割鲁棒性。**
- **问题**: 仅使用单视图语义损失导致不同视图间语义特征不一致，引发空间碎片化（同一对象在不同视角被预测为不同类别）。
- **方法**: 引入多视图对比语义损失，利用SAM掩码从两个视图的渲染特征图中构建正样本对（同语义区域）和负样本对（其他区域），采用InfoNCE对比学习。
- **机制**: 对比学习强制同语义区域特征在多视图中相似，不同语义区域特征远离，从而增强跨视图语义不变性，消除单视图伪影，使3D语义特征空间连续一致。
- **结果**: 去除对比损失后mIoU下降约27%，FWIoU和Acc明显下降，定性观察到预测掩码明显碎片化。
- **前提**: 不同视图之间存在足够共视区域和语义重叠，SAM掩码准确划分区域，且负样本中不含同语义对象（硬负样本可控）。
- **来源**: 第4节: RGB
- **隐含假设**: 不同视图确实覆盖相同语义对象的部分区域; SAM掩码在多个视图中对同一对象有较一致的覆盖; InfoNCE对比学习在此设定下能有效收敛
> 原文: [消融实验] Removing the multi-view contrastive loss ('w/o contrastive loss') causes a substantial mIoU drop of approximately 27%, accompanied by clear declines in FWIoU and pixel accuracy, highlighting the importance of enforcing cross-view semantic consistency. ... In particular, without the contrastiv
  *置信度: 95%*

### claim-005 [对比] [→ method-005]
> **借助MASt3R密集几何对应优化位姿，实现鲁棒单目跟踪，ATE仅1.60 cm。**
- **问题**: 单目SLAM缺乏深度输入，传统基于手工特征或单目深度估计的跟踪方法在室内场景中精度和鲁棒性不足。
- **方法**: 利用MASt3R从图像对中提取密集几何对应，最小化渲染坐标图与MASt3R对应点重投影误差，联合光度误差进行相机位姿优化。
- **机制**: MASt3R在大规模数据上预训练，提供鲁棒的密集几何先验，替代手工特征匹配，即使在弱纹理区域也能获得稳定对应，从而通过强几何约束减少漂移，提升跟踪精度。
- **结果**: 在Replica数据集上平均ATE RMSE为1.60 cm，显著低于其他单目方法；在TUM-D和ScanNet上也取得最低或次低ATE。
- **前提**: 场景外观和几何结构与MASt3R训练数据兼容，图像对具有足够重叠，无快速运动模糊或大范围动态遮挡。
- **来源**: 第3节: RGB
- **隐含假设**: MASt3R预训练权重具有跨场景泛化能力; 图像序列连续，帧间变化平滑，可提取有效对应; 渲染坐标图能有效匹配MASt3R的密集对应
> 原文: [实验节] Camera Tracking. As shown in Tab. IV, our method significantly outperforms existing monocular visual SLAM approaches in absolute trajectory accuracy. ... We attribute this stability to visual foundation models trained on large-scale and diverse data, which provide rich semantic and geometric p
  *置信度: 90%*

## 局限性分析

- **[方法]** (严重) 方法的核心定位依赖于MASt3R提供的初始位姿和密集几何对应。如果MASt3R在弱纹理、重复纹理或大视角变化场景下产生错误匹配，会导致相机位姿严重漂移，且高斯场的几何结构也会被污染。系统缺乏对这一关键模块的故障检测和恢复机制。
  > 建议: 引入多源几何线索（如光流、单目深度估计）进行加权融合，并增加基于对称极线距离的离群点滤波和鲁棒优化回环，以降低对单一预训练模型的致命依赖。

- **[方法]** (较高) 记忆模块建立在场景静态的假设上，动态物体会将错误的CLIP特征累积到记忆库中，污染后续帧的语义监督信号，造成语义特征学习混乱和跟踪精度下降。论文未涉及任何动态物体处理。
  > 建议: 增加运动物体分割或语义掩码时序一致性检查，对记忆库中的特征进行动态遗忘或不确定性加权，并加入动态物体感知的跟踪损失。

- **[泛化性]** (较高) 所有实验均在小型室内数据集（Replica、TUM‑D、ScanNet）上完成，未在室外大规模场景、显著动态环境、剧烈光照变化等真实开放世界中测试，泛化能力缺乏验证。
  > 建议: 在KITTI、Waymo、TartanAir等涵盖室外、动态和多种天气的数据集上补充评估，展示方法在不同分布下的鲁棒性边界。

- **[可复现性]** (较高) 论文未提及代码开源计划，方法需同时加载MASt3R、SAM和CLIP等多个大型预训练模型，计算和显存开销极高，但文中未报告实时性、GPU显存消耗等部署关键指标，严重阻碍复现和实际应用。
  > 建议: 开放源代码并详细记录运行环境，补充单帧跟踪延迟、整体FPS、峰值显存等计算效率数据，探索模型剪枝或高效推理方案以促进社区验证。

- **[实验]** (中等) 消融实验未考察关键超参数的敏感性（如多尺度损失的尺度数S、对比损失的温度系数、记忆库大小等），也未报告多次实验的方差或统计显著性检验，结果可靠性不足。
  > 建议: 增加超参数敏感性分析图表，进行多次随机种子实验并汇报均值和标准差，必要时补充统计显著性检验。

- **[伦理]** (中等) 开放词汇语义功能基于CLIP模型，可能继承其训练数据中的种族、性别等社会偏见；同时，精细的语义建图能力若被滥用，可能引发大规模监控和隐私侵犯，论文未做任何伦理讨论。
  > 建议: 增加对模型偏见的说明，并建议在部署时遵守隐私法规，可讨论差分隐私或数据脱敏等防护手段。

## 总结
本论文提出了OpenMonoGS-SLAM，一个单目开放词汇语义SLAM系统，创新性地将3D高斯泼溅与MASt3R、SAM和CLIP三类大型视觉基础模型相融合，在完全自监督的RGB输入下联合优化几何与语义，无需深度传感器或语义真值。该方法设计了记忆驱动的语义特征聚合、多尺度语义监督以及多视图对比语义损失等关键策略，以应对单帧CLIP噪声大、跨视图不一致等开放词汇学习难题，并利用MASt3R的密集几何对应实现鲁棒的相机跟踪。

在Replica、TUM-D和ScanNet等室内数据集上，系统取得了新视角渲染PSNR 34.47、相机跟踪ATE 1.60 cm和开放词汇分割mIoU 0.845的性能，均达到或超越现有单目乃至部分RGB-D方法。详尽的消融实验证实：去除记忆模块导致mIoU下降21%，去除多视图对比损失则下降约27%，多尺度监督亦对精度有显著贡献，表明各组件对于最终性能均不可或缺。

然而，该工作存在若干重要局限。方法的核心功能严重依赖MASt3R的匹配质量，在弱纹理或动态场景下可能发生位姿漂移；记忆库建立在静态场景假设之上，无法处理运动物体；所有实验局限于小型室内环境，泛化能力缺乏验证。此外，论文未报告计算开销、实时性能及代码开源情况，复现性与实际部署前景存疑。整体而言，该研究为单目开放词汇SLAM提供了新颖的思路和强有力的实验支撑，但在鲁棒性、效率与泛化性方面仍需深入改进。