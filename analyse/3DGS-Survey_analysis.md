# 论文分析报告: A Survey on 3D Gaussian Splatting

- **分析时间**: 2026-06-08T14:41:43.516683
- **作者**: Guikun Chen (The State Key Lab of Brain-Machine Intelligence, Zhejiang University, China), Wenguan Wang (The State Key Lab of Brain-Machine Intelligence, Zhejiang University, China)
- **年份**: 2026

## 摘要
3D Gaussian splatting (GS) has emerged as a transformative technique in radiance fields. Unlike mainstream implicit neural models, 3D GS uses millions of learnable 3D Gaussians for an explicit scene representation. Paired with a differentiable rendering algorithm, this approach achieves real-time rendering and unprecedented editability, making it a potential game-changer for 3D reconstruction and representation. In the present paper, we provide the first systematic overview of the recent developments and critical contributions in 3D GS. We begin with a detailed exploration of the underlying principles and the driving forces behind the emergence of 3D GS, laying the groundwork for understanding its significance. A focal point of our discussion is the practical applicability of 3D GS. By enabling unprecedented rendering speed, 3D GS opens up a plethora of applications, ranging from virtual reality to interactive media and beyond. This is complemented by a comparative analysis of leading 3D GS models, evaluated across various benchmark tasks to highlight their performance and practical utility. The survey concludes by identifying current challenges and suggesting potential avenues for future research. Through this survey, we aim to provide a valuable resource for both newcomers and seasoned researchers, fostering further exploration and advancement in explicit radiance field.

## 方法论

### method-001 3D Gaussian Splatting 研究分类体系 [系统设计]

该综述提出一个多层次的分层分类框架，将3D Gaussian Splatting（3DGS）的研究工作按照基础原理、优化方向和下游应用进行系统化组织。该体系首先界定隐式与显式辐射场的核心差异，然后将改进方法归纳为七个技术维度（如稀疏输入、内存效率、真实感增强等），并将应用划分为七个领域（如机器人、动态场景、生成与编辑等）。框架以树状结构呈现，并配合动态GitHub仓库长期维护，旨在帮助研究者快速定位相关工作和发现新趋势。

**创新点**:
- 首次提供面向3DGS研究的完整、多层级分类体系，覆盖从原理到应用的全链条。
- 将7类优化方法和7类应用场景进行交叉映射，揭示技术发展的宏观趋势。
- 维持两个动态GitHub仓库（组织仓库和性能基准仓库），为社区提供可更新的知识图谱和可复现的评测数据。

**输入**: 2019年后与三维高斯泼溅相关的学术论文、技术报告及项目源码。
**输出**: 结构化的研究分类树、各方向发展趋势总结、待解决问题清单及未来方向建议。

**步骤**:
从宏观视角划分3DGS研究的三大支柱：基础原理（渲染与优化）、优化方向（7类）和应用任务（7类）。
将优化方向细化为稀疏输入处理、内存效率提升、真实感渲染、优化算法改进、附加属性增强、混合表示和新的渲染算法七个子方向。
将应用领域归纳为机器人、动态场景重建、生成与编辑、数字人、医学内窥镜场景、大规模场景重建和物理模拟七个分支。
针对每个子方向或应用领域，进一步按方法类型（数据驱动、混合、基于物理等）进行分类，形成三级标签体系。
建立动态GitHub仓库，持续跟进新论文并维护相应分类条目与性能表格。

### method-002 多任务性能基准评估框架 [评估框架]

该综述构建了一个覆盖定位、静态场景渲染、动态场景渲染、人体头像重建和手术场景重建五类任务的标准化性能评测框架。在每个任务中，统一采用代表性数据集（如Replica、D-NeRF、ZJU-MoCap、EndoNeRF）和一致的评估指标（PSNR、SSIM、LPIPS、ATE等），收集并对比多种方法的量化结果。该框架还公开了所有代码库和超参数配置，以支持结果的复现与公平对比。

**创新点**:
- 首次在统一框架下对3DGS的定位、静态/动态渲染、头像和手术重建进行系统性的定量评估。
- 公开所有对比方法的代码库与超参数，推动领域内可复现的公平评测。
- 不仅关注常用渲染指标，还纳入推理速度和GPU内存占用等实际部署维度。

**输入**: 各方法在Replica、D-NeRF、ZJU-MoCap、EndoNeRF等标准数据集上的推理结果（RGB图像或轨迹位姿）。
**输出**: 五张基准对比表（Table S4–S8），包含精度、速度和内存数据；各任务的洞察总结；以及可交互的性能基准仓库。

**步骤**:
选取SLAM定位（Replica的8条序列）、静态渲染（Replica测试视图）、动态渲染（D-NeRF合成数据）、人体头像（ZJU-MoCap 6个主体）和手术场景（EndoNeRF立体视频）作为五项代表性任务。
统一规定各任务的评价指标：定位任务使用绝对轨迹误差RMSE，渲染类任务使用PSNR、SSIM和LPIPS。
从原始论文或统一环境下收集3DGS及基线方法（如NICE-SLAM、Point-SLAM、D-NeRF、NeuralBody、EndoSurf等）的性能数据。
汇总结果到对照表格（Table S4至S8），直观比较非GS方法与GS方法在精度、速度和资源占用上的差异。
为每项任务标注主要发现，例如3DGS方法在渲染质量上较NeRF基线提升数dB，渲染速度可提升百倍，同时显著降低GPU内存占用。
提供配套的基准仓库，包含数据加载脚本、评估代码和具体运行配置。

**关键公式**:

$$
\text{PSNR} = 10 \log_{10} \left( \frac{MAX_I^2}{MSE} \right)
$$

$$
\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

$$
\text{LPIPS} = \sum_l w_l \cdot \| \hat{y}^l - \hat{y}_0^l \|_2
$$

$$
\text{ATE}_{\text{RMSE}} = \sqrt{ \frac{1}{n} \sum_{i=1}^n \| \mathbf{p}_i - \mathbf{\hat{p}}_i \|^2 }
$$

## 核心观点

### claim-001 [设计] [→ method-001]
> **提出分层分类体系，系统化组织3DGS研究，实现宏观导航。**
- **问题**: 3DGS领域研究呈爆炸式增长，缺乏系统化的宏观框架来组织、关联和导航各个研究方向，导致研究者难以把握全局趋势和定位空白。
- **方法**: 提出一个多层次的分层分类体系，将3DGS研究按照基础原理、优化方向（7个维度）和下游应用（7个领域）进行树状组织，并维护动态更新的GitHub仓库。
- **机制**: 通过清晰的分层和交叉映射，研究者可以快速理解不同工作的定位、识别技术演化路径和未被覆盖的交叉区域，从而加速领域探索和避免重复；动态仓库则确保分类时效性。
- **结果**: 该分类体系帮助研究者从宏观视角把握3DGS全局，揭示技术发展趋势，并促进社区协作。
- **前提**: 分类框架需保持全面且合理；动态仓库需持续维护更新；如果研究出现新的范式突破现有维度，分类可能需要重构。
- **来源**: 第1节: INTRODUCTION
- **隐含假设**: 分类体系的维度能够完整覆盖3DGS研究的核心方向; GitHub仓库会被持续维护并吸纳新研究; 研究者会主动参考该分类体系作为导航工具
> 原文: [第1节] We provide the first systematic and comprehensive review that examines 3D GS from a macro-level perspective by establishing clear taxonomies and frameworks. ... Our organizational structure serves as a roadmap for understanding how different approaches relate to and build upon each other withi
  *置信度: 80%*

### claim-002 [对比] [→ method-002]
> **通过标准化多任务评测框架，公平对比方法，揭示3DGS在精度和速度上的优势。**
- **问题**: 3DGS方法的大量涌现导致缺乏标准化、可复现的性能比较，使得不同方法之间的效能对比困难，社区无法公平评估进展。
- **方法**: 构建一个覆盖定位、静态渲染、动态场景、人体头像和手术场景五个任务的标准化评测框架，采用代表性数据集和统一指标（PSNR、SSIM、LPIPS、ATE等），并公开所有对比方法的代码库和超参数配置。
- **机制**: 通过在相同条件下对多种方法进行定量评估，消除了实现细节差异，确保了对比的公平性；纳入渲染速度和GPU内存等实际部署指标，补充了传统质量指标；公开代码和配置使结果可复现，从而提供可靠、全面的性能画像。
- **结果**: 定量结果显示3DGS方法在定位精度（ATE提升约50%）、渲染质量（PSNR显著提升）和速度（最高578倍加速）上全面超越传统NeRF方法，同时公开的基准仓库促进了可复现研究。
- **前提**: 评测所采用的数据集和任务切分必须具有足够的代表性和广泛认可度；复现其他方法时需严格遵循原论文超参数，否则可能引入偏差。
- **来源**: 第6节: RGB
- **隐含假设**: 所选数据集和任务划分得到社区广泛接受; 所有对比方法的实现忠实于原论文且超参数正确配置; 性能差异主要由方法本身决定而非实现细节或硬件差异
> 原文: [第6节] We provide empirical evidence by presenting the performance of several 3D GS algorithms ... We link the original codebases and configurations in our benchmark repository to support reproducibility. ... As shown in Table S4, the recent 3D Gaussians based localization algorithms have a clear adv
  *置信度: 90%*

## 局限性分析

- **[泛化性]** (较高) 所提出的分层分类体系依赖预定义的七个优化维度和七个应用领域，这种静态结构对3DGS领域未来可能涌现的全新研究范式（如基于大模型的统一框架、神经场与高斯混合的新表示）缺乏适应能力。一旦出现突破现有维度的创新，整个分类框架将面临重构风险，从而削弱其长期导航价值。
  > 建议: 在分类体系中引入灵活的扩展机制，例如允许研究者为新兴方向动态添加子类别或标签，并定期基于社区反馈进行框架版本迭代，以确保分类的生命力。

- **[实验]** (较高) 标准化评测框架仅覆盖定位、静态渲染、动态场景、人体头像和手术场景五类任务，忽视了3DGS在自动驾驶、大规模城市场景以及具身智能等同样重要的下游任务。这导致性能对比的普适性不足，无法全面反映不同方法在多样化真实应用中的优劣。
  > 建议: 将评测任务扩展至更广泛的下游场景，至少纳入自动驾驶仿真和大型室外场景重建等代表性任务，并提供跨任务的综合排名与分析，增强评测的生态覆盖度。

- **[方法]** (中等) 分类体系完全依赖人工专家知识进行维度划分和论文归类，缺乏自动化的量化验证或一致性检验。随着论文数量的爆发，主观归类容易引入偏差和错配，且不同审阅者之间可能存在分类分歧，降低了框架的客观性和可复现性。
  > 建议: 设计半自动分类流程，结合关键词提取、引文网络分析和嵌入聚类技术，为人工归类提供辅助参考；同时公布归类准则和示例，让社区可参与校对，提升分类信度。

- **[实验]** (中等) 在横向对比中，所有方法的结果均以单次运行的指标报告，未提供多次实验的方差、置信区间或统计显著性检验（如Friedman检验）。缺乏此类统计支撑，所声称的性能提升可能源于随机波动而非方法本质优势，削弱了基准结论的可靠性。
  > 建议: 对所有关键指标进行至少3次独立运行并报告均值和标准差，采用非参数统计检验比较方法间的显著差异，并在结果表格中标注显著性标记。

- **[可复现性]** (中等) 评测基准虽然公开了代码库和超参数配置，但对于硬件环境（如GPU型号、驱动版本、CUDA版本）的记录不够详细，而3DGS方法的渲染速度和显存占用对硬件高度敏感。这导致不同实验室在“相同”配置下可能复现出差异较大的性能数值，影响基准的可复现性。
  > 建议: 在公开的基准仓库中引入硬件描述文件，明确记录每一项结果的运行环境（包括GPU型号、显存、驱动、CUDA、PyTorch版本等），并尽可能使用Docker镜像固化环境，最大化复现一致性。

- **[伦理]** (中等) 作为全面介绍3DGS的综述，全文未涉及该技术可能带来的伦理与社会风险，例如高真实感场景生成在深度伪造、虚假信息传播中的滥用，以及大规模3D重建对个人隐私的潜在侵犯。回避这些议题会降低综述对从业者和政策制定者的现实指导意义。
  > 建议: 增设专门的 “伦理与社会影响” 章节，系统梳理3DGS在隐私泄露、内容伪造、版权争议等方面的风险，并讨论可能的技术缓解措施和监管建议，提升综述的社会责任深度。

## 总结
本文作为2026年发表的3D Gaussian Splatting（3DGS）综述，提出了一个多层次分层分类体系和一个多任务标准化评测基准，试图为当时呈爆炸式增长且缺乏系统导航的研究领域提供全局框架和公平性能比较。分类体系按基础原理、七个优化维度和七个应用领域组织文献，并辅以动态GitHub仓库；评测基准则覆盖定位、静态/动态渲染、人体头像重建和手术场景五类任务，采用统一数据集和指标，并公开代码与配置。

关键发现包括：分类框架揭示了3DGS研究的技术演化路径与交叉空白，有助于研究者把握宏观趋势；标准化评测结果显示，3DGS方法在定位精度（ATE提升约50%）、渲染质量（PSNR显著提高）和渲染速度（最高可达传统NeRF的578倍）上全面超越NeRF类方法，公开的基准仓库促进了社区可复现研究。这些结果在当时为3DGS的优越性提供了量化证据，论证力度较强，但评测仅报告单次运行指标，缺乏方差和统计显著性检验，削弱了结论的稳健性。

整体而言，该综述在3DGS研究早期阶段起到了重要的导航与基准作用，其分类与评测框架具有开创性。但亦存在明显的时代局限性：分类体系依赖静态预定义维度，难以适应未来可能出现的全新范式（如基于大模型的统一框架）；评测任务未覆盖自动驾驶、大规模城市场景等重要领域，泛化性受限；人工归类主观性强且缺乏量化验证；硬件细节记录不足影响可复现性；全文未讨论深度伪造、隐私等伦理风险。这些不足提示后续工作需在动态更新机制、任务覆盖广度、统计严谨性及伦理审视方面加以完善。