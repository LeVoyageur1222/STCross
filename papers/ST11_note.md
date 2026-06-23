# ST11: Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting

## 基本信息
- **会议/期刊**: KDD '26 (The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining)
- **年份**: 2026
- **作者**: Hongjun Wang, Jiawei Yong, Jiawei Wang, Shintaro Fukushima, Renhe Jiang*（东京大学 & 丰田汽车）
- **关键词**: Spatio-temporal forecasting, Traffic accident, Urban computing, Conditional Transformer, Guided Layer Normalization
- **代码**: https://github.com/Dreamzz5/ConFormer

## 研究问题与动机

### 核心问题
交通预测在正常模式下已有较好表现，但**交通事故等突发事件导致的异常交通模式**仍是关键挑战。现有模型在事故发生时性能显著下降。

### 动机分析
1. **正常 vs 异常的根本矛盾**：正常交通遵循可预测的时间规律（如早晚高峰），现有模型（如 DCRNN）可以很好捕获；但事故造成的扰动以非线性方式在网络中传播，导致速度骤降和复杂变化，当前时空方法无法有效建模。
2. **事故数据覆盖不足**：大多数交通数据集缺乏详细的事故信息，仅有基础流量数据，阻碍了系统研究事故和管制对交通模式的影响。
3. **事故建模能力不足**：现有方法要么完全忽略事故因素，要么仅将其作为外部特征简单拼接，未能考虑事故如何改变空间关系、时间依赖和数据分布。
4. **实际影响量化**：研究表明事故可使出行时间增加 37%-43%（FHWA 数据），高速公路事故是非经常性拥堵的主要原因。

> **与本项目的直接关联**：本项目关注节假日/极端天气下的异常客流预测，与本文关注的事故场景同属"异常事件驱动的时空预测"范畴，核心矛盾一致——常规模型擅长规律性模式，但在外部事件冲击下失效。

## 核心贡献

1. **ConFormer 模型**：提出条件 Transformer 架构，采用 **Guided Layer Normalization (GLN)** 机制，根据交通条件动态调整归一化参数，自适应捕获时空动态并感知事故信息。
2. **两个大规模数据集**：
   - **Tokyo 高速公路数据集**：1,843 条路段，含 JARTIC 权威事故与管制数据（10 分钟粒度）
   - **California 数据集**：Bay Area（2,352 传感器）+ San Diego（716 传感器），整合 US Accidents 数据库的事故标注
3. **SOTA 性能**：在所有数据集上超越 STAEFormer 等主流方法，同时参数量更少（~684K-784K vs STAEFormer 的 1.7M-18M）、计算开销更低。

## 方法概述

ConFormer 的整体架构包含两条并行分支：

```
输入 X → 嵌入层(含事故/管制/时间嵌入) → ┬→ 条件分支: Graph Propagation → MLP → [γ, β, α]
                                          └→ Transformer 分支: GLN + Condition Self-Attention + 条件残差连接
```

### 核心思想
将事故/管制等外部事件信息通过**图传播**生成上下文条件表示 $X^c$，再用该条件表示生成**动态仿射变换参数** $(\gamma, \beta, \alpha)$，分别控制：
- $\gamma, \beta$：LayerNorm 的动态缩放和平移（替代传统固定的 $\gamma, \beta$）
- $\alpha$：残差连接的调制因子（类似课程学习，逐步引入条件信号）

## 关键技术细节

### 1. 数据嵌入与条件注入
$$X^e = \text{MLP}(X^{data} \| X^{acc} \| X^{reg} \| X^{dow} \| X^{tod} \| X^{stae})$$

其中 $X^{acc} \in \mathbb{R}^{T \times N \times D_{acc}}$ 和 $X^{reg} \in \mathbb{R}^{T \times N \times D_{reg}}$ 分别为事故和管制的指示特征嵌入。

### 2. 图传播（Graph Propagation）
简化 GCN，仅保留图传播部分（去掉特征混合矩阵 $W$），用于建模事故影响的传播效应：
$$X^c = \text{GraphPropagation}(X^e, A) = [X^e \| \tilde{L}X^e \| \tilde{L}^2 X^e \| \dots \| \tilde{L}^K X^e]$$

其中 $\tilde{L}$ 为归一化图拉普拉斯矩阵，$K$ 为切比雪夫多项式阶数（实验中 $k=1\text{-}2$ 即可）。

### 3. Guided Layer Normalization (GLN) ⭐核心创新
传统 LayerNorm：
$$\text{LN}(X) = \gamma \cdot \frac{X - \mu}{\sigma} + \beta$$
其中 $\gamma, \beta$ 是**固定的**可学习参数。

GLN 将 $\gamma, \beta$ 替换为从条件表示 $X^c$ 动态生成的参数：
$$\gamma, \beta = \text{MLP}(X^c)$$
$$\text{GLN}(X, \gamma, \beta) = \gamma \cdot \frac{X - \mu}{\sigma} + \beta$$

**理论解释**：GLN 等价于对 self-attention 中的 Q, K 进行条件化的仿射变换：
$$\text{Softmax}\left(\frac{\gamma^2 Q'K'^T + \gamma K'\beta^T + \gamma Q'\beta^T + \beta^2}{\sqrt{d}}\right)$$

- **$\gamma$（缩放因子）**：高值 → 增强对突发变化的敏感度，快速适应事故；低值 → 抑制局部波动，保持常规稳定性
- **$\beta$（平移因子）**：高值 → 强调节点特异性特征，增强事故响应中的局部表示；低值 → 保持全局一致性

### 4. Condition Self-Attention
将归一化后的输入与条件表示拼接后计算 Q, K, V：
$$Q = X^{(N)} W_Q, \quad K = (X^{(N)} \| X^c) W_K, \quad V = (X^{(N)} \| X^c) W_V$$

使注意力机制同时感知原始特征和事故条件信息。

### 5. 条件残差连接
$$X^{(Res)} = X^{(in)} + \alpha \cdot X^{(out)}$$

其中 $\alpha = \text{MLP}(X^c)$，零初始化策略确保训练初期模型先学习基本时空模式，再逐步融入条件信号（类似课程学习）。

### 6. 计算复杂度
$$\text{FLOPs} = K|\mathcal{E}|D + (TN^2D + NT^2D) + NTD^2$$

仅需单层注意力即可达到优越性能，总参数量远低于 STAEFormer。

## 实验

### 数据集

| 数据集 | 时间范围 | 粒度 | 空间单元数 | 事故来源 | 管制数据 |
|--------|---------|------|-----------|---------|---------|
| Tokyo | 2021/10-2021/12 | 10 min | 1,843 路段 | JARTIC | ✅ |
| San Diego | 2019/1-2019/12 | 15 min | 716 传感器 | US Accidents | ❌ |
| Bay Area | 2019/1-2019/12 | 15 min | 2,352 传感器 | US Accidents | ❌ |

另外在传统 benchmark PEMS03/04/07/08 上也进行了验证（无事故信息）。

### 基线方法
- **时序模型**: LSTM
- **GNN-based**: DCRNN, AGCRN, STGCN, GWNet, DSTAGNN, DGCRN, D²STGNN, STGODE
- **Attention/Transformer**: ASTGCN, STTN, PDFormer, STAEFormer
- **其他**: STID, Historical Inertia

### 主要结果

**含事故信息的数据集（Table 2）**：
- **Tokyo**: ConFormer 平均 MAE=6.37, RMSE=10.22, MAPE=23.13%，相比最佳基线 STAEFormer (MAE=6.56) 提升 ~2.9% MAE，MAPE 降低超 20%
- **San Diego**: ConFormer 平均 MAE=16.75, RMSE=28.59, MAPE=10.88%，全面优于所有基线
- **Bay Area**: ConFormer 平均 MAE=19.82, RMSE=33.00, MAPE=15.24%，在大规模图上仍保持 SOTA

**传统 Benchmark（Table 3，无事故信息）**：
- 在 PEMS03/04/07/08 上同样取得最优或接近最优的结果，证明即使没有显式事故标注，GLN 也能通过交通流信号作为代理变量隐式捕获异常模式

**效率对比（Figure 3）**：
- ConFormer 参数量仅 684K-784K，而 STAEFormer 需 1.7M-18M，DSTAGNN 需 3.9M-26.9M
- 在 FLOPs-MAE 权衡上显著优于其他模型

### 消融实验关键发现（Table 4）
1. **GLN vs LN**：替换为标准 LayerNorm 后，事故场景 MAE 从 8.98 升至 9.58（Tokyo），验证动态归一化的必要性
2. **Attention vs MLP**：将注意力替换为 MLP 导致更大退化（MAE 升至 10.06）
3. **各调制因子的作用**：
   - $\alpha$ 主要影响优化稳定性
   - $\beta$ 增强对突发扰动的敏感度（移除后事故场景 MAE 从 22.36 升至 23.68）
   - $\gamma$ 区分正常/事故场景的节点特异性（移除后 Bay Area 事故 MAE 从 26.95 升至 28.33）
4. **事故/管制数据的重要性**：移除事故信息后 Tokyo 事故场景 MAE 从 8.98 升至 10.97；移除管制数据后管制场景 MAE 从 9.18 升至 10.64
5. **GLN 的可迁移性**：将 GLN 应用于 GWNet（GLN+GWNet），三个数据集上 MAE 平均降低 2.63%

### 案例研究
- **事故场景**：t-SNE 可视化显示 GLN 能在潜在空间中有效区分正常/异常交通模式的特征分布
- **管制场景**：图传播使中心节点与邻居节点的注意力分布在管制实施后高度一致，验证了条件信息的有效传播
- **预测曲线**：ConFormer 准确捕获了 12:00 和 09:00 的速度骤降，显著优于对比模型

## 与本项目的关联

### 高度相关的方面
1. **问题定义一致**：本项目关注节假日/极端天气下的异常客流预测，与本文的事故场景本质相同——都是**外部事件驱动的非常规时空模式预测**。可将节假日/极端天气类比为"事故"，作为条件信号注入模型。
2. **GLN 机制的直接借鉴价值**：Guided Layer Normalization 是一种通用的条件化归一化方法，不局限于交通事故。对于本项目，可以将节假日类型、天气等级、大型活动信息等作为条件特征，通过 GLN 动态调整模型内部表示。**这是一个轻量级且高效的改进点**。
3. **图传播建模事件扩散**：事故影响通过路网传播，类似地，节假日客流冲击会通过地铁/公交网络传播。Graph Propagation 模块可直接用于建模异常客流的网络扩散效应。
4. **数据集构建思路**：本文整合多源异构数据（交通流 + 事故 + 管制）的方式，对本项目整合客流数据 + 节假日日历 + 气象数据有直接参考价值。
5. **条件残差连接的课程学习思想**：先学常规模式再逐步融入异常条件信号的渐进式学习策略，对处理异常样本稀缺问题很有启发。

### 需要注意的差异
1. **数据类型**：本文为道路交通流（连续值），本项目为客流量（离散计数），损失函数和数据预处理需调整
2. **事件性质**：交通事故是局部、突发的；节假日/天气影响通常是全局性、可预见的，条件特征的构造方式需相应调整
3. **图的规模**：本文处理数百到数千节点，本项目地铁站点规模可能更小但拓扑更复杂

## 关键启发

1. **"条件化"是处理异常场景的高效范式**：不需要为异常场景设计单独的模型，而是通过条件机制让同一模型自适应切换行为模式。GLN 的实现非常轻量（仅额外一个 MLP），但效果显著。
2. **归一化层的动态化是被低估的改进方向**：传统深度学习模型的 LayerNorm/BatchNorm 参数是全局固定的，但在时空数据中不同节点、不同时刻的统计特性差异巨大。GLN 证明了动态归一化的有效性，这一思路可推广到其他时空模型中。
3. **简化设计反而更有效**：ConFormer 仅用单层注意力 + 简化的图传播（去掉特征混合），就超越了复杂的 STAEFormer。说明在条件信息充分的情况下，模型不需要过深的架构。
4. **事件信息的显式注入至关重要**：消融实验清楚表明，即使在模型结构不变的情况下，有无事故数据的性能差距可达 20%+。这强调了在本项目中收集和整合节假日/天气等外部事件数据的重要性。
5. **GLN 具有良好的可插拔性**：论文验证了 GLN 可以直接应用到 GWNet 等其他模型上并取得提升，说明它是一种通用的增强组件，值得在本项目的基线模型中尝试。
