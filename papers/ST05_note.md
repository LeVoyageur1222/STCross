# ST05: Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting

## 基本信息
- **会议/期刊**: ICLR 2026
- **年份**: 2026
- **作者**: Yifan Hu (清华大学), Jie Yang (UIC), Tian Zhou (阿里达摩院, 项目负责人), Peiyuan Liu (清华大学), Yujin Tang (Dartmouth College), Rong Jin (阿里达摩院), Liang Sun (阿里达摩院, 通讯作者)
- **关键词**: 时间序列预测、分布对齐、表示学习、重建辅助训练、互信息最大化、即插即用框架
- **代码**: https://github.com/TROUBADOUR000/TimeAlign

## 研究问题与动机

本文指出现有时间序列预测（TSF）模型在表示学习方面存在三个根本性缺陷：

1. **低频周期性捷径（Shortcut through Low-Frequency Periodicity）**：模型在纯误差驱动目标（MSE/MAE）下优化时，倾向于过度依赖低频周期分量。预测结果往往只是历史低频周期信号的重复，patch间的余弦相似度极高。而真实值中的高频分量（反映突变和非周期性波动）被严重低估，但这些高频信号对鲁棒预测不可或缺。

2. **过去与未来的分布失配（Distributional Mismatch between Past and Future）**：从历史序列提取的高维表示的分布与未来目标的分布存在显著差异。嵌入倾向于保留过去的统计特征而非未来的特征，导致将历史嵌入直接映射到目标分布变得困难。

3. **单向范式的结构性缺陷（Structural Flaw of Unidirectional Paradigm）**：主流预测范式是单向的（历史→预测），这种结构限制了模型保留目标关键信息的能力。随着表示逐层编码，过程类似于频率平滑器，丢弃了精细的频率结构（这些细节通常编码了由时变外部事件触发的突变信息）。

**核心问题**：能否设计一个预测框架，弥合过去与未来之间的鸿沟，在缓解分布偏移的同时忠实捕捉多频率动态？

## 核心贡献

1. **范式转变（Paradigm Shift）**：重新审视当前主流预测范式中的表示学习，揭示其过度强调低频周期性、分布失配和丢失精细分量的关键局限。提出融合"预测-重建-对齐"的新范式。

2. **技术贡献（TimeAlign框架）**：提出双分支框架，包含预测分支和重建分支，通过全局和局部对齐机制对两者的潜在表示进行分布感知对齐，有效弥合过去与未来。

3. **理论与实践统一**：提供两个理论保证：(i) 重建改善预测泛化能力；(ii) 对齐增强预测表示与未来目标之间的互信息。在8个基准数据集上验证了SOTA性能。

## 方法概述

TimeAlign 是一个轻量级、即插即用的框架，核心思想是通过辅助重建任务来对齐预测表示和未来目标分布。

### 整体架构（四个组件）

```
历史输入 X → [Predict Branch] → Ŷ_pred (训练+推理)
                                      ↕ Distribution-Aware Alignment
未来输入 Y → [Reconstruct Branch] → Ŷ_recon (仅训练)
```

总损失函数：
$$\mathcal{L} = \mathcal{L}_{pred} + \mathcal{L}_{recon} + \lambda \mathcal{L}_{align}$$

### Predict Branch（预测分支）
- 训练和推理阶段均激活
- 骨干网络可替换（plug-and-play），默认使用轻量级Linear编码器
- 输入 $X \in \mathbb{R}^{C \times L}$ 经Patching和线性投影得到 $X_p \in \mathbb{R}^{(C \cdot N) \times D}$
- 经过 $M$ 层编码器（Linear-Activation-Linear + 残差连接）：
  $$H_x^l = H_x^{l-1} + \text{Linear}(\sigma(\text{Linear}(H_x^{l-1}))), \quad l = 1, \ldots, M$$
- 预测头输出 $\hat{Y}_{pred} = \text{Linear}(H_x^M)$

### Reconstruct Branch（重建分支）
- **仅在训练阶段激活**，推理时丢弃
- 对未来序列 $Y \in \mathbb{R}^{C \times T}$ 进行自身重建，捕获与目标分布对齐的紧凑表示
- 结构与Predict Branch相同（共享patch数量 $N$ 和编码器层数 $M$）
- 重建表示 $H_y$ 作为对齐的参考分布

**关键洞察**：重建嵌入天然与目标分布对齐（因为是从自身恢复自身），且重建鼓励关注频谱细节，同时保留粗粒度周期结构和细粒度高频变化。

## 关键技术细节

### Distribution-Aware Alignment（分布感知对齐）

#### 额外映射层
对齐前对预测分支施加额外的线性映射：
$$\tilde{H}_x^i = \text{Linear}(H_x^i) \in \mathbb{R}^{(C \cdot N) \times D}$$
- 作用：时序重定位和归一化，使预测分支特征与重建分支更兼容
- **非对称梯度流**：对齐损失的梯度仅通过此Linear层传播到预测分支，重建分支stop-gradient，确保重建分支提供稳定的监督信号

#### Local Alignment（局部对齐）
保证patch级别的细粒度语义一致性，对捕捉突变和高频细节至关重要：
$$\mathcal{L}_{local}^i = \frac{1}{n^2} \sum_{j=1}^{n} \text{GELU}\left(1 - |\tilde{h}_{x,j}^i \cdot h_{y,j}^i| - \delta_{loc}\right)$$
其中 $n = C \cdot N$，$\delta_{loc}$ 为松弛margin。本质上度量频谱能量投影，利用多频率信息。

#### Global Alignment（全局对齐）
使特征内部的相对距离矩阵尽可能相似，确保大尺度时序依赖和低频动态也被对齐：
$$\mathcal{L}_{global}^i = \frac{1}{n^2} \sum_{j=1}^{n} \text{GELU}\left(\|\tilde{h}_{x,j}^i (\tilde{h}_{x,j}^i)^\top - h_{y,j}^i (h_{y,j}^i)^\top\|_1 - \delta_{glo}\right)$$

#### Weight-Based Dynamic Loss（自适应权重动态损失）
$$\mathcal{L}_{align} = \frac{1}{M} \sum_{i=1}^{M} (\alpha \mathcal{L}_{local}^i + \beta \mathcal{L}_{global}^i)$$
自适应权重防止任一目标主导：
$$\alpha = \frac{\mathcal{L}_{local} + \mathcal{L}_{global}}{\mathcal{L}_{local}}, \quad \beta = \frac{\mathcal{L}_{local} + \mathcal{L}_{global}}{\mathcal{L}_{global}}$$

### 理论分析

#### 定理1：重建改善泛化
设 $M^*$ 为最优预测解，$M_1$ 为直接从训练样本得到的经验估计器，$M_2$ 为通过重建误差最小化导出一维表示后再学习的估计器。在集中界条件下：
$$\frac{|M_2 - M^*|_F}{|M^*|_F} \ll \frac{|M_1 - M^*|_F}{|M^*|_F}$$
即重建引导的估计器比直接经验估计器有更紧的近似界。（证明基于矩阵集中不等式和Davis-Kahan $\sin\theta$ 定理）

#### 定理2：TimeAlign是隐式互信息增强器
- **预测损失**提供互信息 $I(Y; H_x)$ 的变分下界：$I(Y; H_x) \geq -\mathcal{L}_{pred}$
- **对齐损失**基于InfoNCE推导互补下界：$I(Y; H_x) \geq -\mathcal{L}_{align}$，其中密度比 $f(y, h_x) = \exp(\text{proj}(h_x)^\top \cdot G_\phi(y))$
- 联合优化 $\mathcal{L}_{pred}$ 和 $\mathcal{L}_{align}$ 等价于最大化预测表示与未来目标之间的互信息

## 实验

### 数据集
8个广泛使用的真实世界数据集：
- **ETT系列**：ETTh1, ETTh2, ETTm1, ETTm2（电力变压器温度，7通道）
- **Weather**：德国21个气象指标，10分钟采样
- **Electricity**：321个客户的小时用电量
- **Traffic**：旧金山湾区862个检测器的小时道路占用率
- **Solar-Energy**：阿拉巴马州137个光伏电站的10分钟太阳能产量
- 额外鲁棒性实验：NASDAQ和NYSE股票市场数据

### 基线方法
TimeMixer (2024), iTransformer (2024), PatchTST (2023), Crossformer (2023), TVNet (2025), ModernTCN (2024), TimesNet (2023), DLinear (2023), CMoS (2025), TimeBase (2025), Leddam (2024)

### 主要结果
- **长期预测**：在大多数数据集上取得最优性能。相比次优方法TVNet，MSE/MAE平均降低3.27%/5.20%，Wilcoxon检验p值为 $1.37 \times 10^{-8}$，在99%置信水平下显著
- **分布偏移严重的ETTm1/ETTm2数据集上优势尤为突出**
- **即插即用能力**：作为插件 augment iTransformer和DLinear，在大多数基准上一致提升1%-4%，同时提高预测与真值的余弦相似度
- **效率**：纯Linear架构，GPU显存和速度优于所有Transformer/CNN方法；在Traffic数据集上比vanilla iTransformer提前3000次迭代达到相同验证损失
- **消融实验**：Local+Global组合效果最佳；单独Global优于单独Local（粗粒度分布级拉伸更有效）；无对齐的基线本身已是SOTA级别，证明TimeAlign不是仅在弱基线上有效的trick
- **高频分析**：引入HF-MSE指标，验证TimeAlign在高频分量上的误差显著低于无对齐版本
- **股票市场鲁棒性**：在极端非平稳的NASDAQ/NYSE数据上一致优于所有基线

### 超参数
- 优化器：Adam，学习率从 $\{10^{-3}, 5 \times 10^{-4}, 10^{-4}\}$ 中选取
- Batch size：Electricity/Traffic为16，其余为32
- Look-back长度从 $\{96, 192, 336, 512, 720\}$ 中搜索
- 编码器层数：2层；所有实验在单张NVIDIA V100 32GB GPU上进行

## 与本项目的关联

本项目构建面向节假日/极端天气场景的异常客流预测系统，与本文有以下关键关联：

1. **分布偏移问题的直接对应**：节假日和极端天气场景下的客流模式与正常时段存在显著的分布偏移——这正是TimeAlign要解决的核心问题。历史正常客流的表示分布与异常场景下的未来目标分布差异巨大，TimeAlign的分布对齐机制可直接用于缓解这一挑战。

2. **高频突变信息的捕捉**：异常客流往往表现为突发的高频波动（如节假日前的客流骤增、极端天气导致的突然下降），而传统模型倾向于平滑掉这些高频信号。TimeAlign通过重建分支保留高频细节的能力，对本项目捕捉异常客流的突变特征具有重要参考价值。

3. **即插即用设计的工程价值**：TimeAlign可作为插件集成到任意预测骨干网络中，这意味着我们可以在现有的客流预测模型上直接叠加该模块，无需重构整个系统，降低了工程落地成本。

4. **重建分支作为"未来先验"**：在异常场景中，如果有部分异常时段的历史数据可用于重建分支的训练，则重建分支可以学习到异常场景的目标分布特征，为预测分支提供有效的分布引导。

5. **局限性注意**：论文指出在含有极端离群值和零填充值的数据集（如Weather、Solar）上，对齐可能失效甚至产生负面影响。客流数据中若存在类似的异常值或数据缺失问题，需要对齐模块做适配处理。

## 关键启发

1. **"预测-重建-对齐"三元范式**：不仅适用于通用TSF，对于任何存在训练/测试分布偏移的预测任务都有借鉴意义。在异常客流预测中，可以用异常时段数据构建重建分支，显式引导模型学习目标分布。

2. **重建不仅是预训练手段**：本文证明了重建损失最小化在理论上能获得比直接经验估计更优的泛化界，这为在异常客流系统中引入重建辅助训练提供了理论支撑。

3. **非对称梯度流设计**：重建分支stop-gradient是一个简洁而有效的设计，确保辅助分支提供稳定参考而不被主任务干扰。在多任务学习的异常预测系统中可采用类似策略。

4. **自适应权重避免手动调参**：Weight-Based Dynamic Loss根据当前loss值自动调整Local/Global对齐权重，减少了在不同场景（正常/节假日/极端天气）下手动调参的负担。

5. **高频保真度的重要性**：异常事件的本质就是高频信号。在设计异常客流预测系统时，应特别关注模型对高频分量的保留能力，可考虑引入类似HF-MSE的评估指标来监控模型对突变模式的捕捉效果。

6. **轻量化设计的启示**：TimeAlign仅用2层Linear编码器就达到SOTA，说明在分布对齐做得好的前提下，模型不一定需要很复杂。这对实时性要求高的客流预测系统有重要参考意义。
