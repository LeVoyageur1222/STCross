# ST07: HyperD: Hybrid Periodicity Decoupling Framework for Traffic Forecasting

## 基本信息
- **会议/期刊**: AAAI-26 (The Fortieth AAAI Conference on Artificial Intelligence)
- **年份**: 2026
- **作者**: Minlan Shao, Zijian Zhang*, Yili Wang, Yiwei Dai, Xu Shen, Xin Wang* (吉林大学人工智能学院 & 计算机科学与技术学院)
- **关键词**: 交通预测、周期性解耦、可学习周期嵌入、频域建模、时空注意力、异常波动
- **代码**: https://github.com/ll121202/HyperD

## 研究问题与动机

交通预测面临两大核心挑战：

1. **复杂的空间依赖性**：路网中不同路段和传感器之间存在动态交互，空间依赖关系随时间变化。
2. **多尺度周期性模式与不规则波动的共存**：受人类日常作息驱动，交通流呈现显著的日周期和周周期模式；但同时，事故、恶劣天气等不可预测事件会引发不规则的高频波动。

现有方法的局限性：
- **隐式周期建模**（如趋势-季节分解方法）：周期信息被分散到趋势分量和季节分量中，忽略了跨时间尺度和空间位置的统一周期结构，无法直接建模连贯的周期模式。
- **不精确的分解**：传统分解方法不可学习、效率低，导致趋势与季节分量对齐不佳，难以捕获复杂时序模式。
- **显式周期建模的不足**（如CycleNet）：仅考虑单一周期尺度，且未能建模空间位置与时序动态之间的交互，难以应用于交通预测场景。

**核心动机**：需要一个灵活的框架，能够同时捕获多尺度周期模式和细粒度的时空交互，并有效分离周期性成分与非周期性不规则波动。

## 核心贡献

1. **提出HyperD框架**：将交通数据解耦为**周期分量（Periodic）**和**残差分量（Residual）**，分别用定制化机制建模，解决多尺度周期模式与不规则波动并存的问题。
2. **混合周期表示模块（Hybrid Periodic Representation Module）**：利用可学习的日/周嵌入（Learnable Daily/Weekly Embeddings）和时空注意力编码器（STAE），显式捕获多尺度周期模式。
3. **频率感知残差表示模块（Frequency-Aware Residual Representation Module）**：通过复数域MLP在频域建模非周期高频波动，补充周期模式的不足。
4. **双视图对齐损失（Dual-View Alignment Loss, DVA）**：通过对齐低频信息与周期分支、高频信息与残差分支，确保两个分量的语义分离，防止信息冗余。
5. **全面的实验验证**：在4个真实数据集上达到SOTA，且在扰动下表现出优越的鲁棒性，计算效率显著优于已有解耦方法。

## 方法概述

### 问题定义

交通网络表示为有向图 $\mathcal{G} = \{\mathcal{V}, \mathcal{E}, \mathbf{A}\}$，其中 $\mathcal{V}$ 为 $N$ 个传感器节点，$\mathbf{A} \in \mathbb{R}^{N \times N}$ 为邻接矩阵。给定历史交通时序 $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_{T_1}\} \in \mathbb{R}^{T_1 \times N}$，预测未来 $T_2$ 步的交通状态 $\hat{\mathbf{Y}} \in \mathbb{R}^{T_2 \times N}$：

$$\mathbf{X}, \mathcal{G} \xrightarrow{f_\theta} \hat{\mathbf{Y}}$$

### 整体架构

HyperD包含三大组件：
1. **混合周期表示模块** → 生成周期预测 $\mathbf{S}^{out}$
2. **频率感知残差表示模块** → 生成残差预测 $\mathbf{R}^{out}$
3. **双视图对齐损失** → 训练时约束两分支的频率语义分离

最终预测：$\hat{\mathbf{Y}} = \mathbf{S}^{out} + \mathbf{R}^{out}$

## 关键技术细节

### 1. 混合周期表示模块（Hybrid Periodic Representation Module）

#### 1.1 可学习日/周嵌入（LDE / LWE）

- 引入两个可学习嵌入矩阵：$\mathbf{P}_D \in \mathbb{R}^{L_D \times N}$（日周期）和 $\mathbf{P}_W \in \mathbb{R}^{L_W \times N}$（周周期）
- 以5分钟采样率为例：$L_D = 288$（一天的时间步数），$L_W = 2016$（一周的时间步数）
- **统计先验初始化**：对训练数据归一化后，计算每个节点在每个时间步的均值作为嵌入初始值，加速收敛并融入时序先验知识
- 该设计可扩展至其他周期尺度（小时级、月级等）

#### 1.2 时空注意力编码器（STAE）

对周期嵌入 $\mathbf{P} \in \mathbb{R}^{L \times N}$ 进行精炼：

**Step 1 - GCN编码空间相关性**：
$$\mathbf{H} = \sigma(\hat{\mathbf{A}} \mathbf{P} \mathbf{W})$$
其中 $\hat{\mathbf{A}}$ 为归一化拉普拉斯矩阵。

**Step 2 - 时序自注意力**（每个时间步作为query）：
$$\mathbf{H}_t = \text{Attention}_t(\mathbf{H}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

**Step 3 - 空间自注意力**（每个节点作为query，对 $\mathbf{H}^\top$ 操作）：
$$\mathbf{H}_s = \text{Attention}_s(\mathbf{H}^\top)^\top$$

**Step 4 - 融合**：
$$\tilde{\mathbf{P}} = \mathbf{W}_o [\mathbf{H}_t ; \mathbf{H}_s]$$

分别对日和周嵌入独立执行上述操作，得到精炼后的 $\tilde{\mathbf{P}}_D$ 和 $\tilde{\mathbf{P}}_W$。

#### 1.3 混合周期模式构建

根据时间元数据（一天中的时刻 + 一周中的日期）从嵌入中检索对应片段：

- 历史时段：$\mathbf{S}^{in}_D(t) = \tilde{\mathbf{P}}_D[i^{in}_D(t), :]$，$\mathbf{S}^{in}_W(t) = \tilde{\mathbf{P}}_W[i^{in}_W(t), :]$
- 混合：$\mathbf{S}^{in} = \mathbf{S}^{in}_D + \mathbf{S}^{in}_W$
- 预测时段同理得到 $\mathbf{S}^{out}$

### 2. 频率感知残差表示模块（Frequency-Aware Residual Representation Module）

#### 2.1 残差计算

$$\mathbf{R}^{in} = \mathbf{X} - \mathbf{S}^{in}$$

#### 2.2 时空频率编码器（STFE）

将残差投影到高维空间 $\tilde{\mathbf{R}}^{in} \in \mathbb{R}^{T_1 \times N \times D}$ 后：

**空间维度频域处理**：
1. FFT：$\mathbf{R}^s_f = \text{FFT}_s(\tilde{\mathbf{R}}^{in})$
2. 复数域MLP（C-MLP）：在频域中用两组可学习权重 $(\mathbf{W}^r, \mathbf{W}^i)$ 分别处理实部和虚部
3. IFFT：$\mathbf{R}_s = \text{IFFT}_s(\mathbf{R}_2)$

**时间维度频域处理**：同样执行 FFT → C-MLP → IFFT 流程。

**输出**：$\mathbf{R}^{out} = \text{Proj}(\mathbf{R}_t + \tilde{\mathbf{R}}^{in})$（含残差连接）

> **关键洞察**：频域中的C-MLP等价于时域中的卷积操作，但能实现更高效的全局建模，且通过聚焦关键频率分量降低计算开销。

### 3. 双视图对齐损失（Dual-View Alignment Loss）

对预测 $\hat{\mathbf{Y}}$ 做FFT后，按阈值 $F_{low}$ 分割频谱：

$$\hat{\mathbf{Y}}^{low}_f = \hat{\mathbf{Y}}_f[0:F_{low}], \quad \hat{\mathbf{Y}}^{high}_f = \hat{\mathbf{Y}}_f[F_{low}:]$$

对齐损失：
$$\mathcal{L}_{dva} = \text{MSE}(\text{IFFT}(\hat{\mathbf{Y}}^{low}_f), \mathbf{S}^{out}) + \text{MSE}(\text{IFFT}(\hat{\mathbf{Y}}^{high}_f), \mathbf{R}^{out})$$

总损失：
$$\mathcal{L} = \mathcal{L}_{pred} + \alpha \cdot \mathcal{L}_{dva}$$

> **设计意图**：强制周期分支专注低频信息、残差分支专注高频信息，避免两分支语义重叠。

## 实验

- **数据集**: PEMS03, PEMS04, PEMS07, PEMS08（四个经典交通流数据集）
- **实验设置**: 训练/验证/测试 = 6:2:2；输入12步（1小时），预测12步（1小时）；所有结果取5次重复的平均值
- **基线方法**:
  - 时空预测方法：STGCN, DCRNN, GWNet, ASTGCN, MTGNN, STGODE, ST-WA, DGCRN, STPGNN
  - 时空解耦方法：D2STGNN, STWave, CycleNet-D, CycleNet-W, STDN
- **主要结果**:
  - HyperD在所有4个数据集上全面超越所有基线方法
  - 相比CycleNet-D/CycleNet-W，平均MAE分别降低22.63%和23.27%，验证了多尺度周期建模+时空交互的优势
  - 解耦方法整体优于非解耦方法，说明分量分离建模的有效性
- **消融实验**（PEMS04/PEMS08）:
  - 去除LWE比去除LDE性能下降更大 → 周周期包含工作日/周末动态，信息更丰富
  - 去除STFE比去除STAE性能下降更大 → 残差的时空建模比周期模式的时空建模更关键
  - C-MLP替换为普通MLP（Re-MLP）也有明显下降 → 频域建模不可替代
  - 去除DVA损失也有下降 → 频率对齐约束对解耦质量有实质贡献
- **初始化策略对比**: 统计先验初始化优于Zero/Uniform/Normal/Xavier/He等所有常规初始化
- **鲁棒性测试**（模拟三种扰动：突增×1.5、中断置零、片段打乱）:
  - HyperD在所有扰动类型下保持最低的性能降幅，鲁棒性显著优于D2STGNN、STWave、STDN
- **计算效率**（PEMS04, batch=64）:
  - 最大显存仅1.64GB（vs D2STGNN 17.90GB，降低6.27倍）
  - 每epoch训练时间7.31s（vs 57.29s，降低6.79倍）
  - CPU推理时间1.49s（vs 8.00s，降低4.77倍）
- **嵌入可视化**: HyperD学到的日/周嵌入比CycleNet更加展开和精细，捕获了更清晰的周期模式

## 与本项目的关联

本项目目标是构建**节假日/极端天气场景下的异常客流预测系统**。HyperD的核心思想与本需求高度相关：

1. **周期性解耦思路直接适用**：节假日和极端天气恰好属于论文所定义的"残差分量"——偏离常规日/周周期的不规则波动。HyperD将这类异常从周期模式中显式分离的思路，可为我们的异常检测提供结构化支撑。
2. **频率域分析视角**：DVA损失通过频域分割区分周期与异常，这一思路可用于识别"何时发生了异常"——当高频能量突增时即可能对应突发事件。
3. **鲁棒性验证方法值得借鉴**：论文中的三种扰动模拟（突增、中断、打乱）可作为我们评估模型在异常场景下鲁棒性的参考范式。
4. **局限性提醒**：论文的异常主要是短时、局部的扰动，而节假日/极端天气可能造成**持续性、全局性**的模式偏移，可能需要额外的机制（如外部特征注入、迁移学习）来增强。
5. **计算效率优势**：HyperD极低的资源消耗使其适合作为在线预测系统的基座模型，为异常检测模块留出充足的计算预算。

## 关键启发

1. **显式 > 隐式周期建模**：用可学习嵌入直接表示周期模式，比通过趋势-季节分解间接获取周期信息更有效。这启示我们在处理节假日周期时，也应考虑构建专门的"节假日嵌入"而非依赖通用分解。
2. **统计先验初始化的价值**：用历史均值初始化周期嵌入是一个简单但有效的技巧，在我们的系统中可以用历史同期（如去年同一节假日）的数据初始化异常模式嵌入。
3. **频域C-MLP的高效性**：在频域用复数MLP替代时域卷积，既降低了计算量又实现了全局感受野。对于大规模路网客流预测，这种效率优势尤为重要。
4. **解耦需要约束**：单纯的双分支结构不够，必须通过DVA等损失函数强制语义分离，否则两分支会趋于冗余。这对我们设计"正常客流+异常增量"的双分支预测架构有直接指导意义。
5. **残差分支更重要**：消融实验表明残差分支的时空建模对性能的贡献大于周期分支。这意味着在异常客流预测中，对异常/残差分量的精细建模可能是性能的关键瓶颈。
6. **多尺度周期叠加**：日周期+周周期的加法组合简洁有效，提示我们可以进一步加入"节假日周期"、"季节性周期"等更多尺度，以覆盖更丰富的模式。
