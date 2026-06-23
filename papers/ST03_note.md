# ST03: DoFlow: Flow-Based Generative Models for Interventional and Counterfactual Forecasting on Time Series

## 基本信息
- **会议/期刊**: ICLR 2026 (International Conference on Learning Representations)
- **年份**: 2026
- **作者**: Dongze Wu, Yao Xie (Georgia Institute of Technology); Feng Qiu (Argonne National Laboratory)
- **关键词**: 因果时间序列预测、连续归一化流 (CNF)、干预预测、反事实预测、异常检测、有向无环图 (DAG)、结构因果模型 (SCM)、Conditional Flow Matching

## 研究问题与动机

传统时间序列预测模型（如 ARIMA、Transformer、RNN 等）本质上是**观测性 (observational)** 的：它们仅从历史数据中学习相关性并外推到未来，无法回答"如果……会怎样"的因果问题。然而在实际应用中，决策者常常需要回答两类因果查询：

1. **干预查询 (Interventional Query)**：如果我们对某些变量施加特定干预（例如改变水电涡轮控制信号），系统未来轨迹将如何变化？形式化为 $p(X_{\tau+1:T} | x_{1:\tau}, do(X_I := \gamma_I))$。
2. **反事实查询 (Counterfactual Query)**：给定已经观测到的事实轨迹，如果我们当时采取了不同的干预，该轨迹会变成什么样？形式化为 $p(X^{CF}_{\tau+1:T} | x_{1:\tau}, x^F_{\tau+1:T}, do(X_I := \gamma_I))$。

现有方法的不足：
- 大多数现代预测器仅支持观测预测，无法模拟干预和反事实轨迹。
- 现有时序因果效应方法主要关注离散、固定时间的处理效应估计，不支持对任意时刻连续变量的干预。
- **目前尚无通用框架支持基于因果 DAG 的反事实时间序列预测**。

## 核心贡献

1. **提出 DoFlow 框架**：首个在因果 DAG 上定义的基于连续归一化流 (CNF) 的生成模型，统一支持观测预测、干预预测和反事实预测三种模式。
2. **反事实恢复理论保证**：在单调 SCM 及标准因果假设下（Assumption 4.1），证明了 DoFlow 的编码-解码机制可以几乎必然地恢复真实反事实值（Corollary 4.5）。
3. **基于似然的异常检测**：DoFlow 为未来轨迹提供显式对数密度 $\log p_\theta(\hat{x}_{\tau+1:T} | \hat{H}_\tau)$，可用于原则性的异常检测——当上下文异常时，模型对其生成的预测轨迹赋予较低密度。
4. **广泛的实验验证**：在多种合成因果 DAG 结构（Tree、Diamond、FC-Layer、Chain）及两个真实数据集（水力发电系统、癌症治疗）上进行了全面评估。

## 方法概述

### 整体架构

DoFlow 在一个已知的因果 DAG 上运行，包含 $K$ 个节点 $\{1, \dots, K\}$，按拓扑排序排列。每个节点 $i$ 在时间步 $t$ 的值 $X_{i,t}$ 依赖于自身历史和父节点历史：

$$X_{i,t} := f_i(X_{i,t_-}, X_{pa(i),t_-}, U_{i,t})$$

其中 $U_{i,t}$ 是跨节点和时间独立的外生噪声。假设**同一时间步内无因果关系**（所有因果影响至少滞后一个时间步）。

### 核心组件

每个节点 $i$ 拥有独立的：
- **RNN 编码器**：LSTM/GRU，将节点及其父节点的历史压缩为隐状态 $H_{i,t-1} = \text{concat}(h_{i,t-1}, h_{pa(i),t-1})$
- **条件 CNF**：以 $H_{i,t-1}$ 为条件的连续归一化流，建模 $X_{i,t}$ 的条件分布

### 三种预测模式

| 模式 | 隐变量来源 | 条件状态 | 说明 |
|------|-----------|---------|------|
| 观测预测 | $z_{i,t} \sim \mathcal{N}(0,1)$ | $\hat{H}_{i,t-1}$（从预测值更新） | 标准自回归预测 |
| 干预预测 | $z_{i,t} \sim \mathcal{N}(0,1)$；被干预节点直接赋值 $\gamma_{i,t}$ | $\hat{H}_{i,t-1}$ | 干预通过 RNN 状态传播到下游 |
| 反事实预测 | $z^F_{i,t} = \Phi_\theta(x^F_{i,t}; H^F_{i,t-1})$（从事实轨迹编码） | $\hat{H}^{CF}_{i,t-1}$ | Abduction → Action → Prediction 三步流程 |

## 关键技术细节

### 1. 连续归一化流 (CNF) 基础

CNF 通过 Neural ODE 定义从目标分布 ($s=0$) 到基分布 $\mathcal{N}(0,1)$ ($s=1$) 的连续变换：

$$\frac{dx_{i,t}(s)}{ds} = v_i(x_{i,t}(s), s; H_{i,t-1}), \quad s \in [0,1]$$

- **前向过程（编码）**：$\Phi_\theta(x_{i,t}; H_{i,t-1}) = x_{i,t} + \int_0^1 v_i(x_{i,t}(s), s; H_{i,t-1}) ds$，将观测值推送到潜空间
- **反向过程（解码）**：$\Phi_\theta^{-1}(z_{i,t}; \hat{H}_{i,t-1}) = z_{i,t} - \int_0^1 v_i(x_{i,t}(s), s; \hat{H}_{i,t-1}) ds$，从潜空间生成预测

### 2. Conditional Flow Matching (CFM) 训练

使用线性插值参考路径 $\phi(x_{i,t}, z; s) = (1-s)x_{i,t} + sz$，其导数 $\partial_s \phi = z - x_{i,t}$ 作为参考速度场。训练损失为：

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}\left[\frac{1}{K(T-\tau)} \sum_{i=1}^K \sum_{t=\tau+1}^T \mathbb{E}_{s \sim \mathcal{U}[0,1], z \sim \mathcal{N}(0,I)} \|v_i(\phi(x_{i,t}, z; s), s; H_{i,t-1}) - \partial_s \phi\|_2^2\right]$$

### 3. 干预传播机制

推理时按拓扑排序逐节点预测：
- 被干预节点 $(i,t) \in I$：直接设置 $\hat{x}_{i,t} = \gamma_{i,t}$
- 非干预节点：从 CNF 采样生成
- 所有节点的 RNN 状态均从预测/干预值自回归更新，确保干预信息通过 DAG 边和 RNN 状态向下游传播

### 4. 反事实预测的 Abduction-Action-Prediction

遵循 Pearl 的标准三步流程：
1. **Abduction**：用事实隐状态 $H^F_{i,t-1}$ 将事实观测 $x^F_{i,t}$ 编码为 $z^F_{i,t} = \Phi_\theta(x^F_{i,t}; H^F_{i,t-1})$
2. **Action**：应用干预计划 $I$
3. **Prediction**：用反事实隐状态 $\hat{H}^{CF}_{i,t-1}$ 解码 $\hat{x}^{CF}_{i,t} = \Phi_\theta^{-1}(z^F_{i,t}; \hat{H}^{CF}_{i,t-1})$

关键洞察：由于 CNF 的可逆性，编码后的 $z^F_{i,t}$ 捕获了外生噪声 $U_t$ 的信息（Proposition 4.3），使得在不同条件下解码能产生正确的反事实。

### 5. 反事实恢复理论

**Assumption 4.1**:
- (A1) $U_t \perp\!\!\!\perp (X_{t_-}, X_{pa,t_-})$
- (A2) 结构方程 $f(\cdot, U_t)$ 对 $U_t$ 严格单调连续
- (A3) 编码后 $Z_t$ 满足 $p_\theta(Z_t | H_{t-1}) = q(Z_t)$（即完美训练下 CNF 将所有条件映射到同一基分布）

**Proposition 4.3**：在上述假设下，存在双射 $g: \mathcal{U} \to \mathcal{Z}$ 使得 $Z_t = g(U_t)$ a.s.，且该映射与历史状态无关。

**Corollary 4.5（反事实恢复）**：$\hat{X}^{CF}_t := \Phi_\theta^{-1}(Z^F_t; \hat{H}^{CF}_{t-1}) = X^{CF}_t$ a.s.，即 DoFlow 精确恢复真实反事实值。证明通过对时间步归纳完成。

### 6. 异常检测

利用 CNF 的显式密度计算（Liouville 连续性方程）：

$$\log p_\theta(\hat{x}_{\tau+1:T} | \hat{H}_\tau) = \sum_{t=\tau+1}^T \left[\log q(z_t) + \int_0^1 \nabla \cdot v_\theta(x_t(s), s; \hat{H}_{t-1}) ds\right]$$

当上下文包含异常模式时，模型对其生成的预测轨迹赋予较低的对数密度，从而触发异常警报。

### 7. 局限性

- **需要已知 DAG**：当前假设因果 DAG 已知且正确指定，以及因果充分性（无未观测混杂因子）
- **可扩展性**：每个节点训练独立的 flow 网络，但每个网络比 Transformer 浅得多，总体参数量适中
- 未来方向：与时序因果发现或去混杂因果生成模型结合

## 实验

### 数据集

| 数据集 | 描述 | DAG 规模 | 窗口设置 |
|--------|------|---------|---------|
| Tree (合成) | 8 节点树状 DAG | $K=8$ | $T=120, \tau=90$ |
| Diamond (合成) | 10 节点菱形 DAG | $K=10$ | $T=120, \tau=90$ |
| FC-Layer (合成) | 10 节点全连接层 DAG | $K=10$ | $T=120, \tau=90$ |
| Chain (合成) | 50 节点链式 DAG（含跳跃连接） | $K=50$ | $T=120, \tau=90$ |
| Hydropower (真实) | Argonne 国家实验室水力发电系统，8 个传感器信号 | $K=8$ | context=180, pred=30 |
| Cancer Treatment (真实) | 患者级别肿瘤体积与治疗记录 | 4 actions + 1 outcome | context=55, pred=5 |

每种合成结构都测试了**加性噪声模型 (Additive)** 和**非线性非加性模型 (NLNA)** 两种 SCM。

### 基线方法

- **GRU**：纯 RNN 方法
- **TFT** (Temporal Fusion Transformer)：Transformer 时序预测
- **TiDE** (Time Series Dense Encoder)：MLP-based 长期预测
- **TSMixer**：MLP-Mixer 时序模型
- **DeepVAR**：深度 RNN + 多元高斯输出的概率模型
- **MQF2**：凸深度神经网络学习多元分位数函数

对于干预预测，基线通过为每个 DAG 节点训练独立模型并递归 rollout 来适配。基线方法**无法进行反事实预测**。

### 主要结果

**合成数据（Table 1）**：
- DoFlow 在观测预测上与最优基线持平或略优（RMSE 相当）
- 在干预预测上显著优于所有基线（RMSE 降低约 30%-50%）
- **唯一支持反事实预测的方法**，反事实 RMSE 极低（0.11-0.41）
- MMD 和 CRPS 指标同样一致表明 DoFlow 的优势

**水力发电系统（Table 7）**：
- 观测预测 RMSE：DoFlow 1.13 vs 次优 TiDE 1.49
- 干预预测 RMSE：DoFlow 1.21 vs 次优 TSMixer 2.11
- 异常检测：DoFlow 能在真实停电事件发生前 **10-20 分钟**提前检测到异常（通过对数密度下降触发）

**癌症治疗效果（Table 8）**：
- 在所有治疗预算配置和预测步长下，DoFlow 的 NRMSE 均优于 CRN、RMSN、MSM 三个基线
- 例如 $\gamma_c=5, \gamma_r=5$ 配置下第 3 步：DoFlow 1.25% vs CRN 2.43%

**计算效率（Tables 2-3）**：
- 模型大小与 TFT 相当，远小于 MQF2
- 每 epoch 训练时间 ~42s（A100），收敛需 ~10 epochs
- 采样时间与主流方法可比

## 与本项目的关联

本项目目标是构建**节假日/极端天气场景下的异常客流预测系统**。DoFlow 提供了以下直接相关的技术启示：

1. **异常检测机制的直接借鉴**：DoFlow 基于生成模型对数密度的异常检测方法非常适合本项目的异常客流识别场景。正常客流模式下模型赋予高密度，而节假日/极端天气导致的异常客流会使对数密度显著下降。这种方法比传统的阈值法更具理论基础，且能提供预测性的异常预警（如水电站案例中提前 10-20 分钟检测）。

2. **干预预测用于情景分析**：本项目可以借鉴 DoFlow 的干预预测能力，回答诸如"如果某条地铁线路因极端天气停运，其他线路客流将如何变化"的问题。这需要将交通网络建模为因果 DAG，然后在特定节点上施加干预。

3. **因果 DAG 结构的重要性**：DoFlow 强调显式嵌入因果结构而非仅学习相关性。对于客流系统，站点之间的客流传播具有天然的因果结构（枢纽站→周边站、上游景点→下游商圈等），显式建模这种结构有助于提升异常场景下的预测鲁棒性。

4. **Flow-based 生成模型的概率优势**：相比 diffusion 模型，CNF 提供精确似然，这对异常评分至关重要。同时 CFM 训练方式避免了 ODE 仿真的高成本。

5. **局限性提醒**：DoFlow 要求已知因果 DAG，在实际客流系统中 DAG 可能需要通过领域知识或因果发现算法获取。此外，当前假设因果充分性（无未观测混杂），但在客流场景中天气、大型活动等混杂因素可能未被完全观测，需要考虑去混杂扩展。

## 关键启发

1. **CNF 的可逆性是统一三种预测模式的基石**：编码（前向）将观测映射到潜空间，解码（反向）从潜空间生成预测。反事实预测的关键在于：编码后的潜变量捕获了外生噪声信息，使其可以在不同条件下重新解码。这一 "encode-under-factual-condition → decode-under-counterfactual-condition" 范式值得在其他生成模型中推广。

2. **RNN + CNF 的分工设计精巧**：RNN 负责压缩历史信息为条件向量，CNF 负责在每个时间步建模条件分布。两者解耦使得每个模块可以更简单（3 层 MLP + LSTM hidden=15），总参数量反而比大型 Transformer 更少。

3. **Conditional Flow Matching 的训练效率**：CFM 通过直接回归参考速度场避免了 ODE 仿真的训练开销，使 CNF 的训练成本接近普通回归任务。这对于需要在大量节点上分别训练模型的 DAG 场景尤为重要。

4. **拓扑排序推理保证因果一致性**：按 DAG 拓扑序逐节点生成确保了父节点的预测先于子节点完成，使干预能正确地沿因果边传播。这是将静态因果生成模型扩展到时序的关键设计。

5. **理论保证的实践意义**：Corollary 4.5 的反事实恢复结果虽然在理想假设下成立，但它指导了模型设计方向——确保 CNF 训练充分以满足 (A3)，选择满足单调性的架构等。即使在实际中假设不完全满足，该理论也为模型行为提供了可解释的基准。

6. **异常检测的前瞻性价值**：DoFlow 的水电站案例展示了生成模型异常检测的核心优势——不仅能检测当前异常，还能**提前预警**。这是因为模型学习的是正常动力学，当系统开始偏离正常模式时（即使尚未达到故障阈值），对数密度已经开始下降。这对本项目的客流异常预警具有直接参考价值。
