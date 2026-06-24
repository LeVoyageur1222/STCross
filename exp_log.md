# Data

## 1. 原始数据来源

### traffic_flow

- UCTB 共享单车数据（`.pkl` 格式），含 `Node.TrafficNode` (T, N)、`Node.StationInfo`（站点经纬度等）
- 已处理城市：NYC (820 站点)、CHI (585 站点)

### weather

- `data/raw_data/weather_his/*.csv`: 历史天气数据，用于极端天气掩码
  - BAY: 2017.01.01 - 2018.01.01
  - CHI: 2013.01.01 - 2023.01.01
  - DC: 2013.01.01 - 2018.01.01
  - LA: 2012.01.01 - 2013.01.01
  - MEL: 2021.01.01 - 2023.01.01
  - NYC: 2013.01.01 - 2024.01.01
- `data/raw_data/weather/*.parquet`: 天气预报数据（预留）

### holiday

- `data_preprocess/build_holiday_mask.py`: 基于规则生成，10 类联邦假日 + 选举日 + 城市级假日 + 调休

---

## 2. 数据统计与质量评估

### 2.1 流量数据概览

| 指标 | NYC | CHI |
|------|-----|-----|
| 时间范围 | 2014-10-01 ~ 2017-09-29 | 同左 |
| 时间粒度 | 60 min | 60 min |
| 时间步数 T | 26,280 | 26,280 |
| 节点数 N | 820 | 585 |
| 流量均值 | 1.75 | 0.67 |
| 流量最大值 | 237 | 181 |
| 零值占比 | 66.0% | 76.8% |
| 活跃节点 (std > 0) | 538 (65.6%) | 525 (89.7%) |
| 死节点 (std = 0) | 282 (34.4%) | 60 (10.3%) |
| mean > 1.0 节点数 | 379 (46.2%) | 141 (24.1%) |

**关键特征**：
- 数据极度稀疏（66-77% 零值），大量站点长期闲置
- NYC 有 34% 的完全死节点（全时段零流量），CHI 有 10%
- NYC 流量均值是 CHI 的 **2.6 倍**，跨城市迁移存在量纲差异
- 由于零值过多，MAPE 仅在 |y| > 0.5 时计算，作为参考指标

### 2.2 时序分布漂移

共享单车用户量随时间自然增长，导致后期流量系统性高于前期：

| 时段 | NYC 均值 | CHI 均值 | 说明 |
|------|----------|----------|------|
| 2014-10 ~ 2016-06 | 1.35 | 0.57 | 训练期 |
| 2016-07 ~ 2017-01 | 2.14 | 0.74 | 验证期 |
| 2017-02 ~ 2017-09 | 2.46 | 0.90 | 测试期 |
| **漂移幅度** | **+82%** | **+58%** | test vs train |

这是共享单车数据按时间划分的固有现象，per-node StandardScaler 可部分缓解。

---

## 3. 数据组织方案

### 3.1 处理后数据目录结构

```
data/processed_data/
├── traffic_flow/
│   ├── flow_NYC_2014-10-01_2017-09-30_60min.npy        # (26280, 820) int32
│   └── flow_CHI_2014-10-01_2017-09-30_60min.npy        # (26280, 585) int32
├── adjacency/                                            # ← 待构建
│   ├── adj_NYC_corr_k10.npy                              # (820, 820) float64, 稀疏
│   └── adj_CHI_corr_k10.npy                              # (585, 585) float64, 稀疏
├── holiday/
│   ├── mask_NYC_holiday_2014-10-01_2017-09-30_60min.npy  # (26280,) int8
│   ├── mask_NYC_weekend_2014-10-01_2017-09-30_60min.npy
│   ├── mask_CHI_holiday_2014-10-01_2017-09-30_60min.npy
│   └── mask_CHI_weekend_2014-10-01_2017-09-30_60min.npy
└── weather/
    ├── mask_NYC_all_2014-10-01_2017-09-30_60min.npy      # (26280,) int8
    └── mask_CHI_all_2014-10-01_2017-09-30_60min.npy
```

### 3.2 邻接矩阵构建（Group B 必需）

UCTB 原始 pkl 文件不在磁盘上，无法直接获取站点坐标。采用**基于流量相关性的邻接矩阵**构建方案：

**构建方法**：
1. 使用 **train split** 的流量数据计算节点间 Pearson 相关系数矩阵 R ∈ R^{N×N}
2. 死节点（std=0）与其他节点的相关性设为 0
3. 对角线设为 0（无自环）
4. 负相关截断为 0：`R = max(R, 0)`
5. Top-K 稀疏化：每个节点仅保留相关性最高的 K 个邻居（默认 K=10）
6. 对称化：`A = max(A, A^T)`

**选择理由**：
- 不依赖外部数据（站点坐标），仅用训练集流量即可构建
- 相关性捕获功能相似性（比地理距离更有意义的空间关系）
- Top-10 稀疏化后边密度 ~1.2%（NYC）/ ~1.7%（CHI），适合图卷积

**可选扩展**：若后续获取 UCTB pkl 文件中的 `Node.StationInfo`（含经纬度），可额外构建距离邻接矩阵 `adj_{city}_dist.npy`，使用高斯核 A_ij = exp(-d²/σ²)。

---

## 4. 数据划分方案

### 4.1 月份对齐的时序划分

将划分边界对齐至月份首日，避免切在月中导致不完整的周期模式：

| 划分 | 时间范围 | 时间步数 | 比例 | 覆盖月份 |
|------|----------|:--------:|:----:|----------|
| **Train** | 2014-10-01 ~ 2016-06-30 | 15,336 | 58.4% | 全部 12 个月 ✓ |
| **Val** | 2016-07-01 ~ 2017-01-31 | 5,160 | 19.6% | 7-12, 1（7 个月） |
| **Test** | 2017-02-01 ~ 2017-09-29 | 5,784 | 22.0% | 2-9（8 个月） |
| **合计** | — | 26,280 | 100% | — |

**配置方式**（在 `eval/config.py` 中）：

```python
# 月份对齐的固定划分点（时间步索引）
train_steps: int = 15336   # 2014-10-01 00:00 ~ 2016-06-30 23:00
val_steps: int = 5160      # 2016-07-01 00:00 ~ 2017-01-31 23:00
# test_steps = T - train_steps - val_steps = 5784
```

### 4.2 各划分统计

| 划分 | NYC 均值 | NYC 异常% | CHI 均值 | CHI 异常% |
|------|:--------:|:---------:|:--------:|:---------:|
| Train | 1.35 | 8.62% (1322 步) | 0.57 | 9.74% (1494 步) |
| Val | 2.14 | 9.40% (485 步) | 0.74 | 11.65% (601 步) |
| Test | 2.46 | 10.25% (593 步) | 0.90 | 9.16% (530 步) |

异常分布跨三个划分相对均匀，确保评测可靠性。

### 4.3 已知局限与说明

| 局限 | 原因 | 影响 | 处理方式 |
|------|------|------|----------|
| 时序分布漂移 | 共享单车自然增长 | 模型需泛化到更高流量 | per-node StandardScaler 缓解；业界通用做法 |
| Val/Test 季节不完整 | 3 年数据时序切分的固有约束 | 秋冬模式仅在 train 中出现 | train 覆盖全部 12 月，模型可从中学习 |
| NYC 死节点多 (34%) | 部分站点未运营或极少使用 | 增加零值噪声 | 保留所有节点（保持图结构完整），死节点 std=0 → std=1.0 |

---

## 5. 统一数据接入规范

所有 baseline 使用**同一份数据、同一划分、同一标准化**，确保公平对比。

### 5.1 标准化

- 方法：**Per-node StandardScaler**
- Fit 数据：**target 城市 train split**（无论 source 是哪个城市）
- 死节点处理：std=0 → std=1.0
- 指标计算：在反标准化后的**原始尺度**上

### 5.2 各模型组的数据输入格式

| 模型组 | 输入格式 | 邻接矩阵 | 说明 |
|--------|----------|----------|------|
| **Group A（时序）** | `(batch, H, 1)` per node | 不需要 | 每个节点独立输入，N 个节点共享模型权重 |
| **Group B（时空图）** | `(batch, H, N)` | `(N, N)` | 所有节点联合输入，空间依赖通过邻接矩阵建模 |
| **Group C（事件感知）** | `(batch, H, N)` + event mask | 视方法而定 | 额外输入异常掩码或事件特征 |

### 5.3 跨城市迁移协议

| 阶段 | 数据来源 | 邻接矩阵 | 标准化 |
|------|----------|----------|--------|
| **训练** | source 城市 train split | source 邻接矩阵 | source per-node scaler |
| **验证** | target 城市 val split | target 邻接矩阵 | target per-node scaler |
| **测试** | target 城市 test split | target 邻接矩阵 | target per-node scaler |

> **注**：Group A（时序方法）的训练不涉及邻接矩阵。source→target 迁移时，模型权重直接从 source 训练好的模型加载，仅更换数据和邻接矩阵。

### 5.4 滑动窗口与异常标签

- 滑动窗口：`X = flow[i:i+H]`, `Y = flow[i+H:i+H+P]`，默认 H=12, P=12
- 异常定义：`anomaly = holiday OR weather`（weekend 不算异常）
- 样本标签：预测窗口 P 步中异常步占比 > `anomaly_min_ratio`（默认 0.5）时标记为异常
- 评测子集：full / anomaly / normal

---

# Baseline 方法梳理

> 基于 ST08/ST09/ST10/ST11/ST13 五篇高相关性论文 + 近年时间序列预测领域进展，
> 按跨城市迁移场景下的适用性筛选并分组。

## 1. 跨城市迁移适配性筛选

本项目 NYC 有 820 节点、CHI 有 585 节点，跨城市迁移要求模型能处理不同节点数。

### ❌ 排除的方法（节点数绑定，难以跨城市迁移）

| 方法 | 排除原因 |
|------|----------|
| **AGCRN** (NeurIPS 2020) | 核心依赖节点嵌入矩阵 E∈R^{N×d}，自适应图 A=softmax(EE^T)，维度与 N 绑定 |
| **D²STGNN** (VLDB 2022) | 动态图生成 + 节点嵌入，N-dependent |
| **STID** (CIKM 2022) | 核心设计为节点身份嵌入（Spatial ID），本质上是 per-node lookup，不可跨城市 |
| **STAEFormer** (CIKM 2023) | 自适应空间嵌入与 N 绑定 |
| **ASTGCN** (AAAI 2019) | 空间注意力包含可学习偏置 b_s∈R^{N×N}，N-dependent |
| **DSTAGNN** (NeurIPS 2021) | 动态空间注意力含节点级参数 |
| **iTransformer** (ICLR 2024) | 将每个变量作为一个 token，token 数=节点数，不同 N 无法直接迁移 |
| **PDFormer** (AAAI 2023) | 传播延迟建模含空间位置编码，N-dependent |

### ✅ 保留的方法

**筛选标准**：模型的可学习参数与节点数 N **无关**，可通过替换邻接矩阵或天然 channel-independent 实现跨城市迁移。

## 2. Baseline 分组

### Group A：时序建模方法（Channel-Independent，无图依赖）

每个节点独立建模，共享权重。训练时不使用图结构，天然支持不同节点数的跨城市迁移。
回答核心问题：**纯时序模式能否捕获跨城市可迁移的流量规律？**

| 方法 | 发表 | 类型 | 论文 | 代码 | 概要 |
|------|------|------|------|------|------|
| **HA** | — | 统计 | — | ✅ 已实现 | 按 (hour, weekday, node) 历史均值预测。非参数，始终用 target 训练集 |
| **LSTM** | 1997 | RNN | Hochreiter & Schmidhuber | PyTorch 内置 | 经典循环网络，逐节点共享权重，最基础深度学习 baseline |
| **DLinear** | AAAI 2023 | 线性 | *Are Transformers Effective for Time Series Forecasting?* | https://github.com/cure-lab/LTSF-Linear | 趋势-季节分解 + 线性层。证明简单线性模型可媲美复杂 Transformer |
| **PatchTST** | ICLR 2023 | Transformer | *A Time Series is Worth 64 Words* | https://github.com/yuqinie98/PatchTST | 将时间序列切为 patch 作为 Transformer token，channel-independent 共享权重 |
| **TimesNet** | ICLR 2023 | CNN | *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis* | https://github.com/thuml/TimesNet | FFT 检测周期性 → 展开为 2D 张量 → Inception 2D 卷积捕获多周期模式 |

### Group B：时空图建模方法（图卷积核 N-independent，替换邻接矩阵迁移）

显式建模空间依赖。关键：图卷积核的可学习参数（Chebyshev 系数、扩散权重等）与节点数无关，跨城市时替换邻接矩阵即可。
回答核心问题：**加入空间结构建模是否提升跨城市迁移效果？**

| 方法 | 发表 | 空间模块 | 时间模块 | 论文 | 代码 | 概要 |
|------|------|----------|----------|------|------|------|
| **STGCN** | IJCAI 2018 | Chebyshev 图卷积 | 门控时间卷积 | *Spatio-Temporal Graph Convolutional Networks* | https://github.com/VeritasYin/STGCN_IJCAI-18 | 开创性 STGNN 工作，图卷积 + 1D 因果卷积 |
| **DCRNN** | ICLR 2018 | 扩散卷积 | GRU Seq2Seq | *Diffusion Convolutional Recurrent Neural Network* | https://github.com/liyaguang/DCRNN | 有向图扩散过程建模空间 + 编解码器建模时间 |
| **GWNET** | IJCAI 2019 | 扩散卷积 + 自适应邻接 | 膨胀因果卷积 | *Graph WaveNet for Deep Spatial-Temporal Graph Modeling* | https://github.com/nnzhan/Graph-WaveNet | 固定扩散部分可迁移；自适应邻接含节点嵌入(N×d)，跨城市需禁用或重初始化 |
| **STGODE** | KDD 2021 | 图 ODE | 时间 ODE | *Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting* | https://github.com/square-coder/STGODE | 连续时间 ODE 建模时空耦合，图 Laplacian 为输入非参数 |

> **GWNET 跨城市适配说明**：GWNET 的自适应邻接矩阵由节点嵌入 E₁,E₂∈R^{N×d} 生成，与 N 绑定。跨城市迁移策略：(a) 仅使用预定义邻接矩阵的扩散部分（禁用 adaptive）；(b) 为 target 城市随机初始化新的节点嵌入。两种策略均可实现，实验中可对比。

### Group C：事件感知 / 分布偏移方法（与本文场景直接相关）

这些方法来自五篇高相关论文自身的贡献，直接处理异常事件或分布偏移问题，是与本文方法最直接的对比对象。
回答核心问题：**现有事件感知方法在跨城市异常预测中表现如何？**

| 方法 | 发表 | 论文编号 | 论文 | 代码 | 概要 | 跨城市适配 |
|------|------|----------|------|------|------|:----------:|
| **ST-Norm** | KDD 2021 | — | *ST-Norm: Spatial and Temporal Normalization* | https://github.com/JLDeng/ST-Norm | 时空分别归一化处理分布偏移，轻量级插件可嵌入任意 backbone | ✅ 可迁移 |
| **ConFormer** | KDD 2026 | ST11 | *Conditional Transformer with Guided Layer Normalization* | 待确认 | 条件 LayerNorm 根据外部事件动态调整归一化参数 | ⚠️ 需评估 |
| **STEVE** | KDD 2025 | ST13 | *Spatial-Temporal Self-Supervised Confounder Learning* | 待确认 | 混淆因子基底 + 对抗解耦 + 自监督辅助任务 | ⚠️ 需评估 |

> **注**：Group C 方法为 2025-2026 年最新发表，部分代码可能尚未开源。ConFormer 和 STEVE 在跨城市场景下可能需要额外适配（其空间模块可能含 N-dependent 参数），待获取代码后进一步评估。

## 3. 五篇论文方法与 Baseline 重叠分析（参考）

下表列出上述保留方法在五篇论文中的出现情况（✓=baseline, ★=本文方法），仅含通过适配性筛选的方法：

| 方法 | ST08 | ST09 | ST10 | ST11 | ST13 | 出现次数 |
|------|:----:|:----:|:----:|:----:|:----:|:--------:|
| **HA** | | | ✓ | ✓ | | 2 |
| **LSTM** | | ✓ | ✓ | ✓ | | 3 |
| **STGCN** | | | ✓ | ✓ | ✓ | 3 |
| **DCRNN** | | ✓ | ✓ | ✓ | | 3 |
| **GWNET** | | ✓ | ✓ | ✓ | | 3 |
| **STGODE** | | ✓ | | ✓ | | 2 |
| **ST-Norm** | | | | | ✓ | 1 |
| *DLinear* | | | | | | 0（新增） |
| *PatchTST* | | | | | | 0（新增） |
| *TimesNet* | | | | | | 0（新增） |
| **ConFormer** | | | | ★ | | — |
| **STEVE** | | | | | ★ | — |

> DLinear、PatchTST、TimesNet 虽未被这五篇论文选用，但它们是近年时间序列预测领域的重要方法，且天然 channel-independent，非常适合跨城市迁移场景。

## 4. 实施优先级

| 优先级 | 方法 | 理由 |
|--------|------|------|
| **P0（已完成）** | HA | 非参数基线，框架验证 |
| **P1（基础对比）** | LSTM, DLinear | 最简深度学习 baseline，验证是否需要复杂模型 |
| **P2（时序 SOTA）** | PatchTST, TimesNet | 近年最强 channel-independent 方法，验证时序建模上限 |
| **P3（时空图）** | STGCN, DCRNN | 经典 STGNN，验证空间建模收益 |
| **P4（时空图增强）** | GWNET, STGODE | 更强的 STGNN，含自适应/连续时间建模 |
| **P5（事件感知）** | ST-Norm, ConFormer, STEVE | 与本文方法最直接的对比，待代码可用后实施 |
