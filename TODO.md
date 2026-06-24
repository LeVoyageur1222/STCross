# STCross 实验规划

> 本文件为 STCross 项目的实验主规划，所有后续实验开发均以此为准。
> 最后更新：2026-06-24（Phase 0 完成 + 代码质量审查修复）

---

## 实验总览

### 研究目标

基于时空交叉特征的异常客流预测，核心实验范式：**跨城市迁移**（source 城市训练 → target 城市评测）。

### 数据设置

| 项目 | 值 |
|------|-----|
| 城市 | NYC (820 节点), CHI (585 节点) |
| 时间范围 | 2014-10-01 ~ 2017-09-29, 60min 粒度, T=26280 |
| 划分（月份对齐） | Train: 15336 步 (58.4%) / Val: 5160 步 (19.6%) / Test: 5784 步 (22.0%) |
| 异常定义 | holiday OR weather, anomaly_min_ratio=0.5 |
| 窗口 | H=12 (输入) → P=12 (预测), 可配置 |
| 标准化 | Per-node StandardScaler, fit on target city train split |
| 指标 | MAE, RMSE, MAPE (|y|>0.5), PCC, R² × 3 子集 (full/anomaly/normal) |

### 实验矩阵

| 设置 | 说明 | 训练数据 | 邻接矩阵 | 评测数据 |
|------|------|----------|----------|----------|
| NYC→NYC | 同城基准 | NYC train | NYC adj | NYC test |
| CHI→CHI | 同城基准 | CHI train | CHI adj | CHI test |
| NYC→CHI | 跨城迁移 | NYC train | CHI adj | CHI test |
| CHI→NYC | 跨城迁移 | CHI train | NYC adj | NYC test |

> Group A（时序方法）不使用邻接矩阵。HA 始终用 target train 计算，无需跨城市。

### Baseline 方法清单

| 组别 | 方法 | 实现方式 | 依赖 |
|------|------|----------|------|
| A（时序） | HA | ✅ 已完成 | — |
| A（时序） | LSTM | 自主实现 | PyTorch 训练管线 |
| A（时序） | DLinear | 自主实现 | PyTorch 训练管线 |
| A（时序） | PatchTST | 适配开源 | PyTorch 训练管线 |
| A（时序） | TimesNet | 适配开源 | PyTorch 训练管线 |
| B（时空图） | STGCN | 适配开源 | 训练管线 + 邻接矩阵 |
| B（时空图） | DCRNN | 适配开源 | 训练管线 + 邻接矩阵 |
| B（时空图） | GWNET | 适配开源 | 训练管线 + 邻接矩阵 |
| B（时空图） | STGODE | 适配开源 | 训练管线 + 邻接矩阵 |
| C（事件感知） | ST-Norm | 适配开源 | 训练管线 |
| C（事件感知） | ConFormer | 待确认 | 代码开源后适配 |
| C（事件感知） | STEVE | 待确认 | 代码开源后适配 |

---

## Phase 0：基础设施建设

> **目标**：搭建可复用的 PyTorch 训练管线和数据组件，为所有深度学习 baseline 提供统一基础。
> **依赖**：无
> **预期产出**：`train/` 目录下的训练框架、`data_preprocess/` 下的 Dataset/adjacency 构建

### 0.1 更新依赖

- [x] `requirements.txt` 添加 `torch>=2.0.0`、`einops>=0.6.0`
- [x] 验证环境：Python 3.10.20 + PyTorch 2.5.1 + MPS (Apple Silicon)

### 0.2 PyTorch Dataset 与 DataLoader

> 文件：`data_preprocess/torch_dataset.py` ✅ 已完成

- [x] 实现 `TrafficFlowDataset(torch.utils.data.Dataset)`
  - 包装 `create_sliding_windows` 产出的 numpy 数组 (X, Y, sample_labels)
  - 统一使用 all_nodes 格式存储：`(num_samples, H, N)` / `(num_samples, P, N)`
  - per_node 的 reshape 由 `TorchModel._forward_wrapper()` 在模型内部处理，避免样本数爆炸
- [x] `build_dataloader()` 封装：可配置 batch_size 和 shuffle
- [x] 验证通过：batch shape (B, H, N)/(B, P, N)，数据完整性校验通过

### 0.3 训练管线

> 文件：`train/trainer.py` ✅ 已完成

- [x] 实现 `Trainer` 类，核心功能：
  - `train(model, train_loader, val_loader, config) → best_model_path`
  - 训练循环：epoch loop → batch loop → forward → loss → backward → step
  - 损失函数：MSE（在标准化空间上）
  - 优化器：Adam, lr 可配置（默认 1e-3）
  - 学习率调度：ReduceLROnPlateau（patience=5, factor=0.5）
  - **Early stopping**：val loss 连续 N epochs 无改善则停止（默认 patience=15）
  - 最优模型保存：保存 val loss 最低的 checkpoint 到 `checkpoints/`
  - 梯度裁剪：max_norm=5.0
  - 训练日志：每 epoch 打印 train_loss / val_loss（`*` 标记改善）
- [x] 实现 `TrainConfig` 数据类：
  - epochs, batch_size, lr, weight_decay, early_stop_patience, lr_patience, lr_factor, min_lr, checkpoint_dir, device
  - `resolve_device()`：auto 模式自动选择 cuda > mps > cpu

### 0.4 BaseModel 接口扩展

> 文件：`model/torch_base.py` ✅ 已完成

- [x] 保持现有 numpy 接口 `fit/predict` 不变（HA 等非 PyTorch 模型继续使用）
- [x] 新增 `TorchModel(BaseModel, nn.Module)` 基类：
  - 继承 `BaseModel` + `nn.Module`
  - `fit()` 内部：numpy → DataLoader (all_nodes 格式) → Trainer 训练 → 加载 best checkpoint
  - `predict()` 内部：numpy → Tensor → `_forward_wrapper()` → numpy，分批推理
  - `_forward_wrapper()`：per_node 模型自动将 (B,H,N)→(B\*N,H,1) 送入 forward()，再 reshape 回 (B,P,N)
  - 子类只需实现 `forward(X) → Y_pred` 和 `name` 属性
  - 属性 `input_mode`：`"per_node"` (Group A) 或 `"all_nodes"` (Group B)
- [x] 端到端验证通过：per_node 模式 (DummyLinear) + all_nodes 模式 (DummySTGNN) 均通过 Evaluator 完整流程

### 0.5 邻接矩阵构建

> 文件：`data_preprocess/build_adjacency.py` ✅ 已完成

- [x] 实现 `build_correlation_adj(flow_train, top_k=10) → np.ndarray (N, N)`
  - Pearson 相关系数矩阵
  - 死节点（std=0）相关性设为 0
  - 截断负值：`max(R, 0)`
  - 对角线设为 0
  - Top-K 稀疏化 + 对称化
  - 保存到 `data/processed_data/adjacency/adj_{city}_corr_k{K}.npy`
- [x] NYC 和 CHI 邻接矩阵已生成：
  - NYC: 820×820, 9608 edges (1.43%), 538 active nodes 形成单一连通分量, 282 isolated
  - CHI: 585×585, 9696 edges (2.83%), 525 active nodes 形成单一连通分量, 60 isolated
- [x] EvalConfig 新增 `adj_dir`, `adj_top_k`, `adj_path(city)` 方法

### 0.6 批量实验运行器

> 文件：`experiments/run_all.py` ✅ 已完成

- [x] 实现批量实验脚本：
  - `MODEL_REGISTRY` + `@register` 装饰器：新模型只需注册 build 函数
  - `EXPERIMENT_MATRIX`：4 种实验设置（NYC→NYC, CHI→CHI, NYC→CHI, CHI→NYC）
  - `run_experiments()`: 遍历模型×设置，自动跳过已完成实验（检查 `results/` 目录）
  - `compile_summary()`: 汇总所有结果为 Markdown 表格，保存到 `results/summary.md`
  - CLI 参数：`--models`, `--settings {same-city,cross-city,all}`, `--epochs`, `--batch_size`, `--lr`
- [x] 验证通过：HA same-city 运行正确，重复运行自动跳过

---

## Phase 1：Group A Baselines（时序方法）

> **目标**：完成所有 channel-independent 时序方法的实现与同城评测。
> **依赖**：Phase 0（训练管线）
> **预期产出**：4 个模型实现 + NYC→NYC / CHI→CHI 评测结果

### 1.1 LSTM

> 文件：`model/lstm.py` | 实现方式：自主实现

- [ ] 网络结构：
  - 输入 `(batch, H, 1)` → 2 层 LSTM (hidden=64) → Linear(64, P)
  - 输出 `(batch, P, 1)` → reshape 回 `(batch, P)`
- [ ] 继承 `TorchModel`，`input_mode="per_node"`
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果到 `results/`

### 1.2 DLinear

> 文件：`model/dlinear.py` | 实现方式：自主实现

- [ ] 网络结构（参考 LTSF-Linear）：
  - 移动平均分解为 trend + seasonal
  - 两个独立 Linear(H, P) 分别映射
  - 输出相加
- [ ] 继承 `TorchModel`，`input_mode="per_node"`
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 1.3 PatchTST

> 文件：`model/patchtst.py` | 实现方式：适配开源
> 代码源：https://github.com/yuqinie98/PatchTST

- [ ] 从开源仓库提取核心模块（PatchEmbedding, TransformerEncoder）
- [ ] 适配为 `TorchModel` 子类，`input_mode="per_node"`
- [ ] 关键超参：patch_len=4, stride=2, d_model=64, n_heads=4, n_layers=2
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 1.4 TimesNet

> 文件：`model/timesnet.py` | 实现方式：适配开源
> 代码源：https://github.com/thuml/TimesNet

- [ ] 从开源仓库提取 TimesBlock（FFT 周期检测 + 2D Inception）
- [ ] 适配为 `TorchModel` 子类，`input_mode="per_node"`
- [ ] 关键超参：d_model=64, n_layers=2, top_k_periods=3
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 1.5 Group A 同城结果汇总

- [ ] 汇总表：HA / LSTM / DLinear / PatchTST / TimesNet × NYC/CHI × full/anomaly/normal
- [ ] 初步分析：时序方法间的对比，是否有显著差异

---

## Phase 2：Group B Baselines（时空图方法）

> **目标**：完成 STGNN 方法的适配与同城评测。
> **依赖**：Phase 0（训练管线 + 邻接矩阵）
> **预期产出**：4 个 STGNN 模型 + NYC→NYC / CHI→CHI 评测结果

### 2.0 构建邻接矩阵

- [x] 运行 `build_adjacency.py`，已生成：
  - `data/processed_data/adjacency/adj_NYC_corr_k10.npy` (820×820, 9608 edges)
  - `data/processed_data/adjacency/adj_CHI_corr_k10.npy` (585×585, 9696 edges)
- [x] 邻接矩阵质量验证通过（活跃节点单一连通分量）

### 2.1 STGCN

> 文件：`model/stgcn.py` | 实现方式：适配开源
> 代码源：https://github.com/VeritasYin/STGCN_IJCAI-18

- [ ] 核心模块：ChebConv（Chebyshev 图卷积） + TemporalConv（门控时间卷积） + ST-Conv Block
- [ ] 适配为 `TorchModel` 子类，`input_mode="all_nodes"`
- [ ] 构造函数接受 `adj_matrix` 参数
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 2.2 DCRNN

> 文件：`model/dcrnn.py` | 实现方式：适配开源
> 代码源：https://github.com/liyaguang/DCRNN

- [ ] 核心模块：DiffusionConvGRU + Seq2Seq (编解码器)
- [ ] 扩散矩阵从邻接矩阵计算（前向扩散 + 后向扩散）
- [ ] 适配为 `TorchModel` 子类，`input_mode="all_nodes"`
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 2.3 GWNET

> 文件：`model/gwnet.py` | 实现方式：适配开源
> 代码源：https://github.com/nnzhan/Graph-WaveNet

- [ ] 核心模块：扩散卷积 + WaveNet 膨胀因果卷积 + 自适应邻接
- [ ] **跨城市适配**：提供两种模式：
  - `adaptive=False`：仅使用预定义扩散矩阵（完全可迁移）
  - `adaptive=True`：包含节点嵌入（同城实验用，跨城市需重初始化）
- [ ] 适配为 `TorchModel` 子类，`input_mode="all_nodes"`
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 2.4 STGODE

> 文件：`model/stgode.py` | 实现方式：适配开源
> 代码源：https://github.com/square-coder/STGODE

- [ ] 核心模块：图 ODE 层 + 时间 ODE 层
- [ ] 适配为 `TorchModel` 子类，`input_mode="all_nodes"`
- [ ] 同城评测：NYC→NYC, CHI→CHI
- [ ] 记录结果

### 2.5 Group B 同城结果汇总

- [ ] 汇总表：STGCN / DCRNN / GWNET / STGODE × NYC/CHI × full/anomaly/normal
- [ ] 与 Group A 对比分析：空间建模是否有收益？

---

## Phase 3：跨城市迁移评测

> **目标**：所有 Group A + B 模型的跨城市迁移实验。
> **依赖**：Phase 1 + Phase 2
> **预期产出**：完整的跨城市对比表

### 3.1 Group A 跨城市

- [ ] LSTM：NYC→CHI, CHI→NYC
- [ ] DLinear：NYC→CHI, CHI→NYC
- [ ] PatchTST：NYC→CHI, CHI→NYC
- [ ] TimesNet：NYC→CHI, CHI→NYC

> Group A 模型跨城市迁移：在 source 城市训练（逐节点共享权重），直接在 target 城市各节点上推理。
> 标准化使用 target 城市 train split 的统计量。

### 3.2 Group B 跨城市

- [ ] STGCN：NYC→CHI, CHI→NYC（替换邻接矩阵）
- [ ] DCRNN：NYC→CHI, CHI→NYC（替换扩散矩阵）
- [ ] GWNET (adaptive=False)：NYC→CHI, CHI→NYC
- [ ] GWNET (adaptive=True, re-init embedding)：NYC→CHI, CHI→NYC
- [ ] STGODE：NYC→CHI, CHI→NYC

> Group B 跨城市迁移：在 source 城市图上训练，测试时替换为 target 城市的邻接矩阵。
> 图卷积核权重（Chebyshev 系数、扩散权重等）与 N 无关，可直接迁移。

### 3.3 跨城市结果汇总

- [ ] 完整对比表（示例格式）：

```
| Model    | NYC→NYC MAE | CHI→CHI MAE | NYC→CHI MAE | CHI→NYC MAE | 迁移衰减% |
|----------|-------------|-------------|-------------|-------------|-----------|
| HA       | x.xx        | x.xx        | —           | —           | —         |
| LSTM     | x.xx        | x.xx        | x.xx        | x.xx        | xx%       |
| DLinear  | x.xx        | x.xx        | x.xx        | x.xx        | xx%       |
| ...      |             |             |             |             |           |
```

> "迁移衰减%" = (跨城市 MAE - 同城 MAE) / 同城 MAE × 100%

- [ ] 分析：
  - 同城 vs 跨城迁移的性能差距
  - Group A vs Group B 在跨城市场景下的表现差异
  - 各方法在 anomaly 子集上的表现对比
  - GWNET adaptive vs non-adaptive 的迁移效果

---

## Phase 4：Group C Baselines（事件感知方法）

> **目标**：实现与本文场景最直接相关的事件/分布偏移感知方法。
> **依赖**：Phase 0（训练管线），部分依赖代码开源情况
> **优先级**：低于 Phase 1-3，可与 Phase 5 并行

### 4.1 ST-Norm

> 代码源：https://github.com/JLDeng/ST-Norm

- [ ] 获取代码，评估适配难度
- [ ] 适配到框架（ST-Norm 是插件式归一化，需嵌入 backbone）
- [ ] 选择 backbone（如 STGCN 或 GRU）
- [ ] 同城 + 跨城市评测
- [ ] 记录结果

### 4.2 ConFormer (ST11)

- [ ] 跟踪代码开源状态
- [ ] 若开源：评估跨城市适配性（检查是否有 N-dependent 参数）
- [ ] 适配 + 评测
- [ ] 若未开源：标记为无法复现，在论文中说明

### 4.3 STEVE (ST13)

- [ ] 跟踪代码开源状态
- [ ] 若开源：评估跨城市适配性
- [ ] 适配 + 评测
- [ ] 若未开源：标记为无法复现

---

## Phase 5：本文模型设计与实现

> **目标**：设计并实现本文提出的异常客流预测模型。
> **依赖**：Phase 1-3 全部完成（需要充分了解各 baseline 的优劣）
> **本阶段在所有 baseline 完成后启动**

### 5.1 模型设计

- [ ] 基于 baseline 实验结论，确定模型创新点方向：
  - 异常事件编码方式（掩码注入 / 条件归一化 / 混淆因子学习）
  - 时空特征融合架构
  - 跨城市迁移机制（域不变特征学习 / 城市自适应模块）
- [ ] 撰写模型设计文档（网络结构、损失函数、训练策略）
- [ ] 确认设计方案

### 5.2 模型实现

- [ ] 实现模型核心模块
- [ ] 适配为 `TorchModel` 子类
- [ ] 单元测试（前向传播维度检查、梯度流通检查）

### 5.3 训练与调参

- [ ] 同城训练：NYC→NYC, CHI→CHI
- [ ] 跨城市训练：NYC→CHI, CHI→NYC
- [ ] 超参搜索（学习率、隐藏维度、层数等）
- [ ] 最优模型保存

### 5.4 消融实验

- [ ] 设计消融变体（逐步去除创新模块）
- [ ] 运行消融实验
- [ ] 分析各模块贡献

---

## Phase 6：综合分析与论文撰写

> **依赖**：Phase 1-5

### 6.1 敏感性分析

- [ ] 窗口大小敏感性：H/P ∈ {6, 12, 24}（选取代表性模型）
- [ ] 异常阈值敏感性：anomaly_min_ratio ∈ {0.0, 0.25, 0.5, 0.75}
- [ ] (Group B) 邻接矩阵参数：Top-K ∈ {5, 10, 20}

### 6.2 结果可视化

- [ ] 预测曲线对比图（选取典型节点 + 异常时段）
- [ ] 异常时段预测放大图
- [ ] 跨城市迁移衰减柱状图
- [ ] 各方法在 full/anomaly/normal 三子集上的雷达图

### 6.3 论文撰写

- [ ] 实验章节：实验设置、主实验表格、分析
- [ ] 消融实验章节
- [ ] Case Study（典型异常事件预测示例）

---

## 执行顺序与依赖关系

```
Phase 0 (基础设施)
   │
   ├──→ Phase 1 (Group A: 时序方法)
   │       │
   │       └──→ Phase 3.1 (Group A 跨城市)
   │                │
   ├──→ Phase 2 (Group B: 时空图方法)      ├──→ Phase 5 (本文模型)
   │       │                                │        │
   │       └──→ Phase 3.2 (Group B 跨城市) ─┘        └──→ Phase 6 (分析 & 论文)
   │                                                         ↑
   └──→ Phase 4 (Group C: 事件感知方法) ─────────────────────┘
```

> Phase 1 和 Phase 2 可并行推进（Phase 2 额外需要邻接矩阵，Phase 0.5）。
> Phase 4 独立于 Phase 1-3，可随时穿插。
> Phase 5 严格在 Phase 1-3 完成后开始。

---

## 当前进度

- [x] **Phase 0.1**：依赖更新（PyTorch 2.5.1 + einops 已确认可用）
- [x] **Phase 0.2**：PyTorch Dataset 与 DataLoader（`data_preprocess/torch_dataset.py`）
- [x] **Phase 0.3**：训练管线（`train/trainer.py`：Trainer + TrainConfig + early stopping + checkpointing）
- [x] **Phase 0.4**：TorchModel 基类（`model/torch_base.py`：per_node/all_nodes 双模式 + numpy↔Tensor 桥接）
- [x] **Phase 0.5**：邻接矩阵（`data_preprocess/build_adjacency.py`：NYC 9608 edges / CHI 9696 edges）
- [x] **Phase 0.6**：批量实验运行器（`experiments/run_all.py`：模型注册 + 实验矩阵 + 跳过已完成 + 结果汇总）
- [x] **评测框架**：Evaluator, Reporter, 指标计算（MAE/RMSE/MAPE/PCC/R²）
- [x] **数据管线**：数据加载、滑动窗口、标准化、月份对齐划分
- [x] **HA Baseline**：已完成全量评测（同城 + 跨城市 4 种设置）
- [x] **代码质量审查 & 修复**：3 子智能体审查，修复 P0-P2 共 17 项问题（详见下方）
- [ ] Phase 1-6：全部待开始

### 代码质量审查修复记录（Phase 0 完成后）

**P0 — 正确性修复**：
- [x] 跨城市评测 scaler 逻辑重构：source/target 节点数不同时无法互用 scaler，改为始终用 target scaler 做 test I/O（`eval/evaluator.py`）
- [x] 异常标签 `>` → `>=`：6/12 异常步在 ratio=0.5 时应标记为异常（`data_preprocess/dataset.py`）
- [x] `calc_pcc` 零方差崩溃：添加 std<1e-8 前置检查（`metrics/metrics.py`）
- [x] `_forward_wrapper` shape 断言：防止 per_node 子类返回错误维度（`model/torch_base.py`）
- [x] 空 DataLoader 除零：`max(n_batches, 1)` 防护（`train/trainer.py`）
- [x] 所有指标函数添加空数组前置检查（`metrics/metrics.py`）

**P1 — 健壮性修复**：
- [x] 消除 `hasattr` 鸭子类型：重构为 `forward()` → `_net_forward()`，Trainer 直接调用 `model(X)`（`model/torch_base.py`, `train/trainer.py`）
- [x] 近零 std 防护：`std < 1e-8` → 1.0 + 警告日志（`data_preprocess/normalizer.py`）
- [x] 邻接矩阵 Top-K 确定性：`argpartition` → `argsort(kind='stable')`（`data_preprocess/build_adjacency.py`）
- [x] Checkpoint 存在性检查：训练无改善时抛出明确错误（`train/trainer.py`）
- [x] 跳过逻辑增加 config 校验：验证 H/P/anomaly_min_ratio/train_steps 匹配（`experiments/run_all.py`）
- [x] 共享模型注册表：`model/registry.py` 消除 `run_all.py`/`run_model.py` 重复定义
- [x] 批量运行器错误隔离：try/except 包裹单次实验，失败不中断全局（`experiments/run_all.py`）
- [x] 可复现性种子：`TrainConfig.seed=42` + `set_seed()` 在训练前设置（`train/trainer.py`）
- [x] `input_mode` 校验：构造时检查值合法性（`model/torch_base.py`）
- [x] 梯度裁剪可配置：`TrainConfig.grad_clip_norm`（`train/trainer.py`）

**P2 — 优化修复**：
- [x] `pin_memory` 条件启用：仅在 CUDA 可用时启用（`data_preprocess/torch_dataset.py`）
- [x] 邻接矩阵构建向量化：`np.ix_` 替代 O(N²) Python 循环（`data_preprocess/build_adjacency.py`）
- [x] 训练/评估合并为单一 `_run_epoch()`（`train/trainer.py`）
