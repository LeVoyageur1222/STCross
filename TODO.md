# STCross 实验规划

> 本文件为 STCross 项目的实验主规划，所有后续实验开发均以此为准。
> 相关文档：[运行指南](RUN_GUIDE.md) | [结果汇总](results/RESULTS.md)
>
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

### 实验矩阵（每个模型完成后立即运行全部 4 种设置）

| 设置 | 说明 | 训练数据 | 邻接矩阵 | 评测数据 |
|------|------|----------|----------|----------|
| NYC→NYC | 同城基准 | NYC train | NYC adj | NYC test |
| CHI→CHI | 同城基准 | CHI train | CHI adj | CHI test |
| NYC→CHI | 跨城迁移 | NYC train | CHI adj | CHI test |
| CHI→NYC | 跨城迁移 | CHI train | NYC adj | NYC test |

### 工作流

```
我（开发者）                         你（用户）
    │                                    │
    ├─ 实现模型代码 + 注册到 registry ─→ │
    ├─ 更新 RUN_GUIDE.md ──────────────→ │
    │                                    ├─ 按指南运行 4 种设置
    │                                    ├─ 确认 results/ 下生成 JSON
    │  ←─────── 通知模型完成 ────────────┤
    ├─ 收集结果更新 RESULTS.md ─────────→ │
    ├─ 更新 TODO.md 进度 ──────────────→ │
    └─ 开发下一个模型 ──────────────────→ │
```

---

## Phase 0：基础设施建设 ✅ 已完成

<details>
<summary>展开查看 Phase 0 详情</summary>

### 0.1 更新依赖 ✅
- [x] `requirements.txt` 添加 `torch>=2.0.0`、`einops>=0.6.0`
- [x] 验证环境：Python 3.10.20 + PyTorch 2.5.1 + MPS (Apple Silicon)

### 0.2 PyTorch Dataset 与 DataLoader ✅
> 文件：`data_preprocess/torch_dataset.py`
- [x] `TrafficFlowDataset` + `build_dataloader()`
- [x] all_nodes 格式存储，per_node reshape 在模型内部处理

### 0.3 训练管线 ✅
> 文件：`train/trainer.py`
- [x] `Trainer` + `TrainConfig` + `set_seed()`
- [x] Adam + MSE + ReduceLROnPlateau + early stopping + 梯度裁剪 + checkpoint

### 0.4 TorchModel 基类 ✅
> 文件：`model/torch_base.py`
- [x] `TorchModel(BaseModel, nn.Module)`：子类只需实现 `_net_forward()` + `name`
- [x] per_node/all_nodes 双模式自动 reshape

### 0.5 邻接矩阵 ✅
> 文件：`data_preprocess/build_adjacency.py`
- [x] NYC: 820×820, 9608 edges (1.43%) | CHI: 585×585, 9696 edges (2.83%)

### 0.6 批量实验运行器 ✅
> 文件：`experiments/run_all.py` + `model/registry.py`
- [x] 共享模型注册表 + 实验矩阵 + 跳过已完成 + 错误隔离 + 结果汇总

### 代码质量审查修复 ✅
- [x] P0（6 项）：跨城市 scaler 重构、异常标签 `>=`、PCC 零方差防护、forward shape 断言、空 loader 除零、指标空数组检查
- [x] P1（10 项）：消除 hasattr、近零 std 防护、邻接确定性、checkpoint 检查、skip config 校验、共享 registry、错误隔离、种子、input_mode 校验、梯度裁剪可配置
- [x] P2（3 项）：pin_memory 条件启用、邻接向量化、训练/评估合并

</details>

---

## Phase 1：Group A Baselines（时序方法）

> **目标**：完成所有 channel-independent 时序方法，每个模型完成后立即运行 4 种实验设置。
> **依赖**：Phase 0 ✅
> **开发→运行→收集 循环进行**

### 1.0 HA ✅

- [x] 实现 + 注册
- [x] 评测：NYC→NYC, CHI→CHI（跨城市与同城相同，HA 无迁移概念）
- [x] 结果已录入 `results/RESULTS.md`

### 1.1 LSTM ✅ 代码就绪，待用户运行

> 文件：`model/lstm.py` | 实现方式：自主实现

- [x] 网络结构：2 层 LSTM (hidden=64, dropout=0.1) → Linear(64, P) → (B, P, 1)
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（CHI 3 epochs）
- [ ] **用户运行**：`python experiments/run_all.py --models LSTM --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

### 1.2 DLinear ✅ 代码就绪，待用户运行

> 文件：`model/dlinear.py` | 实现方式：自主实现

- [x] 网络结构：MovingAvg 分解 trend + seasonal → 两个 Linear(H, P) → 相加 → (B, P, 1)
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（CHI 3 epochs）
- [ ] **用户运行**：`python experiments/run_all.py --models DLinear --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

### 1.3 PatchTST ✅ 代码就绪，待用户运行

> 文件：`model/patchtst.py` | 实现方式：自主实现（参考原论文架构）

- [x] 网络结构：PatchEmbedding(patch_len=4, stride=2) → PositionalEncoding → TransformerEncoder(d_model=64, n_heads=4, n_layers=2, d_ff=128) → Linear → (B, P, 1)
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（CHI 3 epochs）
- [ ] **用户运行**：`python experiments/run_all.py --models PatchTST --settings all --epochs 100 --batch_size 64 --lr 1e-4`
- [ ] 结果收集 → 更新 RESULTS.md

### 1.4 TimesNet ✅ 代码就绪，待用户运行

> 文件：`model/timesnet.py` | 实现方式：基于 TSLib 官方代码忠实复现

- [x] 网络结构（忠实还原 TSLib 原版）：
  - TokenEmbedding (Conv1d k=3, circular padding, Kaiming init)
  - predict_linear: Linear(H, H+P) 在 feature 空间扩展时间维
  - TimesBlock × 2: FFT batch-level 周期检测 → 2D reshape → dual InceptionBlockV1 (6 kernels {1,3,5,7,9,11}, mean aggregation, Kaiming init) → reshape back + per-sample softmax 加权聚合 + residual
  - LayerNorm after each block
  - projection: Linear(d_model, 1) per timestep → 取最后 P 步
  - RevIN instance normalization (subtract mean/std → reverse after prediction)
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（CPU，合成数据 50 节点 2 epochs）
- [ ] **用户运行**：`python experiments/run_all.py --models TimesNet --settings all --epochs 100 --batch_size 16 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **注意**：TimesNet 因 PyTorch MPS 后端的 backward bug（多分支 Conv2d+view+residual 模式），自动回退至 CPU 训练。有 CUDA GPU 时会自动使用 GPU。CPU 训练较慢，预计单次训练 ~60-120 min。

---

## Phase 2：Group B Baselines（时空图方法）

> **目标**：完成 STGNN 方法，每个模型完成后立即运行 4 种实验设置。
> **依赖**：Phase 0 ✅（邻接矩阵已构建）
> **可与 Phase 1 并行推进**

### 2.0 邻接矩阵 ✅

- [x] `adj_NYC_corr_k10.npy` (820×820, 9608 edges)
- [x] `adj_CHI_corr_k10.npy` (585×585, 9696 edges)

### 2.1 STGCN ✅ 代码就绪，待用户运行

> 文件：`model/stgcn.py` | 实现方式：基于 hazdzz/STGCN PyTorch 版本忠实复现

- [x] 核心模块（忠实还原 hazdzz benchmark 版本）：
  - `scaled_laplacian()`: L̃ = L_sym - I，从邻接矩阵预计算
  - `ChebGraphConv`: K 阶 Chebyshev 多项式递推 T_0=I, T_1=L̃, T_k=2L̃T_{k-1}-T_{k-2}，可学习权重 (K, C_in, C_out)
  - `TemporalConvLayer`: Conv2d(kernel=(1,Kt)) + GLU 门控 (split + sigmoid)
  - `GraphConvLayer`: 1×1 align + ChebConv + residual + ReLU
  - `STConvBlock`: TemporalConv → GraphConv → TemporalConv → LayerNorm + Dropout
  - `OutputLayer`: TemporalConv(Ko→1) → FC(C→C) → ReLU → FC(C→P)
  - 默认配置：blocks=[[1,16,64],[64,16,64]], Ks=3, Kt=3, Ko=H-8=4
- [x] 继承 `TorchModel`，`input_mode="all_nodes"`
- [x] 构造函数接受 `adj` 邻接矩阵，注册时自动从 `config.adj_path()` 加载
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（合成数据 MPS + 真实 CHI 数据完整 evaluator 流程）
- [ ] **用户运行**：`python experiments/run_all.py --models STGCN --settings all --epochs 100 --batch_size 32 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **跨城市迁移**：图卷积核权重 (K, C_in, C_out) 与 N 无关，跨城市时 registry 自动加载 target 城市邻接矩阵。

### 2.2 DCRNN ✅ 代码就绪，待用户运行

> 文件：`model/dcrnn.py` | 实现方式：基于 liyaguang/DCRNN (TF) + chnsh/DCRNN_PyTorch 忠实复现

- [x] 核心模块（忠实还原原始架构）：
  - `build_supports()`: 从邻接矩阵计算 dual random walk 支撑矩阵 (forward + backward, K-hop powers)，共 2×(K+1) 个
  - `DiffusionConv`: Σ_k θ_k · P^k · X，所有支撑矩阵拼接后线性变换
  - `DCGRUCell`: GRU 三个门 (reset, update, candidate) 全部用 DiffusionConv 替代线性变换
  - `DCRNNEncoder`: L 层 DCGRU 堆叠，逐步处理 T 个时间步
  - `DCRNNDecoder`: L 层 DCGRU + Linear projection，自回归生成 P 步预测
  - 默认配置：hidden_dim=64, num_layers=2, K=2 (6 个支撑矩阵), 445K 参数
- [x] 继承 `TorchModel`，`input_mode="all_nodes"`
- [x] 注册到 `model/registry.py`，自动加载 target 城市邻接矩阵
- [x] Smoke test 通过（合成数据 MPS，forward+backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models DCRNN --settings all --epochs 100 --batch_size 32 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **性能说明**：DCRNN 是 Seq2Seq 架构（24 个顺序 RNN 步），MPS 单 batch (B=16, N=585) ~9s。预计单次训练 ~4-8 小时 (MPS)，建议用 GPU 服务器或减少 epochs。
> **跨城市迁移**：扩散卷积权重 (num_supports × C_in, C_out) 与 N 无关，支撑矩阵在 registry 中自动为 target 城市重建。

### 2.3 GWNET ✅ 代码就绪，待用户运行

> 文件：`model/gwnet.py` | 实现方式：基于 nnzhan/Graph-WaveNet 忠实复现
> 代码源：https://github.com/nnzhan/Graph-WaveNet

- [x] 核心模块（忠实还原原始架构）：
  - `asym_adj()`: 行归一化邻接矩阵 (random walk transition matrix)
  - `NConv`: 图卷积 via einsum `bcnt,nm->bcmt`
  - `MixProp`: Mix-hop propagation — 拼接 [x, Ax, A²x, ...] 对每个 support 矩阵，线性混合
  - Dilated Causal Convolution: Conv2d(kernel=(1,2)) with dilation=[1,2]×4 blocks → receptive field=13
  - Gated TCN: tanh(filter_conv) ⊙ sigmoid(gate_conv)
  - Adaptive Adjacency: learnable E₁(N,10) @ E₂(10,N) → softmax(ReLU(·))
  - Skip connections: 每层 skip 累加，最终 ReLU → end_conv1 → ReLU → end_conv2
  - 默认配置：residual_ch=32, dilation_ch=32, skip_ch=128, end_ch=256, blocks=4, layers=2, order=2, dropout=0.3
- [x] 继承 `TorchModel`，`input_mode="all_nodes"`
- [x] 构造函数接受 `adj` 邻接矩阵，支撑矩阵 (forward + backward RW) 注册为 buffer
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（合成数据 forward+backward 正确，NYC/CHI 尺寸验证）
- [ ] **用户运行**：`python experiments/run_all.py --models GWNET --settings all --epochs 100 --batch_size 32 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~170K (N-independent) + 自适应邻接 (NYC: +16K, CHI: +12K)
> **跨城市迁移**：MixProp 权重与 N 无关，自适应邻接 E₁/E₂ 为 N-dependent 但会随 target 城市重建。固定 supports 在 registry 中自动加载 target 城市邻接。

### 2.4 STGODE

> 文件：`model/stgode.py` | 实现方式：适配开源
> 代码源：https://github.com/square-coder/STGODE

- [ ] 图 ODE + 时间 ODE
- [ ] 继承 `TorchModel`，`input_mode="all_nodes"`
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models STGODE --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

---

## Phase 3：Group C Baselines（事件感知方法）

> **依赖**：Phase 0 ✅，部分依赖代码开源情况
> **可与 Phase 1-2 穿插推进**

### 3.1 ST-Norm

> 代码源：https://github.com/JLDeng/ST-Norm

- [ ] 获取代码，评估适配难度
- [ ] 选择 backbone + 适配
- [ ] **用户运行**：同城 + 跨城市 4 设置
- [ ] 结果收集 → 更新 RESULTS.md

### 3.2 ConFormer (ST11)

> 代码源：https://github.com/Dreamzz5/ConFormer ✅ 已确认对应
> 论文：*Towards Resilient Transportation: A Conditional Transformer for Accident-Informed Traffic Forecasting* (KDD 2026)

- [x] 代码仓库核对：README 确认为 Conditional Transformer + Guided Layer Normalization (GLN)，KDD 2026
- [x] 跨城市适配性评估：**存在 N-dependent 参数**
  - Node Embedding `(N, d)`、Adaptive Adjacency `(N, 10)×(10, N)`、precomputed adj `(N, N)`、output reshape 硬编码 `num_nodes`
  - 同城评测可直接使用，跨城市需去除或重初始化 node embedding + 替换 adj
- [ ] 数据格式适配：原始格式为 `data.npz (T, N, C)` + `adj.npy`，需转换我们的 `(T, N)` flow + corr adj
- [ ] 适配到框架 + 注册
- [ ] **用户运行**：同城 + 跨城市 4 设置
- [ ] 结果收集 → 更新 RESULTS.md

### 3.3 STEVE (ST13)

> 代码源：https://github.com/bigscity/STEVE_CODE ✅ 已确认对应
> 论文：*Seeing the Unseen: Learning Basis Confounder Representations for Robust Traffic Prediction* (KDD 2025)

- [x] 代码仓库核对：README 确认为 STEVE (Spatial-Temporal sElf-superVised confoundEr learning)，KDD 2025
- [x] 跨城市适配性评估：**存在 N-dependent 参数**
  - GNN backbone、node embeddings、graph-tied basis confounder bank、spatial self-supervised task
  - 同城评测可直接使用，跨城市需解耦 confounder learning 与 city-specific spatial 组件
- [ ] 数据格式适配：原始格式为 HDF5 + adjacency pickle（LibCity 生态），需转换
- [ ] 适配到框架 + 注册
- [ ] **用户运行**：同城 + 跨城市 4 设置
- [ ] 结果收集 → 更新 RESULTS.md

---

## Phase 4：本文模型设计与实现

> **依赖**：Phase 1-2 全部完成（需充分了解各 baseline 优劣）
> **本阶段在所有 baseline 完成后启动**

### 4.1 模型设计

- [ ] 基于 baseline 实验结论，确定创新点方向
- [ ] 撰写设计文档
- [ ] 确认方案

### 4.2 模型实现

- [ ] 实现核心模块
- [ ] 适配 `TorchModel` + 注册
- [ ] **用户运行**：4 种实验设置
- [ ] 结果收集 → 更新 RESULTS.md

### 4.3 消融实验

- [ ] 设计消融变体
- [ ] **用户运行**
- [ ] 分析各模块贡献

---

## Phase 5：综合分析与论文撰写

> **依赖**：Phase 1-4

### 5.1 敏感性分析

- [ ] 窗口大小：H/P ∈ {6, 12, 24}
- [ ] 异常阈值：anomaly_min_ratio ∈ {0.0, 0.25, 0.5, 0.75}
- [ ] 邻接矩阵 Top-K ∈ {5, 10, 20}

### 5.2 可视化

- [ ] 预测曲线对比图
- [ ] 跨城市迁移衰减柱状图
- [ ] 各方法 full/anomaly/normal 雷达图

### 5.3 论文撰写

- [ ] 实验设置、主实验表格、分析
- [ ] 消融实验
- [ ] Case Study

---

## 执行顺序

```
Phase 0 ✅
   │
   ├──→ Phase 1 (Group A, 每个模型: 开发→用户运行→收集结果)
   │       │
   ├──→ Phase 2 (Group B, 同上，可与 Phase 1 并行)
   │       │
   ├──→ Phase 3 (Group C, 穿插进行)
   │       │
   │       ├── 全部 baseline 完成
   │       ↓
   │    Phase 4 (本文模型)
   │       │
   └───────→ Phase 5 (分析 & 论文)
```

---

## 当前进度

- [x] **Phase 0**：全部完成（基础设施 + 代码质量审查 + 修复）
- [x] **HA**：全量评测完成
- [ ] **Phase 1**：LSTM → DLinear → PatchTST → TimesNet（待开发）
- [ ] **Phase 2**：STGCN → DCRNN → GWNET → STGODE（待开发）
- [ ] **Phase 3-5**：待开始
