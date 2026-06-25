# STCross 实验规划

> 本文件为 STCross 项目的实验主规划，所有后续实验开发均以此为准。
> 相关文档：[运行指南](RUN_GUIDE.md) | [结果汇总](results/RESULTS.md)
>
> 最后更新：2026-06-25（Phase 1-3 完成 + Phase 4 极端预测方法规划）

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

### 2.4 STGODE ✅ 代码就绪，待用户运行

> 文件：`model/stgode.py` | 实现方式：基于 square-coder/STGODE 忠实复现
> 代码源：https://github.com/square-coder/STGODE

- [x] 核心模块（忠实还原原始架构）：
  - `gcn_norm_adj()`: GCN 对称归一化 + self-loop (D̃^{-1/2} Ã D̃^{-1/2})
  - `ODEFunc`: ODE 动力学函数 f(H,A) = A·H·W + H⊙d (W 初始化为 I, d 初始化为 1)
  - `ODEG`: Euler 方法 ODE 图演化, 6 步积分 (ode_time=6.0, step_size=1.0)
  - `TemporalConv`: 双层 TCN (CausalConv1d, kernel=3, dilation=[1,2]) + residual
  - 双 ODEG 分支 (spatial + semantic)，独立权重，共享邻接
  - 输出：Concat(2×d_model) → Flatten(H×2C) → FC(256) → FC(P)
  - 默认配置：d_model=64, ode_time=6.0, step_size=1.0, dropout=0.1
- [x] 继承 `TorchModel`，`input_mode="all_nodes"`
- [x] 构造函数接受 `adj` 邻接矩阵，GCN 归一化后注册为 buffer
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（合成数据 forward+backward 正确，NYC/CHI 尺寸验证）
- [ ] **用户运行**：`python experiments/run_all.py --models STGODE --settings all --epochs 100 --batch_size 32 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~418K (N-independent), buffer: NYC 2.7MB, CHI 1.4MB
> **跨城市迁移**：所有可学习参数 (W, d, TCN, FC) 均与 N 无关，仅邻接 buffer 随 target 城市变化。
> **原始论文使用双邻接**：A_sp (空间) + A_se (语义/DTW)。本实现使用相同的 correlation 邻接但独立 ODE 动力学权重，两个分支学习不同空间模式。

---

## Phase 3：Group C Baselines（分布鲁棒/可迁移方法）

> **目标**：完成针对分布偏移鲁棒的方法，每个模型完成后立即运行 4 种实验设置。
> **依赖**：Phase 0 ✅
> **可与 Phase 1-2 穿插推进**
>
> **主题调整说明**：原 Group C 定位为"事件感知方法"（ST-Norm / ConFormer / STEVE），
> 但经细致考察发现 **三者均包含 N-dependent 可学习参数**（per-node 归一化系数 / 节点嵌入 / 自适应邻接），
> 无法直接用于跨城市迁移（source 城市 N₁ ≠ target 城市 N₂ 时参数维度不匹配）。
> 进一步排查 ST01(TGDL)、ST07(HyperD)、ST10(IGSTGNN) 等候选，均存在同样问题。
>
> 因此将 Group C 重新定义为 **"分布鲁棒/可迁移方法"**，理由：
> 1. 跨城市迁移本质上是分布偏移问题（源城市分布 → 目标城市分布）
> 2. 异常时段（节假日/极端天气）也是相对正常模式的分布偏移
> 3. 该组方法从时序层面显式处理分布不匹配，与 Group A（基础时序）、Group B（时空图）形成互补
>
> **N-dependent 排查汇总**：
>
> | 原候选 | N-dependent 组件 | 可迁移性 |
> |--------|-----------------|---------|
> | ST-Norm | SN 模块 per-node γ/β | ✗ |
> | ConFormer (ST11) | NodeEmb(N,d), AdaptiveAdj(N,10)×(10,N) | ✗ |
> | STEVE (ST13) | NodeEmb, confounder bank, spatial SSL | ✗ |
> | HyperD (ST07) | 周期嵌入 P_D(288,N), P_W(2016,N) | ✗ |
> | TGDL (ST01) | 子图掩码 K×N×N | ✗ |
> | IGSTGNN (ST10) | 自适应节点嵌入 E₁(N,d), E₂(N,d) | ✗ |
> | **TimeAlign (ST05)** | **无** — CI linear | **✓** |
> | **ProtoTS (ST06)** | **无** — shared prototypes | **✓** |
> | **ModernTCN** | **无** — CI DWConv | **✓** |

### 3.1 TimeAlign (ST05) ✅ 代码就绪，待用户运行

> 论文：*Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting* (ICLR 2026)
> 代码源：https://github.com/TROUBADOUR000/TimeAlign

- [x] 核心模块（忠实还原原始架构）：
  - `PatchEncoder`: Patching(patch_len=4, stride=2, 5 patches) → Linear embed → M 层 (Linear-GELU-Linear + residual)
  - Predict Branch: PatchEncoder(history) → Linear head → (B, P, 1)
  - Reconstruct Branch (仅训练): PatchEncoder(future) → Linear head → (B, P)，重建分支 stop-gradient
  - Local Alignment: patch 级余弦相似度 + margin δ_loc + GELU 软阈值
  - Global Alignment: 距离矩阵匹配 |D_x - D_y| + margin δ_glo + GELU
  - Dynamic Weighting: α/β 自适应平衡 local/global 损失
  - 总损失: L_pred + L_recon + λ · L_align
  - 默认配置: d_model=64, n_layers=2, patch_len=4, stride=2, λ=1.0, δ_loc=0.1, δ_glo=0.1
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 自定义 `fit()` 方法（重建分支需要训练时访问 Y）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（inference + training forward + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models TimeAlign --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~50K (N-independent)，所有参数为共享 Linear 层，天然支持跨城市迁移
> **特色**：重建分支保留高频突变信号（对异常客流预测关键），对齐机制处理 train/test 分布不匹配

### 3.2 ProtoTS (ST06) ✅ 代码就绪，待用户运行

> 论文：*ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting* (ICLR 2026)
> 代码源：https://github.com/SKURA502/ProtoTS

- [x] 核心模块（忠实还原原始架构）：
  - `BottleneckMixerLayer`: MLP-Mixer with bottleneck (feature + time mixing, d_bottle=16)
  - Input: Patching(patch_len=4, stride=2, 5 patches) → Linear embed → 2 层 BottleneckMixer
  - Time Aggregation: (B, n_patches, d) → (B, d) via Linear
  - Prototype Module: K 个原型，每个含嵌入 μ(K,d) + 时序模式 p(K,P)
  - Similarity: Euclidean dist(z, μ_i) → softmax(-dist) → weights
  - Prediction: Ŷ = Σ w_i · p_i（原型模式加权和）
  - Loss: L1 + λ·H(w)（熵最小化 → 稀疏原型激活 → 可解释性）
  - 默认配置：d_model=64, n_protos=8, n_mixer_layers=2, d_bottleneck=16, λ_ent=0.01
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 自定义 `fit()` 方法（L1 + entropy regularization 替代标准 MSE）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（encode/predict/weights/backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models ProtoTS --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~6K (超轻量级, N-independent)，所有参数为共享 Linear 层 + 原型嵌入/模式
> **特色**：每个原型对应一个完整的预测曲线模式，可学习正常/异常的不同时序原型；entropy 正则化确保解释简洁（少数原型主导）

### 3.3 ModernTCN ✅ 代码就绪，待用户运行

> 论文：*ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis* (ICLR 2024 Spotlight, 分数 8/8/8)
> 代码源：https://github.com/luodhhh/ModernTCN

- [x] 核心模块（忠实还原原始架构 CI 模式）：
  - `ModernTCNBlock`: ConvNeXt 风格设计
    - LN → DWConv(kernel=5, groups=d, 大感受野) → residual
    - LN → ConvFFN(Linear d→d_ff→d, GELU, Dropout) → residual
  - Patching(patch_len=4, stride=2, 5 patches) → Linear embed
  - 2 层 ModernTCNBlock → Final LN → Flatten → Linear head → (B, P, 1)
  - 默认配置：d_model=64, d_ff=128, kernel=5, n_layers=2, dropout=0.1
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 使用标准 `Trainer` 训练（MSE loss，无需自定义 fit）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（forward + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models ModernTCN --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~39K (N-independent)，纯卷积架构，channel-independent → 天然支持跨城市迁移
> **特色**：ICLR 2024 Spotlight，证明"纯卷积可以在所有 5 类时序分析任务上达到一致 SOTA"，大 kernel DWConv 是核心创新

---

## Phase 4：Group D Baselines（极端/异常事件预测方法）

> **目标**：完成专门针对极端/异常事件的时序预测方法，每个模型完成后立即运行 4 种实验设置。
> **依赖**：Phase 0 ✅
> **可与 Phase 1-3 穿插推进**
>
> **主题说明**：该组方法专注于**极端事件建模**——在预测任务中显式处理异常值（极端天气/节假日/突发事件）对时序模式的影响。
> 与 Group C（分布鲁棒/可迁移）互补：Group C 从分布层面处理 train/test 不匹配，Group D 从极值建模层面处理异常事件的预测失准。
>
> **方法来源**：基于 `papers/09_Extreme_Traffic_Flow_Prediction.pdf` 调研的 13 篇极端预测论文，
> 经 N-dependency 排查 + 代码可用性验证后选出以下 3 个候选。详见 `papers/ST00_note.md` 极端预测部分。
>
> **选择标准**：
> 1. 可用于跨城市交通客流预测 ✓
> 2. 不含 N-dependent 参数（节点嵌入、自适应邻接等） ✓
> 3. 有开源代码且仓库含实际代码内容 ✓
> 4. 经典论文 + 顶会发表 ✓

### 4.1 EVL（Modeling Extreme Events in Time Series Prediction） ✅ 代码就绪，待用户运行

> 论文：*Modeling Extreme Events in Time Series Prediction* (KDD 2019) — **经典奠基性工作**
> 代码源：https://github.com/HuiqunHuang/EVL

- [x] 核心模块（忠实还原原始架构）：
  - `ExtremeValueLoss`: 基于 EVT 的损失函数，尾部权重 `(1-u/γ)^γ`，γ=3，自动类平衡 β0/β1
  - `MemoryNetwork`: 存储 M=16 个极端事件模式原型，attention-based 检索
  - LSTM backbone: 2 层 LSTM (hidden=64, dropout=0.1)
  - Prediction head: concat(h, memory_out) → FC → ŷ (regression)
  - Extreme classifier: concat(h, memory_out) → FC → sigmoid → P(extreme)
  - 极端标签自动生成：|y| > 90th percentile → extreme=1
  - 总损失：MSE(ŷ, y) + λ · EVL(P(extreme), extreme_label)
  - 默认配置：hidden=64, mem_size=16, λ_evl=0.1, γ=3.0, extreme_pct=90
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 自定义 `fit()` 方法（dual loss: MSE + EVL）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（EVL loss + encode + predict + memory + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models EVL --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~74K (N-independent)，LSTM + Memory + dual heads
> **经典性**：首篇系统研究 DL 极端事件预测失败原因的工作（KDD 2019），EVL 损失被后续多篇论文引用
> **EVL 核心洞察**：MSE 等价于高斯核密度估计→尾部指数衰减→无法恢复厚尾→极值欠拟合；EVL 通过 EVT 尾部权重修正此问题

### 4.2 EALGAP（Extreme-Aware Local-Global Attention） ✅ 代码就绪，待用户运行

> 论文：*Extreme-Aware Local-Global Attention for Spatio-Temporal Urban Mobility Learning* (ICDE 2023)
> 代码源：https://github.com/HuiqunHuang/EALGAP

- [x] 核心模块（忠实还原原始架构）：
  - `GlobalTrendModule`: 输入嵌入 → Multi-Head Self-Attention(4 heads) → LayerNorm + residual → 捕获全局周期/趋势模式
  - `LocalExtremeModule`: 计算运行 z-score 偏差 → 嵌入 → GRU → 预测极端程度
  - Global head: flatten(H×d) → FC(256) → ReLU → FC(P) → 全局趋势预测
  - Local head: GRU output → FC(H→P) → 局部极端度预测
  - Gated Fusion: concat(global, local) → sigmoid gate → 加权融合 → ReLU → output
  - 默认配置：d_model=64, n_heads=4, gru_hidden=32, dropout=0.1
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 使用标准 `Trainer`（MSE loss），无需自定义 fit
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（global + local + fusion + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models EALGAP --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~224K (N-independent)，SA + GRU + FC 全共享
> **直接相关**：原始实验使用 NYC/CHI Bike + Taxi 数据，与本项目数据场景高度匹配
> **核心思想**：全局趋势捕捉正常周期模式，局部极端度捕捉异常偏离程度，门控融合自适应平衡两者

### 4.3 DAN（Distance-weighted Auto-regularized Network） ✅ 代码就绪，待用户运行

> 论文：*Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting* (AAAI 2024)
> 代码源：https://github.com/davidanastasiu/dan

- [x] 核心模块（忠实还原原始架构）：
  - `PolarEncoder/Decoder`: 双通道 FC 编解码器（far + near）
  - RepGen: Far encoder→decoder 产生 y_far; Near encoder→decoder 产生 y_near
  - RepMerg: concat(h_far, y_near) → FC → y_final（融合极端隐变量 + 正常预测）
  - Distance-weighted loss（数据中心化到 mean≈0 后）：
    - w_far = tanh(y)² — 极端值高权重
    - w_near = (1-|tanh(y)|)² — 正常值高权重
    - L = w_far·MSE(y_far,y) + w_near·MSE(y_near,y) + MSE(y_final,y)
  - 默认配置：d_model=64, dropout=0.1
- [x] 继承 `TorchModel`，`input_mode="per_node"`
- [x] 自定义 `fit()` 方法（distance-weighted dual-channel loss）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（dual forward + distance weights + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models DAN --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~26K (超轻量级, N-independent)，per-series 双通道 FC
> **核心优雅设计**：距离加权自动平衡极端/正常样本——无需手动设阈值或采样策略
> **注意**：原始应用为水文径流预测，但远/近极性分解 + 距离加权损失是通用极端值处理技术

### 4.4 ST-SSDL（Self-Supervised Deviation Learning） ✅ 代码就绪，待用户运行

> 论文：*How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning* (NeurIPS 2025)
> 代码源：https://github.com/Jimmy-7664/ST-SSDL

- [x] 核心模块（适配原始架构为 per-node 模式）：
  - Historical Anchoring: x_dev = x - mean(x)，提取偏差信号
  - Encoder: Patching(4,2) → Linear embed → FC(n_patches×d, d) → GELU → FC(d, d) → z ∈ R^d
  - Learnable Prototypes: K=8 个全局共享原型 μ ∈ R^(K, d)
  - Prototype Assignment: argmin_k ||z - μ_k|| → 找到最近原型
  - Prediction head: concat(z, μ_nearest) → FC → ŷ (加回 x_mean 恢复尺度)
  - Triple loss:
    - L_pred = MSE(ŷ, y)
    - L_contrast = -mean(pairwise prototype distances) → 推开原型增强判别性
    - L_deviation = MSE(||z - μ*||, |ŷ - y|_mean) → 距离≈误差一致性
    - L_total = L_pred + α·L_contrast + β·L_deviation
  - 默认配置：d_model=64, n_protos=8, α=0.01, β=0.1, patch_len=4, stride=2
- [x] 继承 `TorchModel`，`input_mode="per_node"`（替换 STGNN backbone 为 per-node encoder）
- [x] 自定义 `fit()` 方法（triple loss）
- [x] 注册到 `model/registry.py`
- [x] Smoke test 通过（anchor + encode + assign + contrastive + deviation + backward 正确）
- [ ] **用户运行**：`python experiments/run_all.py --models STSSDL --settings all --epochs 100 --batch_size 64 --lr 1e-3`
- [ ] 结果收集 → 更新 RESULTS.md

> **参数量**：~35K (N-independent)，原型 + encoder 均为共享参数
> **顶会最新**：NeurIPS 2025（东京大学 + 丰田汽车）
> **核心洞察**：偏差学习天然对应异常预测——高偏差(远离原型) = 异常时段，低偏差(靠近原型) = 正常时段
> **适配说明**：原始论文使用 STGNN backbone（可能含 N-dependent 组件），本实现替换为 per-node Linear encoder 以确保完全 N-independent

---

## Phase 5：本文模型设计与实现

> **依赖**：Phase 1-4 全部完成（需充分了解各 baseline 优劣）
> **本阶段在所有 baseline 完成后启动**

### 5.1 模型设计

- [ ] 基于 baseline 实验结论，确定创新点方向
- [ ] 撰写设计文档
- [ ] 确认方案

### 5.2 模型实现

- [ ] 实现核心模块
- [ ] 适配 `TorchModel` + 注册
- [ ] **用户运行**：4 种实验设置
- [ ] 结果收集 → 更新 RESULTS.md

### 5.3 消融实验

- [ ] 设计消融变体
- [ ] **用户运行**
- [ ] 分析各模块贡献

---

## Phase 6：综合分析与论文撰写

> **依赖**：Phase 1-5

### 6.1 敏感性分析

- [ ] 窗口大小：H/P ∈ {6, 12, 24}
- [ ] 异常阈值：anomaly_min_ratio ∈ {0.0, 0.25, 0.5, 0.75}
- [ ] 邻接矩阵 Top-K ∈ {5, 10, 20}

### 6.2 可视化

- [ ] 预测曲线对比图
- [ ] 跨城市迁移衰减柱状图
- [ ] 各方法 full/anomaly/normal 雷达图

### 6.3 论文撰写

- [ ] 实验设置、主实验表格、分析
- [ ] 消融实验
- [ ] Case Study

---

## 执行顺序

```
Phase 0 ✅
   │
   ├──→ Phase 1 (Group A 时序方法, 每个模型: 开发→用户运行→收集结果)
   │       │
   ├──→ Phase 2 (Group B 时空图方法, 同上，可与 Phase 1 并行)
   │       │
   ├──→ Phase 3 (Group C 分布鲁棒方法, 穿插进行)
   │       │
   ├──→ Phase 4 (Group D 极端事件方法, 穿插进行)
   │       │
   │       ├── 全部 baseline 完成
   │       ↓
   │    Phase 5 (本文模型)
   │       │
   └───────→ Phase 6 (分析 & 论文)
```

---

## 当前进度

- [x] **Phase 0**：全部完成（基础设施 + 代码质量审查 + 修复）
- [x] **HA**：全量评测完成
- [x] **Phase 1**：LSTM ✅ / DLinear ✅ / PatchTST ✅ / TimesNet ✅（代码就绪，待用户运行）
- [x] **Phase 2**：STGCN ✅ / DCRNN ✅ / GWNET ✅ / STGODE ✅（代码就绪，待用户运行）
- [x] **Phase 3**：TimeAlign ✅ / ProtoTS ✅ / ModernTCN ✅（代码就绪，待用户运行）
- [x] **Phase 4**：EVL ✅ / EALGAP ✅ / DAN ✅ / ST-SSDL ✅（代码就绪，待用户运行）
- [ ] **Phase 5-6**：待开始
