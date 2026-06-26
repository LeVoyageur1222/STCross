# 运行指南

> 本文件说明如何运行各 baseline 模型的训练和评测。
> 每个模型开发完成后，按此指南运行 4 种实验设置，将结果保存到 `results/` 目录。
>
> 最后更新：2026-06-26（方案 C：离群值裁剪 + 每模型 LR + CosineAnnealingLR）

## 环境准备

```bash
cd /path/to/STCross
pip install -r requirements.txt
```

验证：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.isavailable()}, MPS={torch.backends.mps.is_available()}')"
```

## 核心优化（方案 C）

本框架已实施三项训练优化：

1. **离群值裁剪**：标准化后数据裁剪到 `[-10, 10]`，消除极端离群值（max=383.86）对 MSE 和梯度的主导。仅影响训练输入/目标和测试输入，**评测真值 (raw test Y) 不裁剪**，保证公平性。
2. **每模型 LR**：根据模型架构自动选择最优学习率（见下表）。用户也可通过 `--lr` 覆盖。
3. **CosineAnnealingLR**：默认 LR 调度器（`T_max=100, eta_min=1e-6`），平滑退火，与现代时序论文一致。可通过 `--lr_scheduler plateau` 切换回 ReduceLROnPlateau。

### 每模型推荐学习率

| 模型 | 推荐 LR | 理由 |
|------|--------|------|
| LSTM | **1e-2** | RNN，诊断显示 4x 更快收敛 |
| DLinear | 1e-3 | 微型 Linear (312 参数)，论文标准 |
| PatchTST | **1e-4** | Transformer，对大 LR 敏感 |
| TimesNet | 1e-3 | CNN+FFT，TSLib 默认 |
| TimeAlign | 1e-3 | Linear + alignment，中等 |
| ProtoTS | 1e-3 | MLP-Mixer + prototype，中等 |
| ModernTCN | 1e-3 | 现代 CNN，ICLR 标准 |
| EVL | 1e-3 | LSTM + Memory + dual loss |
| EALGAP | 1e-3 | SA + GRU，中等 |
| DAN | 1e-3 | CNN+LSTM，标准 |
| STSSDL | 1e-3 | LSTM + prototype + triplet |

> 不指定 `--lr` 时，批量运行器自动使用上表中的每模型默认值。

## 目录结构

```
results/
├── RESULTS.md                    # 结果汇总表（每完成一个模型更新）
├── {Model}_{Source}_to_{Target}_{timestamp}.json   # 单次实验详细结果
└── summary.md                    # 批量运行器自动生成的汇总

checkpoints/
└── {Model}_{Source}_best.pt      # 每城市训练的最佳 checkpoint（不互相覆盖）
```

---

## 方式一：批量运行器（推荐）

使用 `experiments/run_all.py` 一次性运行指定模型的全部实验设置。

**核心优化**：每个模型每个源城市只训练一次，复用模型评测同城+跨城市（4 设置 → 2 次训练）。

```bash
# 运行指定模型的全部实验（同城 + 跨城市），使用每模型默认 LR
python experiments/run_all.py --models LSTM --settings all

# 运行多个模型
python experiments/run_all.py --models LSTM DLinear PatchTST --settings all

# 运行所有已注册模型的全部实验
python experiments/run_all.py --settings all

# 覆盖 LR（对所有指定模型使用同一 LR）
python experiments/run_all.py --models LSTM --settings all --lr 1e-2

# 切换 LR 调度器为 ReduceLROnPlateau（默认是 cosine）
python experiments/run_all.py --models LSTM --settings all --lr_scheduler plateau

# 自定义训练参数
python experiments/run_all.py --models LSTM --settings all \
    --epochs 5000 --batch_size 64
```

**特性**：
- 每模型每源城市只训练 1 次（非 4 次），节省 50% 训练算力
- 自动使用每模型推荐 LR（不指定 `--lr` 时）
- 自动跳过已完成的实验（基于 `results/` 目录中的 JSON 文件 + config 校验）
- 单个模型失败不影响其他模型
- 运行完毕自动生成 `results/summary.md`
- Checkpoint 按源城市保存：`{Model}_{Source}_best.pt`（不互相覆盖）

**强制重跑**：使用 `--force` 参数跳过缓存检查：
```bash
python experiments/run_all.py --models LSTM --settings all --force
```

**如需重跑某个实验**，删除对应 JSON 文件后重新运行：
```bash
rm results/LSTM_NYC_to_NYC_*.json
python experiments/run_all.py --models LSTM --settings same-city
```

---

## 方式二：单模型运行

```bash
python experiments/run_model.py \
    --model LSTM \
    --source NYC --target NYC \
    --epochs 5000 --batch_size 64 --lr 1e-2 \
    --output_dir results/
```

> 注意：单模型运行不会自动应用每模型 LR 默认值，需手动指定 `--lr`。

---

## 各模型运行命令

### HA（已完成 ✅）

```bash
python experiments/run_all.py --models HA --settings all
```

> HA 无需训练，跨城市结果与同城相同（始终使用 target 训练集）。
> 也可单独运行：`python experiments/run_ha_baseline.py --output_dir results/`

### LSTM ✅ 已就绪

```bash
# 使用每模型默认 LR=1e-2（不指定 --lr）
python experiments/run_all.py --models LSTM --settings all

# 或显式指定
python experiments/run_all.py --models LSTM --settings all --lr 1e-2
```

> 模型结构：2 层 LSTM (hidden=64, dropout=0.1) → Linear → (B, P, 1)
> 参数量：~51K | 推荐 LR：1e-2（诊断显示 4x 更快收敛）
> checkpoint：`checkpoints/LSTM_NYC_best.pt`, `checkpoints/LSTM_CHI_best.pt`

### DLinear ✅ 已就绪

```bash
python experiments/run_all.py --models DLinear --settings all
```

> 模型结构：MovingAvg(kernel=3) 分解 trend/seasonal → 两个 Linear(H, P) → 相加
> 参数量：~312（极轻量）| 推荐 LR：1e-3
> 训练最快，通常 3-5 epoch 即收敛。

### PatchTST ✅ 已就绪

```bash
python experiments/run_all.py --models PatchTST --settings all
```

> 模型结构：PatchEmbedding(patch=4, stride=2) → Learnable PE → TransformerEncoder(d=64, heads=4, layers=2) → Linear → RevIN
> 参数量：~71K | 推荐 LR：1e-4（Transformer 对大 LR 敏感）
> H=12 产生 5 个 patch。

### TimesNet ✅ 已就绪

```bash
python experiments/run_all.py --models TimesNet --settings all --batch_size 16
```

> **重要**：TimesNet 因 PyTorch MPS backward bug 自动回退至 **CPU** 训练（有 CUDA 时自动使用 GPU）。
> 模型结构（忠实 TSLib 原版）：TokenEmbedding(Conv1d k=3) → predict_linear(H→H+P) → TimesBlock×2(FFT+2D Inception×6) → RevIN
> 参数量：~1.2M | 推荐 LR：1e-3 | batch_size=16（大模型需小 batch）
> 超参：d_model=32, d_ff=32, top_k=3, num_kernels=6, n_layers=2

### TimeAlign ✅ 已就绪

```bash
python experiments/run_all.py --models TimeAlign --settings all
```

> 模型结构（忠实 TROUBADOUR000/TimeAlign）：
> RevIN → PatchEncoder(非重叠 patch, M 层 Linear-GELU-Dropout-Linear + LN + residual) → Predict Branch + Reconstruct Branch(训练时, stop-gradient) → Distribution-Aware Alignment(local element-wise + global distance matrix + dynamic weighting)
> 参数量：~148K | 推荐 LR：1e-3 | 自定义 fit()（L1 loss + 重建 + 对齐）

### ProtoTS ✅ 已就绪

```bash
python experiments/run_all.py --models ProtoTS --settings all
```

> 模型结构（忠实 SKURA502/ProtoTS）：
> Patching → Bottleneck MLP-Mixer(2 层, d_bottle=16) → Time Aggregation → Prototype Similarity(Euclidean + softmax) → 原型模式加权和
> 参数量：~6K（超轻量）| 推荐 LR：1e-3 | 自定义 fit()（L1 + 熵正则化）

### ModernTCN ✅ 已就绪

```bash
python experiments/run_all.py --models ModernTCN --settings all
```

> 模型结构（忠实 luodhhh/ModernTCN CI 模式）：
> Patching → 2 × ModernTCNBlock(DWConv k=5 + ConvFFN d→2d→d + LN + residual) → Flatten → Linear head
> 参数量：~39K | 推荐 LR：1e-3 | 标准 Trainer（MSE）

### EVL ✅ 已就绪

```bash
python experiments/run_all.py --models EVL --settings all
```

> 模型结构（忠实 HuiqunHuang/EVL）：
> 2 层 LSTM encoder (mean of all outputs) → Memory Network(16 slots, attention retrieval) → dual heads (regression + extreme classifier)
> 参数量：~74K | 推荐 LR：1e-3 | 自定义 fit()（MSE + λ·EVL）
> 极端标签：数据集级 |y| > 90th percentile

### EALGAP ✅ 已就绪

```bash
python experiments/run_all.py --models EALGAP --settings all
```

> 模型结构（忠实 HuiqunHuang/EALGAP）：
> Global Trend(指数分布 PDF 生成 Q/K/V → SA) + Local Extreme(z-score + γ + tanh + GRU) → 乘法融合 output = ReLU(global × (1 + extreme))
> 参数量：~244K | 推荐 LR：1e-3 | 标准 Trainer（MSE）
> 已知近似：PDF 动态计算（非预计算）、instance z-score（非 same-slot）

### DAN ✅ 已就绪

```bash
python experiments/run_all.py --models DAN --settings all
```

> 模型结构（忠实 davidanastasiu/dan，CNN+LSTM backbone）：
> Dual Polar Channels (Far CNN+LSTM encoder/decoder + Near CNN+LSTM encoder/decoder) → RepMerg(concat h_far + y_near → FC) → y_final
> 参数量：~134K | 推荐 LR：1e-3 | 自定义 fit()（distance-weighted loss）

### STSSDL ✅ 已就绪

```bash
python experiments/run_all.py --models STSSDL --settings all
```

> 模型结构（ST-SSDL-lite：LSTM backbone 替代 GCRU）：
> Historical Anchoring(减均值) → LSTM encoder → Wq projection → Prototype Assignment(nearest) → concat(z, μ*) → FC → prediction
> 参数量：~64K | 推荐 LR：1e-3 | 自定义 fit()（L1 + α·triplet contrastive + β·deviation）

---

## 完成一个模型后的流程

1. **运行**：按上述命令运行全部 4 种实验设置（批量运行器自动训练 2 次）
2. **检查**：确认 `results/` 下生成了 4 个 JSON 文件
3. **通知我**：告知模型名称，我会：
   - 收集 JSON 结果
   - 更新 `results/RESULTS.md` 表格
   - 更新 `TODO.md` 进度

---

## 训练资源估算

| 模型组 | 模型 | 参数量 | 估算单次训练 (GPU) | 2 次训练合计 |
|--------|------|--------|-------------------|-------------|
| Group A | LSTM | 51K | ~5-10 min | ~10-20 min |
| Group A | DLinear | 312 | ~3-5 min | ~6-10 min |
| Group A | PatchTST | 71K | ~10-20 min | ~20-40 min |
| Group A | TimesNet | 1.2M | ~60-120 min (CPU) | ~2-4 h |
| Group C | TimeAlign | 148K | ~5-10 min | ~10-20 min |
| Group C | ProtoTS | 6K | ~3-5 min | ~6-10 min |
| Group C | ModernTCN | 39K | ~3-5 min | ~6-10 min |
| Group D | EVL | 74K | ~5-10 min | ~10-20 min |
| Group D | EALGAP | 244K | ~5-10 min | ~10-20 min |
| Group D | DAN | 134K | ~5-10 min | ~10-20 min |
| Group D | STSSDL | 64K | ~5-10 min | ~10-20 min |

> 优化后每模型只需 2 次训练（NYC + CHI），而非 4 次。
> 以上为粗略估算，实际取决于 early stopping 触发时间。

---

## 故障排查

**OOM（内存不足）**：减小 `--batch_size`（如 64→32→16）

**训练无收敛（val_loss 不降）**：
- 检查 LR 是否合适（LSTM 用 1e-2，Transformer 用 1e-4）
- 检查 `--lr_scheduler` 设置（cosine 默认）
- 离群值裁剪已自动启用，不应出现 loss=22+ 的极端情况

**NaN/Inf 损失**：训练中自动检测 `torch.isnan(loss)` 和 `torch.isinf(loss)`，遇到时跳过该 batch 并继续训练

**PerNodeStandardScaler 警告**："N/820 nodes have near-zero std" 是正常现象（部分节点流量恒定），已自动设 std=1.0，不影响训练

**跨城市结果异常**：所有模型均为 N-independent（per_node），无跨城市维度问题

**跳过逻辑误判**：删除 `results/` 下对应的 JSON 文件强制重跑，或使用 `--force`

**Checkpoint 覆盖**：checkpoint 按源城市保存（`{Model}_{Source}_best.pt`），不会互相覆盖
