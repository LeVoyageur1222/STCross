# 运行指南

> 本文件说明如何运行各 baseline 模型的训练和评测。
> 每个模型开发完成后，按此指南运行 4 种实验设置，将结果保存到 `results/` 目录。
>
> 最后更新：2026-06-24

## 环境准备

```bash
cd /path/to/STCross
pip install -r requirements.txt
```

验证：
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}, MPS={torch.backends.mps.is_available()}')"
```

## 目录结构

```
results/
├── RESULTS.md                    # 结果汇总表（每完成一个模型更新）
├── {Model}_{Source}_to_{Target}_{timestamp}.json   # 单次实验详细结果
└── summary.md                    # 批量运行器自动生成的汇总
```

---

## 方式一：批量运行器（推荐）

使用 `experiments/run_all.py` 一次性运行指定模型的全部实验设置。

```bash
# 运行指定模型的同城实验
python experiments/run_all.py --models HA LSTM --settings same-city

# 运行指定模型的全部实验（同城 + 跨城市）
python experiments/run_all.py --models LSTM --settings all

# 自定义训练参数
python experiments/run_all.py --models LSTM --settings all \
    --epochs 100 --batch_size 64 --lr 1e-3

# 运行所有已注册模型的全部实验
python experiments/run_all.py --settings all
```

**特性**：
- 自动跳过已完成的实验（基于 `results/` 目录中的 JSON 文件 + config 校验）
- 单个模型失败不影响其他模型
- 运行完毕自动生成 `results/summary.md`

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
    --epochs 100 --batch_size 64 --lr 1e-3 \
    --output_dir results/
```

---

## 各模型运行命令

### HA（已完成 ✅）

```bash
python experiments/run_all.py --models HA --settings all
```

> HA 无需训练，跨城市结果与同城相同（始终使用 target 训练集）。

### LSTM ✅ 已就绪

```bash
python experiments/run_all.py --models LSTM --settings all \
    --epochs 100 --batch_size 64 --lr 1e-3
```

> 预计 MPS 单次训练 ~5-10 min，4 设置合计 ~20-40 min。
> 模型结构：2 层 LSTM (hidden=64, dropout=0.1) → Linear → (B, P, 1)
> checkpoint 保存在 `checkpoints/LSTM_best.pt`，每次新设置会覆盖。

### DLinear ✅ 已就绪

```bash
python experiments/run_all.py --models DLinear --settings all \
    --epochs 100 --batch_size 64 --lr 1e-3
```

> 预计 MPS 单次训练 ~3-5 min，4 设置合计 ~12-20 min。
> 模型结构：MovingAvg(kernel=3) 分解 trend/seasonal → 两个 Linear(H, P) → 相加
> 参数量极少（仅 2×H×P = 288 参数），训练最快的 baseline。

### PatchTST ✅ 已就绪

```bash
python experiments/run_all.py --models PatchTST --settings all \
    --epochs 100 --batch_size 64 --lr 1e-4
```

> **注意**：PatchTST 使用较小学习率 `1e-4`（Transformer 对大 lr 敏感）。
> 预计 MPS 单次训练 ~10-20 min，4 设置合计 ~40-80 min。
> 模型结构：PatchEmbedding(patch=4, stride=2) → TransformerEncoder(d=64, heads=4, layers=2) → Linear
> H=12 产生 (12-4)/2+1=5 个 patch。

### TimesNet ✅ 已就绪

```bash
python experiments/run_all.py --models TimesNet --settings all \
    --epochs 100 --batch_size 16 --lr 1e-3
```

> **重要**：TimesNet 因 PyTorch MPS backward bug 自动回退至 **CPU** 训练（有 CUDA 时自动使用 GPU）。
> CPU 训练较慢，预计单次 ~60-120 min，4 设置合计 ~4-8 小时。
> 建议优先跑 LSTM/DLinear/PatchTST，TimesNet 放到最后或用 GPU 服务器跑。
> 模型结构（忠实 TSLib 原版）：TokenEmbedding(Conv1d k=3) → predict_linear(H→H+P) → TimesBlock×2(FFT+2D Inception×6) → RevIN
> 超参：d_model=32, d_ff=32, top_k=3, num_kernels=6, n_layers=2

### STGCN ✅ 已就绪

```bash
python experiments/run_all.py --models STGCN --settings all \
    --epochs 100 --batch_size 32 --lr 1e-3
```

> 预计 MPS 单次训练 ~15-30 min，4 设置合计 ~60-120 min。
> 模型结构（忠实 hazdzz benchmark 版本）：2 × STConvBlock (TemporalConv-GLU → ChebGraphConv(K=3) → TemporalConv-GLU → LayerNorm) → OutputLayer(TemporalConv → FC → FC)
> 默认 blocks=[[1,16,64],[64,16,64]], Ks=3, Kt=3
> 邻接矩阵自动从 `data/processed_data/adjacency/` 加载（跨城市时加载 target 城市的）。

### DCRNN

```bash
python experiments/run_all.py --models DCRNN --settings all \
    --epochs 100 --batch_size 32 --lr 1e-3
```

### GWNET

```bash
python experiments/run_all.py --models GWNET --settings all \
    --epochs 100 --batch_size 32 --lr 1e-3
```

### STGODE

```bash
python experiments/run_all.py --models STGODE --settings all \
    --epochs 100 --batch_size 32 --lr 1e-3
```

---

## 完成一个模型后的流程

1. **运行**：按上述命令运行全部 4 种实验设置
2. **检查**：确认 `results/` 下生成了 4 个 JSON 文件
3. **通知我**：告知模型名称，我会：
   - 收集 JSON 结果
   - 更新 `results/RESULTS.md` 表格
   - 更新 `TODO.md` 进度

---

## 训练资源估算

| 模型组 | 模型 | 估算单次训练时长 (MPS) | 4 设置合计 |
|--------|------|------------------------|-----------|
| Group A | LSTM | ~5-10 min | ~20-40 min |
| Group A | DLinear | ~3-5 min | ~12-20 min |
| Group A | PatchTST | ~10-20 min | ~40-80 min |
| Group A | TimesNet | ~15-30 min | ~60-120 min |
| Group B | STGCN | ~15-30 min | ~60-120 min |
| Group B | DCRNN | ~20-40 min | ~80-160 min |
| Group B | GWNET | ~20-40 min | ~80-160 min |
| Group B | STGODE | ~20-40 min | ~80-160 min |

> 以上为粗略估算，实际取决于 early stopping 触发时间。
> Group B 模型因涉及图卷积 + 全节点输入 (batch, H, N)，显存/内存占用更大。

---

## 故障排查

**OOM（内存不足）**：减小 `--batch_size`（如 32→16→8）

**训练无收敛（val_loss 不降）**：降低 `--lr`（如 1e-3→1e-4），或检查模型输出维度

**跨城市结果异常**：确认模型的图卷积核参数与 N 无关（Group A 无此问题）

**跳过逻辑误判**：删除 `results/` 下对应的 JSON 文件强制重跑
