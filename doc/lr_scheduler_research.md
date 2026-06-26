# 学习率调度策略调研报告

> 调研日期: 2026-06-26
> 状态: **on hold** — 调研完成，等待下一步指令

---

## 1. 当前策略分析

### 1.1 当前实现

所有 6 个训练路径（标准 `Trainer` + 5 个自定义 `fit`: EVL/TimeAlign/ProtoTS/DAN/STSSDL）均使用：

```python
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min",
    patience=10,        # val_loss 10 epoch 无改善则降 LR
    factor=0.5,         # 每次降至 50%
    min_lr=1e-6
)
```

### 1.2 行为特征

- **触发条件**: 验证集 loss 在 `patience` 个 epoch 内未下降
- **衰减方式**: 乘法衰减 `lr *= factor` (0.5)
- **下限**: 衰减到 `min_lr=1e-6` 后停止
- **更新频率**: 每个 epoch 一次，基于 val_loss

### 1.3 当前配置下的行为推演

设 `epochs=5000`, `early_stop_patience=30`, `lr_patience=10`, `lr=1e-3`, `factor=0.5`:

| 时间线 | 事件 | LR |
|--------|------|-----|
| Epoch 1-10 | val_loss 持续下降 | 1e-3 |
| Epoch 11-20 | val_loss 停滞 | 1e-3 → 5e-4 (epoch 21) |
| Epoch 21-30 | val_loss 仍停滞 | 5e-4 → 2.5e-4 (epoch 31) |
| Epoch 31-40 | 继续停滞 | 2.5e-4 → 1.25e-4 |
| ... | 持续衰减 | ... |
| ~Epoch 200 | LR 到达 1e-6 | 1e-6 (不再降) |
| Epoch 230 | 早停触发 (30 epoch 无改善) | 1e-6 |

**问题**: 如果模型在 LR=5e-4 时本可以继续优化，但被过早衰减到 1e-6 后失去优化能力。反之，如果模型在 LR=1e-3 时不稳定，10 epoch 的 patience 可能太短就触发了衰减。

### 1.4 当前策略的优缺点

**优点**:
- 自适应验证集表现，不需要预知训练长度
- 实现简单，PyTorch 原生支持
- 对传统时空图模型 (STGCN/DCRNN/GWNET) 效果良好

**缺点**:
- 对 val_loss 噪声敏感（单次波动可能触发衰减）
- 衰减后不可逆（LR 只能降不能升）
- 可能过早降到 min_lr，后期训练失去优化能力
- 不符合现代时序论文 (PatchTST/TimesNet/iTransformer) 的标准实践

---

## 2. 候选 LR 策略调研

### 2.1 CosineAnnealingLR (余弦退火)

**公式**: `lr(t) = min_lr + 0.5 * (lr - min_lr) * (1 + cos(π * t / T_max))`

- `T_max`: 总 epoch 数 (通常 = max_epochs)
- 平滑从初始 LR 退火到 min_lr

**论文实践**:
- **TimesNet** (ICLR 2023, TSLib): ✅ 使用 `CosineAnnealingLR(optimizer, T_max=epochs)`
- **PatchTST** (ICLR 2023): ✅ Cosine + linear warmup
- **iTransformer** (ICLR 2024): ✅ Cosine
- **ModernTCN** (ICLR 2024): ✅ 类似

**优点**:
- 平滑衰减，不会突然降低 LR
- 保底到达 min_lr，训练后期有微调能力
- 符合现代时序论文标准，baseline 复现更忠实

**缺点**:
- 不自适应训练动态（固定退火曲线）
- 需要预知 T_max（但我们的 epochs=5000 已知）
- 早停时 LR 可能还没退完（不是问题，早停就是提前结束）

**集成难度**: ⭐ 极低
- PyTorch 原生: `CosineAnnealingLR(optimizer, T_max=cfg.epochs)`
- 与现有 `scheduler.step()` 接口兼容（epoch 级更新）
- 不需要传 val_loss

### 2.2 CosineAnnealingWarmRestarts (余弦退火 + 热重启)

**公式**: 周期性余弦退火，每个周期结束时 LR 重置到初始值

- `T_0`: 第一个周期长度
- `T_mult`: 周期倍增因子

**论文实践**:
- SGDR (Stochastic Gradient Descent with Warm Restarts, ICLR 2017)
- 部分语音/图像任务使用

**优点**:
- 热重启可逃出局部最优
- 适合长训练（我们的 5000 epoch）

**缺点**:
- 重启时 LR 突然升高，可能破坏已学到的特征
- 超参敏感（T_0, T_mult 需调）
- 时序预测论文较少使用

**集成难度**: ⭐⭐ 低
- PyTorch 原生支持
- 但需调 T_0, T_mult

### 2.3 OneCycleLR (超级收敛)

**公式**: 先 warmup 到 max_lr，再退火到 min_lr

- 两阶段：warmup (低→高) + anneal (高→低)
- 需要总步数 (epochs × batches_per_epoch)

**论文实践**:
- Super-Convergence (Smith & Topin 2019)
- 视觉任务常用

**优点**:
- 超级收敛，训练速度快
- 短训练 (50-100 epoch) 效果显著

**缺点**:
- 需要 step 级更新（每个 batch），不是 epoch 级
- 超参敏感 (max_lr, pct_start)
- 我们训练 5000 epoch + 早停，OneCycleLR 设计用于固定短训练，不太匹配

**集成难度**: ⭐⭐⭐ 中
- 需要 step 级更新，改 TrainingLoop 的调用位置
- 需要总步数参数

### 2.4 Linear Warmup + Cosine Decay (Transformer 标准)

**公式**:
- 前 `warmup_steps`: `lr = lr * step / warmup_steps` (线性升温)
- 之后: `lr = min_lr + 0.5 * (lr - min_lr) * (1 + cos(π * (step - warmup) / (total - warmup)))`

**论文实践**:
- **PatchTST** (ICLR 2023): ✅ 使用此策略
- BERT, ViT, 大多数 Transformer
- TSLib 的部分模型

**优点**:
- Warmup 防止 Transformer 早期训练不稳定
- Cosine decay 平滑退火
- 最符合现代 Transformer-based baseline 的复现标准

**缺点**:
- 需要 step 级更新
- 需要预设 warmup 比例 (通常 5-10% 总步数)

**集成难度**: ⭐⭐⭐ 中
- 需要 step 级更新
- 需自定义 scheduler 或用 `LambdaLR`

### 2.5 ExponentialLR (指数衰减)

**公式**: `lr = lr * gamma^epoch`

**论文实践**:
- 较老，现代论文少用

**优点**: 简单
**缺点**: 衰减过快或过慢，需调 gamma

**集成难度**: ⭐ 极低

---

## 3. 综合评估

### 3.1 策略对比矩阵

| 策略 | 自适应 | 平滑度 | 集成难度 | 论文匹配度 | 推荐度 |
|------|--------|--------|---------|-----------|--------|
| ReduceLROnPlateau (当前) | ✅ val_loss | ✗ 阶跃 | — | 老论文 | ⭐⭐ |
| **CosineAnnealingLR** | ✗ 固定 | ✅ 平滑 | ⭐ 极低 | **TimesNet/PatchTST/iTransformer** | ⭐⭐⭐⭐⭐ |
| CosineWarmRestarts | ✗ 固定+重启 | ✅ 平滑 | ⭐⭐ 低 | 少见 | ⭐⭐⭐ |
| OneCycleLR | ✗ 固定 | ✅ 平滑 | ⭐⭐⭐ 中 | 视觉 | ⭐⭐ |
| Warmup+Cosine | ✗ 固定 | ✅ 平滑 | ⭐⭐⭐ 中 | PatchTST/Transformer | ⭐⭐⭐⭐ |
| ExponentialLR | ✗ 固定 | ✗ 平滑 | ⭐ 极低 | 老论文 | ⭐⭐ |

### 3.2 推荐: CosineAnnealingLR

**理由**:

1. **论文匹配度最高**: TimesNet、PatchTST、iTransformer、ModernTCN 等我们复现的 baseline 原始论文均使用 Cosine Annealing。使用相同策略可让复现更忠实，结果更可比。

2. **集成成本极低**: 只需改 1 行代码 + TrainConfig 加 1 个字段。不需要改 TrainingLoop 的调用接口（仍为 epoch 级 `scheduler.step()`）。

3. **与早停兼容**: Cosine 退火到 min_lr 后，模型在低 LR 下继续微调，直到早停触发。比 ReduceLROnPlateau 更可控（不会因 val_loss 噪声过早降 LR）。

4. **可预测性**: 固定的退火曲线让实验更可复现，不同模型间的 LR 曲线一致，对比更公平。

### 3.3 备选: Warmup + Cosine

如果发现 Transformer-based 模型 (PatchTST/TimesNet) 训练初期不稳定，可升级为 Warmup + Cosine。但需要 step 级更新，集成成本更高。

### 3.4 不推荐: OneCycleLR / WarmRestarts

- OneCycleLR 设计用于固定短训练，与我们 5000 epoch + 早停的设置不匹配
- WarmRestarts 的重启行为可能破坏已学到的时序模式

---

## 4. 集成方案 (CosineAnnealingLR)

### 4.1 改动范围

| 文件 | 改动 |
|------|------|
| `train/trainer.py` | TrainConfig 加 `lr_scheduler: str = "cosine"` 字段; 新增 `_build_scheduler()` 工厂函数; Trainer.train() 和 TrainingLoop.run() 调用工厂 |
| `model/evl.py`, `timealign.py`, `protots.py`, `dan.py`, `stssdl.py` | 5 个自定义 fit 的 scheduler 构建改用工厂函数 |
| `experiments/run_all.py` | 加 `--lr_scheduler` CLI 参数 |

### 4.2 预计工时

- 代码改动: ~30 分钟 (6 处 scheduler 构建统一改)
- 测试: ~15 分钟 (冒烟测试各模型)
- 文档更新: ~10 分钟 (TODO/RUN_GUIDE)
- **总计: ~1 小时**

### 4.3 预期效果

**正面**:
- baseline 复现更忠实（与原始论文一致）
- 训练更稳定（不受 val_loss 噪声影响 LR 衰减）
- 不同模型间 LR 曲线一致，对比更公平

**风险**:
- ReduceLROnPlateau → Cosine 对某些模型可能改变收敛行为
- 需要重新跑已有实验结果（如果已有结果的话）
- `T_max=5000` 但实际早停在 ~50 epoch，Cosine 只退了 1% → 影响较小，可考虑 `T_max=100` (预期训练长度)

### 4.4 T_max 选择建议

由于 `epochs=5000` 是上限但实际早停通常在 30-100 epoch：
- **方案 A**: `T_max=cfg.epochs` (5000) — LR 退火极慢，接近恒定 LR
- **方案 B**: `T_max=100` — 假设典型训练 100 epoch，完整退火
- **方案 C**: `T_max=cfg.early_stop_patience * 3` (90) — 自适应于 patience

推荐 **方案 B** (`T_max=100`)，符合典型训练长度，且即使早停在 50 epoch，LR 也已退到合理低位。

---

## 5. 结论

**当前策略**: ReduceLROnPlateau — 可用但不符现代时序论文标准。

**推荐升级**: CosineAnnealingLR — 集成成本极低 (1 小时)，论文匹配度最高，效果预期正面。

**等待指令**: 是否执行此修改。如果执行，建议同时清理已有 checkpoint 和结果缓存，用新 LR 策略重新训练。
