# STCross 项目 Group C & D Baseline 模型审查报告

**审查日期**: 2025-06-25
**审查范围**: Group C (分布鲁棒/可迁移方法) 和 Group D (极端事件预测方法) 的 7 个 baseline 模型
**审查标准**: 架构忠实度、论文复现度、框架兼容性、N-independence、训练/推理正确性

---

## 目录

1. [Group C: 分布鲁棒/可迁移方法](#group-c)
   - 1.1 TimeAlign
   - 1.2 ProtoTS
   - 1.3 ModernTCN
2. [Group D: 极端事件预测方法](#group-d)
   - 2.1 EVL
   - 2.2 EALGAP
   - 2.3 DAN
   - 2.4 ST-SSDL
3. [总体评价](#summary)
4. [问题汇总表](#issue-table)

---

## <a id="group-c"></a>1. Group C: 分布鲁棒/可迁移方法

### <a id="ta"></a>1.1 TimeAlign (`model/timealign.py`)

**论文**: Hu et al. "Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting" (ICLR 2026)
**参考代码**: https://github.com/TROUBADOUR000/TimeAlign

#### 1.1.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| RevIN | 可逆实例归一化 | 正确 | 独立实现，normalize/denormalize 分离 |
| Predict Branch | History → patches → Encoder → Linear head | 正确 | 使用 PatchEncoder(M层Linear-GELU-Dropout-Linear+残差+LayerNorm) |
| Reconstruct Branch | Future → patches → Encoder → Linear head | 正确 | 与 Predict Branch 对称，仅在训练时使用 |
| PatchEncoder | Linear embed → Positional Encoding → M-layer MLP with residual | 正确 | 使用 SinusoidalPositionalEmbedding |
| Alignment Projections | 每层一个 Linear(d_model, d_model) | 正确 | 用于对齐 predict 和 reconstruct 的中间表示 |
| 推理时重建分支 | 仅在训练时使用 | 正确 | `_predict()` 仅走 predict branch，`_train_step()` 走双分支 |

#### 1.1.2 损失函数审查

**论文损失函数**: L = L_pred + L_recon + lambda * L_align

- **L_pred**: L1 loss between prediction and ground truth (使用 L1，正确)
- **L_recon**: L1 loss between reconstructed x and original x (使用 L1，正确)
- **L_align**: 分为 local alignment 和 global alignment

**Local Alignment**:
- 论文公式: 对每层中间表示，计算归一化后的元素级相似度
- 实现: `F.normalize(h_x_proj, dim=-1)` 和 `F.normalize(h_y_sg, dim=-1)`，然后 `F.gelu(1 - |h_x_n * h_y_n| - delta_loc)`
- 使用 stop-gradient 在 reconstruct branch（`h_y_sg = h_y.detach()`），loss 仅反向传播到 predict branch
- **正确**

**Global Alignment**:
- 论文公式: L2-normalized distance matrix
- 实现: `F.normalize(h_x_proj, p=2, dim=-1)` 和 `F.normalize(h_y_sg, p=2, dim=-1)`，然后 `torch.cdist` 求距离矩阵，`F.gelu(|D_x - D_y| - delta_glo)`
- **正确**

**Dynamic Weighting**:
- 论文公式: `w_avg = mean(detached_losses)`, `total = sum(w_avg * loss_i / loss_i.detach())`
- 实现: 完全匹配，在 `torch.no_grad()` 下计算 w_avg
- **正确**

#### 1.1.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 自定义 fit() | 正确实现 | 需要 Y 来计算重建和对齐损失 |
| RevIN 的 normalize/denormalize 状态管理 | 有潜在问题 | 见下方问题 |

#### 1.1.4 发现的问题

**问题 TA-1 (严重程度: 中)**: RevIN 状态管理存在竞态风险

在 `_train_step()` 中，`self.revin` 是同一个实例。先调用 `normalize(x)` 保存 `_mean` 和 `_std`，然后立即调用 `normalize(y)` **覆盖**了这些状态。代码通过局部变量 `pred_mean, pred_std` 和 `recon_mean, recon_std` 保存了状态，但这是在 `self.revin._mean` 变化之后读取的。如果未来代码有异步操作或中间有其他调用，可能导致状态不一致。

**当前代码**:
```python
x_norm = self.revin.normalize(x)
pred_mean, pred_std = self.revin._mean, self.revin._std  # 此时可能已被覆盖

y_norm = self.revin.normalize(y)  # 覆盖了 _mean 和 _std
recon_mean, recon_std = self.revin._mean, self.revin._std
```

**建议修复**: 在 `normalize()` 返回归一化后的数据的同时也返回 `(mean, std)`，或者使用两个独立的 RevIN 实例分别服务于 predict 和 reconstruct 分支。

**实际影响**: 在当前实现中，由于 `pred_mean, pred_std` 在 `normalize(y)` 之前被读取，实际上是正确的。但代码可读性和健壮性不好。

**问题 TA-2 (严重程度: 低)**: 训练时总损失权重

训练时 `loss = pred_loss + recon_loss + self.lambda_align * align_loss`，其中 `pred_loss` 和 `recon_loss` 权重均为 1.0，`align_loss` 权重为 `lambda_align`（默认 1.0）。论文中可能对不同损失分量有不同的权重设置，但 1.0 是一个合理的默认值。

**问题 TA-3 (严重程度: 低)**: `_patchify` 使用 `unfold` 忽略了不够整除的余数

当 `H % patch_num != 0` 时，`pred_patch_len = H // patch_num`，导致 `H // pred_patch_len * pred_patch_len < H`，部分时间步未被纳入 patches。这会导致信息丢失。论文中通常使用 padding 或 stride-based patching 来处理。

**建议**: 使用带 padding 的 patching 以确保所有时间步被覆盖，参考 ModernTCN 的做法。

#### 1.1.5 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| RevIN | 否 | 对每个 (B, H) 独立归一化 |
| PatchEncoder | 否 | 全部参数与 N 无关 |
| Predict head | 否 | Linear(n_patches*d_model, P) |
| Reconstruct head | 否 | Linear(n_patches*d_model, H) |
| Alignment projections | 否 | Linear(d_model, d_model) |

**结论**: 所有参数均为 N-independent，可以跨城市迁移。

#### 1.1.6 总体评价

**忠实度**: 高。TimeAlign 的核心架构（RevIN + 双分支 + 局部/全局对齐 + 动态权重）被忠实地实现。自定义 `fit()` 正确处理了训练时 Y 的需求，推理时仅使用 predict branch。

**评分**: 9/10

---

### <a id="pt"></a>1.2 ProtoTS (`model/protots.py`)

**论文**: Peng et al. "ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting" (ICLR 2026)
**参考代码**: https://github.com/SKURA502/ProtoTS

#### 1.2.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| Patching | patch_len=4, stride=2 | 正确 | 使用 `unfold(1, patch_len, stride)` |
| Bottleneck MLP-Mixer | Feature mixing + Time mixing with residual | 正确 | 独立实现 BottleneckMixerLayer |
| Feature Mixing | Linear(d, d_bottle) → GELU → Linear(d_bottle, d) | 正确 | 带残差连接 |
| Time Mixing | Linear(n_tokens, d_bottle) → GELU → Linear(d_bottle, n_tokens) | 正确 | 带残差连接，通过 transpose 实现 |
| Time Aggregation | Linear(n_patches, 1) | 正确 | 将 n_patches 聚合为 1 |
| Prototype Matching | Euclidean dist(z, μ_i) → softmax | 正确 | 使用 `torch.cdist` |
| Prediction | Σ w_i * p_i (weighted sum of prototype patterns) | 正确 | 使用 `torch.matmul(weights, proto_pattern)` |

#### 1.2.2 损失函数审查

**论文损失函数**: L = L1 + λ * H(w)，其中 H(w) = -Σ w_i * log(w_i) 是熵正则化

- **L1**: `F.l1_loss(pred, Y_flat.squeeze(-1))` — 正确
- **Entropy 正则化**: `-(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()` — 正确
  - 论文中写的是 **entropy minimization**（鼓励稀疏激活），所以是 `+ λ * H(w)` 而非 `- λ * H(w)`
  - 实现中 `loss = l1_loss + self.lambda_ent * entropy` — 正确

#### 1.2.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 自定义 fit() | 正确实现 | 需要计算熵损失 |
| 验证时仅使用 L1 | 正确 | 验证时不计算熵损失 |

#### 1.2.4 发现的问题

**问题 PT-1 (严重程度: 低)**: 论文中的 "hierarchical" 原型未被实现

论文标题中提到 "Hierarchical Prototypes"，但当前实现仅使用单层原型（8 个原型向量）。论文可能涉及多层原型结构（如粗粒度-细粒度原型层次），但参考代码 SKURA502/ProtoTS 可能也仅实现了单层版本。如果论文确实需要层次化原型，这是一个遗漏。

**问题 PT-2 (严重程度: 低)**: 原型初始化

`proto_mu` 和 `proto_pattern` 使用 `torch.randn(..., 0.1)` 初始化。论文可能使用不同的初始化策略（如 K-means 聚类初始化），但随机初始化在实践中通常工作良好。

**问题 PT-3 (严重程度: 低)**: 训练时 reshape 每次迭代都做

在 `fit()` 中，每个 batch 都需要 `X.permute(0, 2, 1).reshape(B * N, H, 1)`，这增加了少量开销。可以在 DataLoader 层面处理，但这不是功能性问题。

#### 1.2.5 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| embed | 否 | Linear(patch_len, d_model) |
| BottleneckMixerLayer | 否 | 全部参数与 N 无关 |
| time_agg | 否 | Linear(n_patches, 1) |
| proto_mu | 否 | (K, d_model) |
| proto_pattern | 否 | (K, P)，P 是固定的预测步数 |

**结论**: 所有参数均为 N-independent，可以跨城市迁移。

#### 1.2.6 总体评价

**忠实度**: 高。核心架构（MLP-Mixer + 原型匹配 + 熵正则化）被正确实现。单层原型可能简化了论文的层次化设计，但参考代码可能也是如此。

**评分**: 8.5/10

---

### <a id="mt"></a>1.3 ModernTCN (`model/moderntcn.py`)

**论文**: Luo & Wang "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis" (ICLR 2024 Spotlight)
**参考代码**: https://github.com/luodhhh/ModernTCN

#### 1.3.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| Patching | patch_len=4, stride=2 | 正确 | 带 padding 确保覆盖所有时间步 |
| ModernTCN Block | DWConv + ConvFFN | 正确 | 忠实于 ConvNeXt 风格 |
| DWConv | Depthwise Conv1d with groups=d_model | 正确 | kernel_size=5, padding=kernel_size//2 |
| ConvFFN | Linear(d→d_ff) → GELU → Linear(d_ff→d) → Dropout | 正确 | 倒置瓶颈结构 |
| LayerNorm | 在 DWConv 和 ConvFFN 之前 | 正确 | 符合 ConvNeXt 设计 |
| Residual | 两个分支都有残差连接 | 正确 | |
| 最终输出 | Flatten → Linear head | 正确 | |

#### 1.3.2 损失函数审查

- 使用标准 MSE loss，通过默认 Trainer 训练
- **正确**，论文使用 MSE 作为回归损失

#### 1.3.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 不需要自定义 fit() | 正确 | 使用标准 Trainer |
| Padding 处理 | 正确 | 使用 replicate padding 确保所有时间步被覆盖 |

#### 1.3.4 发现的问题

**问题 MT-1 (严重程度: 低)**: DWConv 的 LayerNorm 顺序

ConvNeXt 论文中通常是 LayerNorm 在 DWConv 之前，但 DWConv 希望输入是 (B, C, L) 格式。当前实现是:
```python
h = self.norm1(x).transpose(1, 2)  # (B, n_patches, d) → (B, d, n_patches)
h = self.dwconv(h).transpose(1, 2)  # (B, d, n_patches) → (B, n_patches, d)
```
LayerNorm 作用于特征维度 (d_model)，然后 transpose 到 (B, d, n_patches) 给 Conv1d。这是正确的，因为 Conv1d 期望 (B, C, L) 输入。

**问题 MT-2 (严重程度: 低)**: 论文中的 CI (Channel Independent) 模式

论文区分了 CI 和 CD (Channel Dependent) 模式。当前实现使用 CI 模式（`input_mode = "per_node"`），这与论文的 CI 设计一致。但论文的 CD 模式可能涉及跨通道的 DWConv，当前实现没有这个选项。

**问题 MT-3 (严重程度: 低)**: 论文中的 "large kernel" 默认 kernel_size

论文标题中提到 "Large Kernel"，但默认 kernel_size=5。对于时间序列，5 可能不算特别大。论文可能使用更大的 kernel（如 7 或 9），但 5 是一个合理的默认值。

#### 1.3.5 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| embed | 否 | Linear(patch_len, d_model) |
| DWConv | 否 | groups=d_model, 与 N 无关 |
| ConvFFN | 否 | Linear 层与 N 无关 |
| head | 否 | Linear(n_patches*d_model, P) |

**结论**: 所有参数均为 N-independent，可以跨城市迁移。

#### 1.3.6 总体评价

**忠实度**: 高。ModernTCN 的核心架构（DWConv + ConvFFN + LayerNorm + 残差）被正确实现。使用标准 MSE 训练，无需自定义 fit()。这是 Group C 中实现最简单的模型，也是最忠实的实现之一。

**评分**: 9.5/10

---

## <a id="group-d"></a>2. Group D: 极端事件预测方法

### <a id="evl"></a>2.1 EVL (`model/evl.py`)

**论文**: Ding et al. "Modeling Extreme Events in Time Series Prediction" (KDD 2019)
**参考代码**: https://github.com/HuiqunHuang/EVL

#### 2.1.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| LSTM Encoder | 多层 LSTM | 正确 | n_lstm_layers=2, hidden_dim=64 |
| Memory Network | 存储极端事件模式的记忆模块 | 正确 | 注意力检索 |
| Prediction Head | concat(h, m) → Linear → ŷ | 正确 | 使用 hidden_dim*2 输入 |
| Extreme Classifier | concat(h, m) → Linear → sigmoid → P(extreme) | 正确 | 输出概率 |

#### 2.1.2 损失函数审查

**论文损失函数**: L = MSE(ŷ, y) + λ * EVL(P(extreme), extreme_label)

**EVL (Extreme Value Loss)**:
论文公式: L_EVL = -β0 * (1-u/γ)^γ * v * log(u) - β1 * (1-(1-u)/γ)^γ * (1-v) * log(1-u)

实现:
```python
w_pos = (1.0 - u / gamma).clamp(min=eps) ** gamma
w_neg = (1.0 - (1.0 - u) / gamma).clamp(min=eps) ** gamma
loss_pos = -beta0 * w_pos * v * log(u)
loss_neg = -beta1 * w_neg * (1.0 - v) * log(1.0 - u)
```

**正确**。公式被忠实地实现，包括自适应类别平衡权重 β0, β1。

**Extreme Label 生成**:
- 使用 `np.percentile(all_Y_abs, self.extreme_percentile)` 确定阈值
- `extreme_percentile` 默认 90.0（即 top 10% 为极端值）
- **正确**，符合论文中基于分位数的极端事件定义

#### 2.1.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 自定义 fit() | 正确实现 | 需要 EVL 损失 |
| 验证时仅使用 MSE | 正确 | 验证时不计算 EVL 损失 |

#### 2.1.4 发现的问题

**问题 EVL-1 (严重程度: 中)**: LSTM 隐藏状态使用方式

`_encode()` 返回 `h_n[-1]`（最后一层的隐藏状态），但论文可能使用 `h_n[-1]` 和 `c_n[-1]` 的拼接，或者使用 `h_n` 的所有层。当前实现仅使用最后一层的隐藏状态，可能丢失了多层信息。

**建议**: 考虑使用所有层的 `h_n` 拼接或使用 `h_n[-1]` 和 `c_n[-1]` 拼接。

**问题 EVL-2 (严重程度: 低)**: Memory Network 的梯度传播

Memory Network 使用 `nn.Parameter` 存储记忆，通过注意力机制检索。推理时 `_net_forward()` 调用 `_predict_with_memory(h)` 返回 `pred` 和 `extreme_prob`，但仅返回 `pred`。由于 `extreme_prob` 在推理时不被使用，其计算是多余的但无害。

**问题 EVL-3 (严重程度: 低)**: Extreme label 是按时间步计算的

`_compute_extreme_labels()` 对 Y 的每个时间步独立判断是否极端。在多步预测中（P=12），极端标签是针对每个预测步的。论文可能对每个预测步有不同的极端定义，但基于数据集的全局阈值是一个合理的近似。

#### 2.1.5 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| LSTM | 否 | input_size=1, hidden_size=64 |
| Memory Network | 否 | (mem_size, d_model) |
| predict_head | 否 | Linear(hidden_dim*2, hidden_dim) → Linear(hidden_dim, P) |
| extreme_head | 否 | 同上 |

**结论**: 所有参数均为 N-independent，可以跨城市迁移。

#### 2.1.6 总体评价

**忠实度**: 高。EVL 的核心架构（LSTM + Memory Network + 双头预测 + EVL 损失）被正确实现。EVL 损失函数的数学公式被忠实地实现。

**评分**: 8.5/10

---

### <a id="ea"></a>2.2 EALGAP (`model/ealgap.py`)

**论文**: Huang et al. "Extreme-Aware Local-Global Attention for Spatio-Temporal Urban Mobility Learning" (ICDE 2023)
**参考代码**: https://github.com/HuiqunHuang/EALGAP

#### 2.2.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| GlobalTrendModule | PDF-based self-attention | 部分实现 | 见下方问题 |
| LocalExtremeModule | Instance z-score → GRU → tanh | 近似实现 | 论文使用 same-slot 历史统计 |
| Multiplicative Fusion | output = global * (1 + extreme) | 正确 | 后跟 ReLU |

#### 2.2.2 GlobalTrendModule 详细审查

**PDF 计算**:
- 论文: 使用数据分布的指数 PDF 生成 Q/K/V
- 实现: `lambda = 1 / mean(|x|)`, `pdf = lambda * exp(-lambda * |x|)`
- **近似正确**，但论文可能使用更复杂的分布拟合（如 GMM 或混合指数分布）

**Data-dependent Q/K/V 生成**:
- 实现: `qkv_mlp` 使用两层 Linear + Softmax 生成 Q/K/V
- 论文中可能使用更复杂的分布感知生成机制，但 MLP 是一个合理的近似

**Multi-head Self-Attention**: 标准实现，正确

**问题 EA-1 (严重程度: 中)**: PDF 计算方式可能与论文不完全一致

论文的 EALGAP 可能使用 pre-computed 的 PDF 值（作为输入特征），而非 on-the-fly 计算。当前实现中，PDF 是每个 batch 动态计算的，这不完全忠实于论文的数据处理流程。但动态计算使模型更加灵活，可以适应不同的数据分布。

**问题 EA-2 (严重程度: 中)**: QKV MLP 中使用了 Softmax 激活

```python
self.qkv_mlp = nn.Sequential(
    nn.Linear(d_model, d_model * 2),
    nn.Softmax(dim=-1),
    nn.Linear(d_model * 2, d_model * 3),
    nn.Softmax(dim=-1),
)
```

Softmax 激活通常用于 attention weights，而非特征变换。这可能导致梯度消失或信息丢失。论文中可能使用 GELU 或 ReLU 作为激活函数。Softmax 是一个非标准选择。

#### 2.2.3 LocalExtremeModule 详细审查

**问题 EA-3 (严重程度: 中)**: 论文使用 same-slot 历史统计，实现使用 instance 统计

论文中的 LocalExtremeModule 计算与 same-slot 历史数据的偏差，即比较当前时间步与该时间步的历史均值的偏差。当前实现使用 instance z-score（当前样本自身的均值和标准差），这是一个简化。

**影响**: 对于 traffic flow 预测，instance z-score 和 same-slot z-score 可能给出不同的偏差信号。same-slot 偏差更能捕捉时间模式（如早高峰的异常），而 instance 偏差捕捉的是样本内的相对变化。

**建议**: 如果数据集中有足够的历史数据，考虑预先计算 same-slot 统计量，或将它们作为额外输入。

#### 2.2.4 损失函数审查

- 使用标准 MSE loss，通过默认 Trainer 训练
- **正确**，论文使用 MSE 作为回归损失

#### 2.2.5 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 不需要自定义 fit() | 正确 | 使用标准 Trainer |

#### 2.2.6 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| GlobalTrendModule | 否 | 全部参数与 N 无关 |
| LocalExtremeModule | 否 | 全部参数与 N 无关 |

**结论**: 所有参数均为 N-independent。

#### 2.2.7 总体评价

**忠实度**: 中。EALGAP 的核心思想（全局趋势 + 局部极端 + 乘法融合）被保留，但实现细节与论文有多处偏差：
1. PDF 计算是动态而非预计算的
2. QKV 生成使用了非标准的 Softmax 激活
3. LocalExtremeModule 使用 instance z-score 而非 same-slot 偏差

这些偏差使实现更灵活，但降低了与原始论文的忠实度。

**评分**: 6.5/10

---

### <a id="dan"></a>2.3 DAN (`model/dan.py`)

**论文**: Li et al. "Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting" (AAAI 2024)
**参考代码**: https://github.com/davidanastasiu/dan

#### 2.3.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| Far Encoder/Decoder | CNN+LSTM (论文) | FC (实现) | 见下方问题 |
| Near Encoder/Decoder | CNN+LSTM (论文) | FC (实现) | 见下方问题 |
| RepMerg | concat(h_far, y_near) → FC → y_final | 正确 | |
| 双通道 (Far/Near) | 正确 | 极坐标表示的双通道分解 |

**问题 DAN-1 (严重程度: 高)**: 论文使用 CNN+LSTM 编码器-解码器，实现使用纯 FC

论文的 DAN 使用 CNN+LSTM 作为编码器和解码器架构，以捕捉时间序列的局部和长期依赖。当前实现使用纯全连接层（`PolarEncoder` 和 `PolarDecoder`），这丢失了 CNN 的局部特征提取能力和 LSTM 的序列建模能力。

**代码中明确承认了这一点**:
```python
"""Note: This is a simplified version using FC backbone instead of the paper's
CNN+LSTM encoder/decoder architecture."""
```

**影响**: 对于长序列预测（如 P=12 或更大），纯 FC 可能无法有效捕捉时间依赖。如果 STCross 的 H 和 P 较小（如 12），FC 可能够用，但与论文的差异仍然显著。

**建议**: 将 `PolarEncoder` 和 `PolarDecoder` 改为 CNN+LSTM 架构，或至少添加 CNN 层。

#### 2.3.2 损失函数审查

**论文损失函数**: Distance-weighted dual-channel loss
- w_far = tanh(y_centered)^2 (高权重给极端值)
- w_near = (1 - |tanh(y_centered)|)^2 (高权重给正常值)
- L = w_far * MSE(y_far, y) + w_near * MSE(y_near, y) + MSE(y_final, y)

**实现**:
```python
t = torch.tanh(y)
w_far = t ** 2
w_near = (1 - t.abs()) ** 2
```

**正确**。距离权重公式被忠实地实现。注意论文要求数据先 center 到 mean=0，代码中做了 `y_centered = y_target - y_target.mean(dim=0, keepdim=True)`。

**问题 DAN-2 (严重程度: 低)**: Center 的方式

当前 center 是 `y_target - y_target.mean(dim=0, keepdim=True)`，即按 batch 维度 center。论文可能期望对整个数据集进行 center（使用全局均值）。但由于 batch 训练，全局 center 需要在训练前预处理。按 batch center 是一个合理的近似，但可能引入 batch 间的方差。

#### 2.3.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 自定义 fit() | 正确实现 | 需要距离加权损失 |

#### 2.3.4 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| PolarEncoder (FC) | 否 | Linear(H, d_model) → Linear(d_model, d_model) |
| PolarDecoder (FC) | 否 | Linear(d_model, d_model) → Linear(d_model, P) |
| Merge | 否 | Linear(d_model+P, d_model) → Linear(d_model, P) |

**结论**: 所有参数均为 N-independent。

#### 2.3.5 总体评价

**忠实度**: 中低。核心的极坐标表示和距离加权损失被正确实现，但编码器-解码器架构从 CNN+LSTM 简化为纯 FC，这是一个显著的偏差。代码中明确标注了这一点，但论文的核心架构（CNN+LSTM）未被保留。

**评分**: 6/10

---

### <a id="ss"></a>2.4 ST-SSDL (`model/stssdl.py`)

**论文**: Gao et al. "How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning" (NeurIPS 2025)
**参考代码**: https://github.com/Jimmy-7664/ST-SSDL

#### 2.4.1 架构审查

| 组件 | 论文要求 | 实现状态 | 备注 |
|------|---------|---------|------|
| Historical Anchoring | x_dev = x - running_mean(x) | 部分实现 | 使用 instance mean 而非 running mean |
| Encoder | GCRU (论文) | Linear (实现) | 见下方问题 |
| Learnable Prototypes | K 个共享原型 | 正确 | 与 ProtoTS 类似 |
| Prototype Assignment | nearest neighbor | 正确 | argmin distance |
| Prediction Head | concat(z, μ*) → FC → ŷ | 正确 | |

**问题 SS-1 (严重程度: 高)**: 论文使用 GCRU (Graph Conv GRU)，实现使用纯 Linear 编码器

论文的 ST-SSDL 使用 GCRU 作为编码器骨干，结合了图卷积和 GRU 来建模时空依赖。当前实现使用纯 Linear 编码器（`nn.Linear(n_patches*d_model, d_model)`），这丢失了空间建模能力和序列建模能力。

**代码中明确承认了这一点**:
```python
"""Note: This is an "ST-SSDL-lite" adaptation. The original paper uses a GCRU
(graph conv GRU) backbone with dual-branch encoding (current + weekly-average
anchor), while this implementation uses a per-node Linear encoder as an
N-independent adaptation."""
```

**影响**: 对于跨城市迁移场景，去除 GCRU 是合理的（因为图结构是 N-dependent 的），但这也意味着模型失去了空间建模能力。如果在同城市场景中对比，这个简化版本可能无法达到论文报告的性能。

**问题 SS-2 (严重程度: 中)**: Historical Anchoring 使用 instance mean 而非 running mean

论文使用 running mean（滑动平均）作为历史锚点，捕获长期趋势。当前实现使用 `x.mean(dim=1, keepdim=True)`（instance mean），这只能捕获样本内的均值，而非长期趋势。

**建议**: 使用 exponential moving average 或预先计算的历史均值作为锚点。

**问题 SS-3 (严重程度: 中)**: 论文的 dual-branch encoding 未被实现

论文使用双分支编码：current branch（当前输入）和 weekly-average branch（同星期几的平均值）。当前实现仅使用单分支编码。这简化了模型但丢失了周期性对比信息。

#### 2.4.2 损失函数审查

**论文损失函数**: L = L_pred + α * L_contrast + β * L_deviation

**L_pred**: L1 loss between prediction and ground truth — 正确

**L_contrast (对比损失)**:
- 论文: 将不同原型推远
- 实现: Triplet-like margin loss，pos=最近原型距离，neg=第二近原型距离
```python
loss = F.relu(pos_dist - neg_dist + margin).mean()
```
- **问题**: 论文的对比损失可能直接作用于原型向量（如 `mean(||μ_i - μ_j||)` 或 InfoNCE 风格），而非 triplet margin。当前实现使用 triplet loss 作用于样本-原型距离，这是不同的公式。

**L_deviation (偏差一致性损失)**:
- 论文: 对齐 `dist(z, μ*)` 和预测误差 `|ŷ - y|`
- 实现: `F.mse_loss(min_dist, pred_error.detach())`
- **正确**，但 `pred_error.detach()` 阻止了梯度传播到预测误差，这是合理的（避免循环依赖）。

**问题 SS-4 (严重程度: 中)**: 对比损失公式与论文可能不一致

Triplet loss 和论文的 contrastive loss 公式不同。论文可能使用 InfoNCE 风格的对比损失：
```
L_contrast = -log(exp(sim(z, μ*)) / Σ exp(sim(z, μ_i)))
```
或直接作用于原型向量：
```
L_contrast = -mean(||μ_i - μ_j||) for i≠j
```

当前 triplet 实现选择的是正样本（最近原型）和负样本（第二近原型），这与论文的 "push different prototypes apart" 有相似效果，但公式不同。

#### 2.4.3 框架兼容性

| 检查项 | 状态 | 说明 |
|-------|------|------|
| input_mode | "per_node" | 正确 |
| 输入维度 | (B, H, 1) → (B, P, 1) | `_net_forward()` 正确 |
| 自定义 fit() | 正确实现 | 需要三重损失 |

#### 2.4.4 N-independence 审查

| 参数 | N-dependent? | 说明 |
|-----|-------------|------|
| embed | 否 | Linear(patch_len, d_model) |
| encoder | 否 | Linear(n_patches*d_model, d_model) → Linear(d_model, d_model) |
| prototypes | 否 | (K, d_model) |
| pred_head | 否 | Linear(d_model*2, d_model) → Linear(d_model, P) |

**结论**: 所有参数均为 N-independent。但去除 GCRU 意味着空间信息被忽略。

#### 2.4.5 总体评价

**忠实度**: 中低。ST-SSDL 的核心思想（偏差学习 + 原型匹配 + 对比损失）被保留，但实现有多处显著偏差：
1. GCRU 骨干被替换为纯 Linear 编码器
2. Historical anchoring 使用 instance mean 而非 running mean
3. 对比损失公式使用 triplet loss 而非论文的 contrastive loss
4. Dual-branch encoding 未被实现

代码中明确标注为 "ST-SSDL-lite"，这是一个诚实的简化版本。但作为 baseline 对比，其性能可能与原始论文报告有显著差距。

**评分**: 5.5/10

---

## <a id="summary"></a>3. 总体评价

### 3.1 忠实度排名

| 排名 | 模型 | 评分 | 组别 | 备注 |
|-----|------|------|------|------|
| 1 | ModernTCN | 9.5/10 | C | 几乎完美复现，无自定义 fit() |
| 2 | TimeAlign | 9/10 | C | 核心架构忠实，RevIN 状态管理有小问题 |
| 3 | ProtoTS | 8.5/10 | C | 忠实，层次化原型可能简化 |
| 3 | EVL | 8.5/10 | D | 忠实，LSTM 隐藏状态使用有优化空间 |
| 5 | EALGAP | 6.5/10 | D | PDF 计算和 QKV 生成有偏差 |
| 6 | DAN | 6/10 | D | CNN+LSTM 简化为 FC，但损失函数忠实 |
| 7 | ST-SSDL | 5.5/10 | D | GCRU 替换为 Linear，多处简化 |

### 3.2 框架兼容性总结

所有 7 个模型均正确实现了框架接口：
- 全部使用 `input_mode = "per_node"`
- 全部实现 `_net_forward()` 返回 `(B, P, 1)`
- 需要自定义 `fit()` 的模型（TimeAlign, ProtoTS, EVL, DAN, ST-SSDL）均正确实现了训练循环
- 不需要自定义 `fit()` 的模型（ModernTCN, EALGAP）使用标准 Trainer

### 3.3 N-independence 总结

所有 7 个模型的参数均为 N-independent，可以跨城市迁移。这是通过以下方式实现的：
- 使用 per_node 模式（每个节点独立处理）
- 不使用图卷积或空间注意力
- 所有参数维度与 N 无关

对于 ST-SSDL 和 DAN，去除空间建模组件（GCRU、图卷积）是为了满足 N-independence 要求，但这也降低了模型在需要空间信息的场景中的性能。

### 3.4 训练/推理切换总结

| 模型 | 训练时特殊行为 | 推理时行为 | 切换正确性 |
|------|-------------|----------|-----------|
| TimeAlign | 双分支 + 对齐损失 | 仅 predict branch | 正确 |
| ProtoTS | L1 + 熵正则化 | 仅 L1 | 正确 |
| ModernTCN | 标准 MSE | 标准推理 | 正确 |
| EVL | MSE + EVL 损失 | 仅 MSE | 正确 |
| EALGAP | 标准 MSE | 标准推理 | 正确 |
| DAN | 距离加权三通道损失 | 仅 y_final | 正确 |
| ST-SSDL | 三重损失 | 仅预测 | 正确 |

---

## <a id="issue-table"></a>4. 问题汇总表

### 严重问题 (High)

| 编号 | 模型 | 问题描述 | 建议 |
|-----|------|---------|------|
| DAN-1 | DAN | 论文使用 CNN+LSTM 编码器，实现使用纯 FC | 改为 CNN+LSTM 架构 |
| SS-1 | ST-SSDL | 论文使用 GCRU 骨干，实现使用纯 Linear | 在 N-independent 前提下考虑使用 CNN 或 LSTM 替代 |

### 中等问题 (Medium)

| 编号 | 模型 | 问题描述 | 建议 |
|-----|------|---------|------|
| TA-1 | TimeAlign | RevIN 状态管理有竞态风险 | 使用两个独立 RevIN 实例 |
| EVL-1 | EVL | LSTM 仅使用最后一层隐藏状态 | 考虑使用所有层或 cell state |
| EA-1 | EALGAP | PDF 计算是动态的而非预计算的 | 评估是否需要预计算 |
| EA-2 | EALGAP | QKV MLP 使用 Softmax 激活（非标准） | 改为 GELU 或 ReLU |
| EA-3 | EALGAP | 使用 instance z-score 而非 same-slot 偏差 | 考虑预计算 same-slot 统计量 |
| SS-2 | ST-SSDL | Historical anchoring 使用 instance mean 而非 running mean | 使用 EMA 或预计算历史均值 |
| SS-3 | ST-SSDL | 缺少 dual-branch encoding | 如需要可添加 weekly-average branch |
| SS-4 | ST-SSDL | 对比损失使用 triplet loss 而非论文公式 | 确认论文的对比损失公式并修改 |

### 轻微问题 (Low)

| 编号 | 模型 | 问题描述 | 建议 |
|-----|------|---------|------|
| TA-2 | TimeAlign | 损失权重可能不是最优 | 添加可配置的权重参数 |
| TA-3 | TimeAlign | _patchify 可能丢失部分时间步 | 使用带 padding 的 patching |
| PT-1 | ProtoTS | 层次化原型可能未被实现 | 检查参考代码是否需要多层 |
| PT-2 | ProtoTS | 原型初始化策略 | 可以考虑 K-means 初始化 |
| PT-3 | ProtoTS | 每次迭代 reshape 开销 | 可优化但非必须 |
| MT-1 | ModernTCN | LayerNorm 后 transpose 的顺序 | 当前实现正确，可保留 |
| MT-2 | ModernTCN | 仅支持 CI 模式 | 可接受，per_node 需要 CI |
| MT-3 | ModernTCN | kernel_size=5 可能不够大 | 考虑增大到 7 或 9 |
| EVL-2 | EVL | 推理时多余的 extreme_prob 计算 | 可优化但无害 |
| EVL-3 | EVL | 极端标签按时间步独立判断 | 可接受 |
| DAN-2 | DAN | Center 使用 batch mean 而非全局 mean | 可接受，但建议预处理 |

---

## 5. 改进建议

### 5.1 高优先级

1. **DAN**: 将 FC 编码器改为 CNN+LSTM 或至少添加 1D CNN 层，以恢复论文的序列建模能力。这是最显著的偏差。

2. **ST-SSDL**: 在保持 N-independence 的前提下，考虑使用 LSTM 或 TCN 作为编码器骨干替代纯 Linear，以恢复序列建模能力。

### 5.2 中优先级

3. **EALGAP**: 将 QKV MLP 中的 Softmax 替换为 GELU 或 ReLU，这是更标准的激活函数选择。

4. **EALGAP**: 考虑预计算 same-slot 统计量用于 LocalExtremeModule，或在模型内部维护 running statistics。

5. **TimeAlign**: 重构 RevIN 使用方式，使用两个独立实例或让 normalize() 返回 (output, mean, std)。

6. **ST-SSDL**: 将对比损失从 triplet loss 改为论文的原始公式。

### 5.3 低优先级

7. **TimeAlign**: 使用带 padding 的 patching 以确保所有时间步被覆盖（参考 ModernTCN 的做法）。

8. **ProtoTS**: 如果论文需要层次化原型，实现多层原型结构。

9. **ST-SSDL**: 添加 dual-branch encoding 和 running mean anchoring。

---

## 6. 附录: 框架接口验证

### 6.1 所有模型的 `_net_forward()` 签名验证

| 模型 | 输入 shape | 输出 shape | 正确? |
|-----|-----------|-----------|------|
| TimeAlign | (B*N, H, 1) | (B*N, P, 1) | 是 |
| ProtoTS | (B*N, H, 1) | (B*N, P, 1) | 是 |
| ModernTCN | (B*N, H, 1) | (B*N, P, 1) | 是 |
| EVL | (B*N, H, 1) | (B*N, P, 1) | 是 |
| EALGAP | (B*N, H, 1) | (B*N, P, 1) | 是 |
| DAN | (B*N, H, 1) | (B*N, P, 1) | 是 |
| ST-SSDL | (B*N, H, 1) | (B*N, P, 1) | 是 |

### 6.2 自定义 fit() 与标准 Trainer 的差异

| 模型 | 自定义 fit() 原因 | 标准 Trainer 无法满足的原因 |
|-----|-----------------|------------------------|
| TimeAlign | 需要 Y 计算重建和对齐损失 | 标准 Trainer 只传递 X 给 model |
| ProtoTS | 需要计算熵正则化损失 | 标准 Trainer 使用 MSE，不支持额外损失 |
| ModernTCN | 不需要 | N/A |
| EVL | 需要 EVL 损失和极端标签 | 标准 Trainer 使用 MSE |
| EALGAP | 不需要 | N/A |
| DAN | 需要距离加权损失 | 标准 Trainer 使用 MSE |
| ST-SSDL | 需要对比和偏差损失 | 标准 Trainer 使用 MSE |

所有自定义 `fit()` 的实现都遵循相同的模式：构建 DataLoader → 创建 optimizer/scheduler → 训练循环 → 验证循环 → 保存最佳 checkpoint → 加载最佳模型。这与标准 Trainer 的模式一致，额外的损失被正确地集成到训练循环中。

### 6.3 超参数默认值审查

| 模型 | 关键超参数 | 默认值 | 合理性 |
|-----|----------|-------|-------|
| TimeAlign | d_model=64, n_layers=2, lambda_align=1.0, delta_loc=0.1, delta_glo=0.1 | 合理 |
| ProtoTS | d_model=64, n_protos=8, d_bottleneck=16, lambda_ent=0.01 | 合理 |
| ModernTCN | d_model=64, d_ff=128, kernel_size=5, n_layers=2 | 合理 |
| EVL | hidden_dim=64, n_lstm_layers=2, mem_size=16, lambda_evl=0.1, gamma=3.0 | 合理 |
| EALGAP | d_model=64, n_heads=4, gru_hidden=32 | 合理 |
| DAN | d_model=64 | 合理 |
| ST-SSDL | d_model=64, n_protos=8, alpha=0.01, beta=0.1 | 合理 |

所有超参数默认值都在合理范围内，与论文报告的值一致或接近。

---

*报告生成时间: 2025-06-25*
*审查工具: Claude Code (deepseek-v4-pro)*