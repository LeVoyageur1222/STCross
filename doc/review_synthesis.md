# STCross 项目代码审查综合分析报告

**审查召集人**: 数据挖掘教授  
**审查日期**: 2026-06-25  
**输入报告**: 5 份博士生独立审查报告  
**审查范围**: 数据处理、模型实现、训练管线、评测管线、交叉城市设定  
**覆盖文件**: 核心代码库共 17 个核心文件 + 所有模型实现文件

---

## 1. 问题汇总

### 问题发现总览

| 严重程度 | 数量 | 涉及报告 |
|----------|------|----------|
| HIGH (严重) | 4 | review_model_groupAB (GWNET 输出形状), review_model_groupAB (交叉城市 Group B 维度), review_model_groupCD (DAN 架构简化), review_model_groupCD (ST-SSDL 架构简化) |
| MEDIUM (中等) | 9 | review_model_groupAB (STGCN 输出层), review_model_groupCD (TimeAlign RevIN, EALGAP PDF, EALGAP QKV, EALGAP instance-zscore, ST-SSDL historical anchoring, ST-SSDL 对比损失, ST-SSDL 双分支), review_cross_city (交叉城市无预热), review_training_pipeline (自定义 fit 代码重复) |
| LOW (轻微) | 17 | 跨 5 份报告 |

### 1.1 严重问题 (HIGH)

| 优先级 | 编号 | 问题 | 文件 | 行号 | 来源报告 | 影响 |
|--------|------|------|------|------|----------|------|
| P0 | H1 | **GWNET 输出形状错误** -- `_net_forward()` 末尾 `x.squeeze(-1)` 对维度为 5 的时间维度无效，实际输出形状为 `(B, P, N, 5)` 而非 `(B, P, N)` | `model/gwnet.py` | ~167 | review_model_groupAB | **实验结果完全不可信。** PyTorch MSELoss 会广播 `(B,P,N,5)` 与 `(B,P,N)` 目标，导致模型优化了一个与预期完全不同的目标函数。所有 GWNET 实验结果必须作废。 |
| P0 | H2 | **交叉城市 Group B 模型维度不匹配** -- STGCN/DCRNN/GWNET/STGODE 在 `registry.py` 中以 `N_target` 和 `adj_target` 构建，但交叉城市训练时接收 `N_source` 节点数据，若 `N_source != N_target` 则图形卷积层 `einsum` 维度不匹配 | `model/registry.py` + `eval/evaluator.py` | 多个 | review_model_groupAB + review_cross_city_setting | **交叉城市 Group B 模型无法运行**（若 NYC 和 CHI 节点数不同）。需先确认两城市节点数是否一致，若不一致则需设计 N-alignment 机制。 |
| P1 | H3 | **DAN 架构严重简化** -- 论文使用 CNN+LSTM 编码器-解码器，实现使用纯 FC (`PolarEncoder`/`PolarDecoder`)，丢失了 CNN 的局部特征提取和 LSTM 的序列建模能力 | `model/dan.py` | 全文 | review_model_groupCD | **与论文不忠实。** 性能可能显著低于论文报告值。纯 FC 无法有效建模长序列的时间依赖。代码中明确标注了简化，但作为 baseline 其参考价值有限。 |
| P1 | H4 | **ST-SSDL 架构严重简化** -- 论文使用 GCRU (Graph Conv GRU) 骨干，实现使用纯 Linear 编码器，且缺少 dual-branch encoding 和 running mean anchoring | `model/stssdl.py` | 全文 | review_model_groupCD | **与论文不忠实。** 代码标注为 "ST-SSDL-lite"，但作为 baseline 对比可能严重低估原始方法的性能。 |

### 1.2 中等问题 (MEDIUM)

| 优先级 | 编号 | 问题 | 文件 | 行号 | 来源报告 | 影响 |
|--------|------|------|------|------|----------|------|
| P2 | M1 | **自定义 fit 与标准 Trainer 间大量代码重复** -- 5 个自定义 fit 模型 (EVL/TimeAlign/ProtoTS/DAN/STSSDL) 重复实现了完整的训练循环模板代码，与 `train/trainer.py` 的 `Trainer` 高度重复 | `model/evl.py`, `model/timealign.py`, `model/protots.py`, `model/dan.py`, `model/stssdl.py`, `train/trainer.py` | 各 fit() 方法 | review_training_pipeline | 维护成本高。修改一处逻辑（如早停策略、日志格式）需同步修改 6 个文件。 |
| P2 | M2 | **STGCN 输出层与论文偏差** -- 输出层使用 `TemporalConv(Ko) → LayerNorm → TemporalConv(1) → FC`，论文使用更简单的单次 TemporalConv + FC | `model/stgcn.py` | 输出层部分 | review_model_groupAB | 可能影响与论文结果的直接可比性。 |
| P2 | M3 | **TimeAlign RevIN 状态管理竞态风险** -- `_train_step()` 中先调用 `self.revin.normalize(x)` 再调用 `self.revin.normalize(y)`，虽当前代码通过局部变量保存了状态，但代码结构脆弱 | `model/timealign.py` | `_train_step()` | review_model_groupCD | 当前代码功能正确，但缺乏健壮性。未来修改容易引入 bug。 |
| P2 | M4 | **EALGAP QKV MLP 使用 Softmax 激活** -- 特征变换层使用 Softmax 而非标准的 GELU/ReLU，可能导致梯度消失或信息丢失 | `model/ealgap.py` | QKV MLP 构建处 | review_model_groupCD | 可能影响模型收敛和性能。Softmax 通常用于 attention weights 而非特征变换。 |
| P2 | M5 | **EALGAP Instance z-score vs same-slot 偏差** -- LocalExtremeModule 使用 instance z-score 而非论文中的 same-slot 历史统计偏差 | `model/ealgap.py` | LocalExtremeModule | review_model_groupCD | 对捕捉时间模式异常的敏感度不同，可能影响极端事件检测能力。 |
| P2 | M6 | **ST-SSDL 对比损失公式不一致** -- 实现使用 triplet margin loss 作用于样本-原型距离，论文可能使用 InfoNCE 风格或直接作用于原型向量的对比损失 | `model/stssdl.py` | `_compute_contrastive_loss()` | review_model_groupCD | 损失函数语义不同，可能影响原型学习质量。 |
| P2 | M7 | **ST-SSDL Historical anchoring 使用 instance mean** -- 论文使用 running mean (滑动平均)，实现使用 `x.mean(dim=1)` (instance mean) | `model/stssdl.py` | encoder 部分 | review_model_groupCD | 无法捕获长期趋势，降低了偏差学习的有效性。 |
| P2 | M8 | **交叉城市无目标城市验证集** -- 交叉城市模式下验证集来自源城市，早停准则基于源城市验证损失，与目标城市性能可能不相关 | `eval/evaluator.py` | 第 62-74 行 | review_cross_city_setting + review_dataflow_architecture | 早停可能在非最优位置停止，影响目标城市性能。这是无监督域适应的已知局限，但需在论文中明确讨论。 |
| P2 | M9 | **EALGAP PDF 动态计算** -- 论文可能使用预计算的 PDF 值作为输入特征，实现每个 batch 动态计算 | `model/ealgap.py` | GlobalTrendModule | review_model_groupCD | 不完全忠实于论文的数据处理流程，但动态计算更灵活。 |

### 1.3 轻微问题 (LOW) -- 汇总列表

| 编号 | 问题 | 文件 | 来源报告 |
|------|------|------|----------|
| L1 | DCRNN 无 scheduled sampling (论文有) | `model/dcrnn.py` | review_model_groupAB |
| L2 | TimesNet MPS 设备强制回退 CPU | `model/timesnet.py` | review_model_groupAB |
| L3 | STGODE 双分支共用同一邻接矩阵 (论文使用不同矩阵) | `model/stgode.py` | review_model_groupAB |
| L4 | PatchTST 短序列配置 (H=12, patch_len=4, stride=2) 仅 5 个 patches | `model/patchtst.py` | review_model_groupAB |
| L5 | DLinear kernel_size=3 对 H=12 序列平滑效果有限 | `model/dlinear.py` | review_model_groupAB |
| L6 | `model.predict()` 无进度条 | `model/torch_base.py` | review_training_pipeline |
| L7 | Checkpoint 路径为相对路径 | `train/trainer.py` | review_training_pipeline |
| L8 | `calc_pcc` 零方差返回 0.0 而非 nan | `metrics/metrics.py` | review_training_pipeline |
| L9 | `_config_matches` 不检查 `test_steps` | `experiments/run_all.py` | review_training_pipeline |
| L10 | `run_ha_baseline.py` 硬编码 `output_dir` | `experiments/run_ha_baseline.py` | review_training_pipeline |
| L11 | `predict()` 中 batch_size 硬编码 2x 放大 | `model/torch_base.py` | review_dataflow_architecture |
| L12 | 训练中无 NaN/Inf 检测 | `train/trainer.py` | review_dataflow_architecture |
| L13 | 结果缓存机制无强制重跑选项 | `experiments/run_all.py` | review_dataflow_architecture |
| L14 | `build_flow_data.py` 硬编码日期范围 | `data_preprocess/build_flow_data.py` | review_dataflow_architecture |
| L15 | 极端天气类型 (extreme_heat/cold/uv) 已定义但未启用 | `data_preprocess/build_weather_mask.py` | review_dataflow_architecture |
| L16 | EVL LSTM 仅使用最后一层隐藏状态 | `model/evl.py` | review_model_groupCD |
| L17 | TimeAlign `_patchify` 使用 `unfold` 可能丢失部分时间步 | `model/timealign.py` | review_model_groupCD |

---

## 2. 交叉验证

### 2.1 独立发现的问题 -- 被多位学生共同确认

| 问题 | 发现者 | 强度 |
|------|--------|------|
| **Group B 模型交叉城市 N 维度不匹配** | review_model_groupAB (详细分析) + review_cross_city_setting (确认) | **强确认** -- 两位学生从不同角度 (模型忠实度 + 评测 setting) 独立发现了同一问题，且相互印证了分析。 |
| **交叉城市无目标城市验证** | review_cross_city_setting (Issue 2) + review_dataflow_architecture (M1) | **强确认** -- 两位学生分别从评测 setting 和数据流架构角度发现，说明了同一设计决策的两个方面。 |
| **自定义 fit 代码重复** | review_training_pipeline (M1) + review_model_groupCD (各模型 fit 审查) | **中等确认** -- review_training_pipeline 从训练管线角度指出重复，review_model_groupCD 从各模型 fit 实现中确认了重复模式。 |

### 2.2 互补发现 -- 同一问题的不同侧面

| 主题 | review_cross_city_setting | review_training_pipeline | review_model_groupAB | review_model_groupCD | review_dataflow_architecture |
|------|--------------------------|-------------------------|---------------------|---------------------|------------------------------|
| 交叉城市 scaler 设计 | 确认 source/target scaler 分离设计正确 | -- | 确认 distribution shift 是设计意图 | -- | 确认数据流正确，无数据泄露 |
| 时间序列划分 | 确认 month-aligned 正确 | -- | -- | -- | 确认 train/val/test 无重叠，无泄露 |
| 模型兼容性 | 指出 Group B N 维度问题 | 确认自定义 fit 兼容 | 详细分析 Group B 维度问题 | 确认所有 Group C/D 模型 N-independent | -- |
| 指标计算 | -- | 确认空数组/零方差/阈值防护 | -- | -- | 确认 MAPE 阈值等正确 |

### 2.3 无冲突发现

所有 5 份报告在核心结论上**无冲突**。不同学生关注的侧面不同，但相互印证：
- 对评测管线的正确性评价高度一致（均为通过）
- 对数据流无泄露的结论一致
- 对 Group B 模型交叉城市问题的认识一致

---

## 3. 关键风险分析

### 3.1 H1: GWNET 输出形状错误 -- 严重程度: CRITICAL

**技术细节**:

`model/gwnet.py` 中 `_net_forward()` 方法，经过 8 层因果卷积后，时间维度 `T = H + 1(padding) - 8(layers) = 12 + 1 - 8 = 5`。`end_conv_2` 是 `Conv2d(end_channels, P, kernel_size=1)`，输出形状为 `(B, P, N, 5)`。末行的 `x.squeeze(-1)` 仅移除大小为 1 的维度，对大小为 5 的维度无效。

**错误机制**:
```python
x = self.end_conv_2(x)       # (B, P, N, 5)
return x.squeeze(-1)          # (B, P, N, 5) -- 无变化！
```

**损失函数影响**: PyTorch `nn.MSELoss()` 在计算 `loss(pred, Y)` 时，`pred` 为 `(B, P, N, 5)`，`Y` 为 `(B, P, N)`。广播机制将 `Y` 扩展为 `(B, P, N, 5)`，MSELoss 计算 `(pred - Y_expanded)²` 在所有 5 个时间维度上求和。这意味着：
- 模型优化的不是"预测 P 步"的误差，而是"5 个不同时间余量各自预测 P 步"的平均误差
- 模型学到的映射与预期完全不同
- **所有 GWNET 实验结果不可信，必须全部作废重跑**

**风险评估**: 极高。这是最严重的 bug，影响整个 GWNET baseline 的有效性。

**修复方案**:
```python
# 方案 A: 取最后一个时间步（与原论文 nnzhan/Graph-WaveNet 做法一致）
return x[..., -1]  # (B, P, N)

# 方案 B: 如果希望保留更多时间信息，使用时间聚合
return x.mean(dim=-1)  # (B, P, N)
```

建议使用方案 A，与原始 GWNET 论文保持一致。

### 3.2 H2: 交叉城市 Group B 模型维度不匹配 -- 严重程度: HIGH (条件性)

**技术细节**:

`model/registry.py` 中所有 Group B 模型 (STGCN, DCRNN, GWNET, STGODE) 的构建函数均以 `target_city` 的邻接矩阵和节点数构建模型：
```python
# registry.py - build_stgcn 示例
adj = np.load(config.adj_path(config.target_city))
N = adj.shape[0]
```

`eval/evaluator.py` 交叉城市路径中，训练数据来自源城市 (`source_flow`)，其节点数为 `N_source`。若 `N_source != N_target`，则 `_net_forward()` 收到的输入张量空间维度为 `N_source`，但模型内部 `ChebGraphConv.gso` 形状为 `(N_target, N_target)`，`einsum("nm,btmc->btnc", self.gso, x)` 会因 `n=N_target` 与 `x` 的 `m=N_source` 不匹配而失败。

**前置条件**: 需要确认 NYC 和 CHI 的传感器节点数是否相同。如果相同，此问题失效；如果不同，此问题会导致所有 Group B 交叉城市实验崩溃。

**风险评估**: 需要立即验证。如果两城市节点数确实不同，需要以下之一：
- (a) 选择相同数量的节点（如按覆盖率取前 N 个节点）
- (b) 实现节点对齐机制（如空间插值或节点匹配）
- (c) 将 Group B 模型限制为同城实验

**修复方案**: 无论验证结果如何，建议在 `evaluator.py` 中添加防御性断言：
```python
if is_cross_city and input_mode == "all_nodes":
    assert train_X.shape[2] == self.model.N, \
        f"Node mismatch: source has {train_X.shape[2]} nodes, model expects {self.model.N}"
```

### 3.3 H3: DAN 架构简化 -- 严重程度: HIGH

**技术细节**:

`model/dan.py` 中 `PolarEncoder` 和 `PolarDecoder` 使用纯 FC 层：
```python
class PolarEncoder(nn.Module):
    def __init__(self, H, d_model):
        self.encoder = nn.Sequential(
            nn.Linear(H, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
```

论文中的 DAN 使用 CNN + LSTM 编码器-解码器，以捕捉局部模式 (CNN) 和长期依赖 (LSTM)。纯 FC 不具有这些能力。

**风险评估**: 中高。虽然代码标注了简化，但：
- 简化版 DAN 的性能可能显著低于论文报告值
- 在论文中引用这些结果作为 baseline 时，需要明确说明是简化版
- 如果想公正地评估 DAN 在交通流预测中的表现，应使用 CNN+LSTM 版本

**修复方案**:
```python
class PolarEncoder(nn.Module):
    def __init__(self, H, d_model, hidden_dim=64):
        # CNN for local patterns
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        # Projection
        self.proj = nn.Linear(hidden_dim, d_model)
```

### 3.4 H4: ST-SSDL 架构简化 -- 严重程度: HIGH

**技术细节**:

`model/stssdl.py` 将论文的 GCRU (Graph Conv GRU) 骨干替换为纯 Linear 编码器，且缺少 dual-branch encoding 和 running mean anchoring。代码标注为 "ST-SSDL-lite"。

**风险评估**: 中高。与 DAN 类似，简化版 ST-SSDL 的性能可能严重低估原始方法。由于 ST-SSDL 是较新的方法 (NeurIPS 2025)，如果论文中将其作为 baseline，简化的架构可能引发审稿人的质疑。

**修复方案**: 在保持 N-independence 的前提下，可以考虑：
- 使用 LSTM 或 TCN 替代 GCRU 作为编码器骨干（至少保留序列建模能力）
- 添加 weekly-average branch 的 dual-branch encoding
- 将 instance mean 替换为 exponential moving average 作为 historical anchoring

---

## 4. 修改建议 (按优先级)

### 优先级 P0 (实验开始前必须修复)

| 序号 | 修改项 | 负责文件 | 具体修改 | 预计工时 |
|------|--------|----------|----------|----------|
| 1 | **修复 GWNET 输出形状** | `model/gwnet.py` 末行 | `x.squeeze(-1)` → `x[..., -1]` | 5 分钟 |
| 2 | **验证 NYC/CHI 节点数是否一致** | `data_preprocess/` 下数据文件 | 检查 `flow_NYC_*.npy` 和 `flow_CHI_*.npy` 的 `shape[1]` | 15 分钟 |
| 3 | **添加 N 维度防御性断言** | `eval/evaluator.py` 或 `model/torch_base.py` | 在训练前检查 `train_X.shape[2] == model.N` | 15 分钟 |

### 优先级 P1 (实验开始前建议修复)

| 序号 | 修改项 | 负责文件 | 具体修改 | 预计工时 |
|------|--------|----------|----------|----------|
| 4 | **DAN 编码器改为 CNN+LSTM** | `model/dan.py` | 重构 `PolarEncoder`/`PolarDecoder` 为 CNN+LSTM | 2-3 小时 |
| 5 | **ST-SSDL 编码器添加 LSTM/TCN** | `model/stssdl.py` | 将 Linear 编码器替换为 LSTM 或 TCN | 2-3 小时 |
| 6 | **重跑 GWNET 全部实验** | 实验脚本 | H1 修复后必须重跑所有 GWNET 相关实验 | 取决于 GPU 资源 |

### 优先级 P2 (建议在论文撰写前修复)

| 序号 | 修改项 | 负责文件 | 具体修改 | 预计工时 |
|------|--------|----------|----------|----------|
| 7 | **重构训练循环，提取公共组件** | `train/trainer.py` + 5 个自定义 fit 模型 | 提取 `run_training_loop` 通用函数或 `TrainingLoopMixin`，接受 `train_step_fn` 和 `val_step_fn` 回调 | 4-6 小时 |
| 8 | **修复 EALGAP QKV MLP 激活函数** | `model/ealgap.py` | `Softmax` → `GELU` 或 `ReLU` | 10 分钟 |
| 9 | **修复 EALGAP LocalExtremeModule** | `model/ealgap.py` | Instance z-score → same-slot 偏差 (或添加 running statistics) | 2-3 小时 |
| 10 | **修复 ST-SSDL 对比损失** | `model/stssdl.py` | Triplet loss → 论文原始 InfoNCE 或原型距离对比损失 | 2-3 小时 |
| 11 | **修复 ST-SSDL historical anchoring** | `model/stssdl.py` | `x.mean(dim=1)` → EMA 或预计算历史均值 | 1-2 小时 |
| 12 | **修复 TimeAlign RevIN 状态管理** | `model/timealign.py` | 使用两个独立 RevIN 实例或让 `normalize()` 返回 `(output, mean, std)` | 1-2 小时 |
| 13 | **STGCN 输出层对齐论文** | `model/stgcn.py` | 简化输出层为论文的单次 TemporalConv + FC | 1-2 小时 |

### 优先级 P3 (论文撰写后可考虑)

| 序号 | 修改项 | 负责文件 | 具体修改 |
|------|--------|----------|----------|
| 14 | DCRNN 添加 scheduled sampling | `model/dcrnn.py` | 在 decoder 中实现 scheduled sampling |
| 15 | STGODE 使用独立语义邻接矩阵 | `model/stgode.py` | 为第二 ODEG 分支构建 DTW 邻接矩阵 |
| 16 | 添加 `predict()` 进度条 | `model/torch_base.py` | 在循环外包装 tqdm |
| 17 | 添加训练 NaN/Inf 检测 | `train/trainer.py` | `torch.isnan(loss)` 检测 |
| 18 | 添加 `--force` 重跑选项 | `experiments/run_all.py` | 跳过缓存检查 |
| 19 | 统一 checkpoint 路径 | `train/trainer.py` | 使用绝对路径或基于项目根目录 |
| 20 | `calc_pcc` 零方差返回 nan | `metrics/metrics.py` | 更语义化的返回值 |
| 21-24 | 其他 LOW 问题 | 各文件 | 参见 1.3 节 |

---

## 5. 实验准备就绪度评估

### 5.1 总体评估: 基本就绪，但有 1 个阻塞性问题和 3 个高优先级问题

| 维度 | 就绪度 | 说明 |
|------|--------|------|
| 数据管线 | 就绪 | 数据加载、划分、标准化、滑窗构建均正确，无数据泄露 |
| 评测管线 | 就绪 | 同城/交叉城市评测流程正确，scaler 设计合理，指标计算严谨 |
| 训练管线 | 就绪 | Trainer 功能完善，早停/学习率调度/梯度裁剪/checkpoint 管理完整 |
| Group A 模型 (时序) | 就绪 | LSTM, DLinear, PatchTST, TimesNet, HA 实现正确 |
| Group B 模型 (时空图) - 同城 | 就绪 (GWNET 除外) | STGCN, DCRNN, STGODE 同城正确，GWNET 需修复 |
| Group B 模型 (时空图) - 交叉城市 | 待验证 | 需确认节点数一致性 |
| Group C 模型 (可迁移) | 就绪 (有改进空间) | TimeAlign, ProtoTS, ModernTCN 忠实度高 |
| Group D 模型 (极端事件) | 部分就绪 | EVL 忠实，EALGAP 有偏差，DAN 和 ST-SSDL 简化严重 |
| 实验编排 | 就绪 | 批量运行、结果缓存、错误隔离、摘要生成均正常 |

### 5.2 阻塞性问题 (开始实验前必须解决)

**1 个**:
- **H1: GWNET 输出形状错误** (P0)。修复简单，但修复后必须重跑所有 GWNET 实验。

### 5.3 建议在开始正式实验前解决

- **H2: 验证交叉城市节点数** (P0)。如果 NYC 和 CHI 节点数不同，需决定 Group B 模型的交叉城市策略。
- **H3: DAN 架构** (P1)。如果要将 DAN 作为 baseline，建议修复。
- **H4: ST-SSDL 架构** (P1)。同上。

### 5.4 可以开始但需注意的问题

- Group D 模型 (EALGAP, DAN, ST-SSDL) 与论文有偏差，结果引用时需注明是简化版
- 交叉城市无目标城市验证集 (M8)，需在论文中讨论
- 自定义 fit 代码重复 (M1)，不影响功能但增加维护成本

---

## 6. 团队分工建议

### 6.1 按专业领域分工

| 学生 | 专长 | 建议负责 | 对应问题 |
|------|------|----------|----------|
| **学生 A** (review_cross_city_setting) | 评测系统设计 | 交叉城市 N 维度验证与修复 (H2) + 添加防御性断言 (P0-3) + 交叉城市验证集讨论 (M8) | H2, M8, L14 |
| **学生 B** (review_training_pipeline) | 训练管线 | 重构训练循环 (M1/P2-7) + 添加 NaN/Inf 检测 (L12) + 统一 checkpoint 路径 (L19) | M1, L12, L19 |
| **学生 C** (review_model_groupAB) | 图神经网络 | GWNET 输出形状修复 (H1/P0-1) + STGCN 输出层对齐 (M2/P2-13) + DCRNN scheduled sampling (L1/P3-14) + STGODE 双邻接矩阵 (L3/P3-15) | H1, M2, L1, L3, L4, L5 |
| **学生 D** (review_model_groupCD) | 时序建模 | DAN 编码器修复 (H3/P1-4) + ST-SSDL 编码器修复 (H4/P1-5) + EALGAP 修复 (M4/M5/M9/P2-8,9) + ST-SSDL 损失修复 (M6/M7/P2-10,11) + TimeAlign RevIN (M3/P2-12) | H3, H4, M3, M4, M5, M6, M7, M9 |
| **学生 E** (review_dataflow_architecture) | 数据工程 | 数据验证层 + 单元测试 (review_dataflow_architecture 建议) + predict batch_size 可配置化 (L11) + `--force` 选项 (L13) + 天气类型文档化 (L15) | L11, L13, L15 |

### 6.2 协作修复项

| 问题 | 主负责 | 协作 | 原因 |
|------|--------|------|------|
| H2 交叉城市 N 维度 | 学生 A | 学生 C | 学生 A 从评测角度验证，学生 C 从模型角度修复 |
| GWNET 重跑 | 学生 C | 学生 B | 学生 C 修复代码，学生 B 确保训练管线正常 |
| 训练循环重构 | 学生 B | 学生 D | 学生 D 的 5 个自定义 fit 模型需要适配新接口 |

### 6.3 建议工作流

```
第 1 天 (P0 修复):
- 学生 C: 修复 GWNET 输出形状 (H1)
- 学生 A: 验证 NYC/CHI 节点数，添加防御性断言 (H2)

第 2-3 天 (P1 修复):
- 学生 D: 修复 DAN 和 ST-SSDL 编码器 (H3, H4)
- 学生 C: 重跑 GWNET 实验

第 4-7 天 (P2 修复):
- 学生 B: 重构训练循环 (M1)
- 学生 D: 修复 EALGAP, ST-SSDL, TimeAlign 问题 (M3-M7, M9)
- 学生 A: 文档化交叉城市无验证集问题 (M8)

第 8-10 天 (P3 修复):
- 各学生: 各自的 P3 问题
- 学生 E: 数据验证和单元测试
```

---

## 7. 总结

### 7.1 核心结论

1. **代码库整体质量良好**。数据处理管线、评测管线、训练管线、实验编排等基础设施设计正确，无数据泄露，架构分层清晰。5 位学生的审查结论在这点上高度一致。

2. **存在 1 个阻塞性 bug (GWNET 输出形状)**，修复简单但影响重大。

3. **Group B 模型交叉城市兼容性需要立即验证**，可能影响 4 个模型的交叉城市实验计划。

4. **Group D 模型中有 2 个 (DAN, ST-SSDL) 与论文存在显著架构偏差**，需要考虑是否修复或标注为"简化版"。

5. **5 份审查报告相互印证，无冲突发现**，提升了审查结论的可信度。

### 7.2 关键数字

- 审查覆盖: 17 个核心文件 + 15 个模型文件
- 发现问题: 4 个 HIGH，9 个 MEDIUM，17 个 LOW
- 交叉验证: 3 个问题被多位学生独立发现
- 阻塞性: 1 个必须在实验前修复

### 7.3 下一步行动

1. **立即行动**: 修复 GWNET 输出形状 (H1)，验证 NYC/CHI 节点数 (H2)
2. **本周内**: 讨论 DAN 和 ST-SSDL 的修复策略 (H3, H4)，决定是否修复或标注为简化版
3. **两周内**: 完成 P2 优先级修复，开始正式实验

---

*报告生成时间: 2026-06-25*
*综合审查人: 数据挖掘教授 (AI 辅助)*
*输入: 5 份博士生独立审查报告*