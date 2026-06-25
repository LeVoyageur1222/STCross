# STCross 训练与评测基础设施审查报告

> 审查日期: 2026-06-25
> 审查人: 机器学习系统工程方向
> 审查范围: 训练循环、模型保存与加载、评测流程、结果报告、批量实验、HA baseline、指标计算、自定义 fit 兼容性

---

## 1. 审查范围

| 序号 | 文件 | 组件 | 角色 |
|------|------|------|------|
| 1 | `train/trainer.py` | Trainer / TrainConfig | 标准训练循环 |
| 2 | `model/torch_base.py` | TorchModel 基类 | 模型抽象层，含 fit/predict/forward |
| 3 | `eval/evaluator.py` | Evaluator | 评测主流程 |
| 4 | `eval/reporter.py` | Reporter | 结果打印与 JSON 保存 |
| 5 | `experiments/run_all.py` | Batch Runner | 批量实验编排 |
| 6 | `experiments/run_model.py` | Single Model Runner | 单模型评测入口 |
| 7 | `experiments/run_ha_baseline.py` | HA Baseline Runner | 专用 baseline 脚本 |
| 8 | `metrics/metrics.py` | Metrics | MAE / RMSE / MAPE / PCC / R2 |
| 9 | `model/registry.py` | Model Registry | 模型注册与构建 |
| 10 | `model/evl.py` | EVLModel | 自定义 fit (双 loss) |
| 11 | `model/timealign.py` | TimeAlignModel | 自定义 fit (对齐 loss) |
| 12 | `model/protots.py` | ProtoTSModel | 自定义 fit (熵正则化) |
| 13 | `model/dan.py` | DANModel | 自定义 fit (距离加权 loss) |
| 14 | `model/stssdl.py` | STSSDLModel | 自定义 fit (三 loss) |
| 15 | `model/ha.py` | HAModel | 无需训练的 baseline |

---

## 2. 逐组件详细审查

### 2.1 Trainer (`train/trainer.py`)

**进度条**: 有。`_run_epoch` 使用 `tqdm(loader, desc=..., leave=False, unit="batch")` 并设置 `pbar.set_postfix(loss=...)` 实时显示 moving average loss。同时每个 epoch 结束后有 `print` 输出 epoch 级 loss 和 lr。

**结论**: 通过。

**潜在问题**: 无。

---

### 2.2 TorchModel (`model/torch_base.py`)

**fit 方法**: 调用 `build_dataloader` 构建 train/val loader，打印训练信息，然后创建 `Trainer` 实例调用 `trainer.train()`。逻辑清晰。

**checkpoint 路径**: Trainer 内 `ckpt_path = os.path.join(self.config.checkpoint_dir, f"{model_name}_best.pt")`。默认 `checkpoint_dir="checkpoints"`，即相对路径。`os.makedirs` 会创建目录，因此不会崩溃。

**评测时模型加载**: Trainer 在训练结束后执行 `model.load_state_dict(torch.load(ckpt_path, ...))`，将最佳 checkpoint 加载回模型。Evaluator 在 `model.fit()` 之后立即调用 `model.predict()`，因此模型处于最佳状态。

**结论**: 通过。

**潜在问题**:
- **严重程度: 低** checkpoint 路径为相对路径，`checkpoint_dir` 固定为 `"checkpoints"`，不随实验设置变化。如果多个实验同时运行会共享同一目录，但不同模型名称会区分文件。

---

### 2.3 Evaluator (`eval/evaluator.py`)

**评测流程**:
1. 加载目标城市流量数据
2. 加载异常标签 mask
3. 按 `train_steps` / `val_steps` 划分数据
4. 对目标数据进行 `PerNodeStandardScaler` 归一化
5. 跨城市场景: 额外加载源城市数据，用源数据训练
6. 构建滑动窗口训练/验证集
7. 调用 `model.fit(train_X, train_Y, val_X, val_Y)`
8. 构建测试集，调用 `model.predict(test_X)`
9. 对每个子集 (full/anomaly/normal) 计算指标

**进度条**: 评测阶段没有进度条。`model.fit()` 内部有（由 Trainer 或自定义 fit 提供），但 `model.predict()` 没有进度条。对于大规模测试集，这可能影响用户体验。

**模型加载**: 见 2.2 节，模型在 fit 过程中已加载最佳 checkpoint。

**结论**: 通过。

**潜在问题**:
- **严重程度: 低** `model.predict()` 没有进度条，对于大规模测试数据可能不直观。

---

### 2.4 Reporter (`eval/reporter.py`)

**结果打印**: 清晰的分隔线、表格格式，包含模型名、城市对、H/P、数据划分、指标值。

**JSON 保存**: 包含完整元信息:
- `model`: 模型名
- `source_city`, `target_city`: 城市对
- `history_len`, `predict_len`: 窗口参数
- `train_steps`, `val_steps`, `test_steps`: 数据划分
- `anomaly_min_ratio`: 异常阈值
- `num_test_samples`: 测试样本数
- `subsets`: 每个子集的指标字典

**文件名**: `{model}_{source_city}_to_{target_city}_{timestamp}.json`。包含时间戳，避免覆盖。

**结论**: 通过。

**潜在问题**: 无。

---

### 2.5 Batch Runner (`experiments/run_all.py`)

**跳过已完成实验**: 有。通过检查 `output_dir` 中已存在的 JSON 文件，匹配 `model_name`、`source_city`、`target_city`、以及 `EvalConfig` 的关键参数（`history_len`, `predict_len`, `anomaly_min_ratio`, `train_steps`）。如果匹配则跳过。

**错误隔离**: 有。每个 (model, setting) 组合被 `try/except` 包裹，失败时打印 traceback 并 `continue`。

**模型独立运行**: 有。每个模型独立构建、独立评测，互不干扰。

**编译摘要**: `compile_summary` 生成 Markdown 表格，按模型和城市对排序。

**CLI 参数**: 支持 `--models`、`--settings`、`--output_dir`、`--epochs`、`--batch_size`、`--lr`。

**结论**: 通过。

**潜在问题**:
- **严重程度: 低** 跳过已完成实验的匹配逻辑依赖于从 JSON 中读取的 `_config_matches` 字典。如果旧结果文件格式不完整（缺少 `history_len` 等字段），会跳过匹配失败，重新运行实验。这是合理的保守行为。
- **严重程度: 低** `_config_matches` 的 `test_steps` 不在匹配条件中，这意味着如果 `EvalConfig` 的 `train_steps` 或 `val_steps` 变化引起 `test_steps` 变化，旧结果不会被自动跳过。虽然 `test_steps` 由 `train_steps` 和 `val_steps` 决定（`test_steps = T - train_steps - val_steps`），但如果 `T` 变化（如数据更新），可能需要额外检查。

---

### 2.6 Single Model Runner (`experiments/run_model.py`)

提供 CLI 接口，支持 `--model`、`--source`、`--target` 等参数。调用 Evaluator 和 Reporter 完成评测。

**结论**: 通过。

**潜在问题**: 无。

---

### 2.7 HA Baseline Runner (`experiments/run_ha_baseline.py`)

**特殊处理**: 是。HA 模型不需要训练（`fit` 仅计算训练数据上的统计量）。Evaluator 中通过 `isinstance(model, HAModel)` 检查，跳过归一化，直接调用 `predict_with_timestamps`。

**流程**: 对 NYC 和 CHI 分别运行 same-city 评测。

**结论**: 通过。

**潜在问题**:
- **严重程度: 低** `run_ha_baseline.py` 硬编码了 `output_dir="results/"`，与其他实验脚本通过 `--output_dir` 参数设置不同。但这是专门的 baseline 脚本，使用上可以接受。

---

### 2.8 Metrics (`metrics/metrics.py`)

**空数组防护**:
- `calc_mae`: `len(y_true) == 0` 返回 `nan`
- `calc_rmse`: `len(y_true) == 0` 返回 `nan`
- `calc_mape`: `len(y_true) == 0` 返回 `nan`；`mask.sum() == 0` 返回 `inf`
- `calc_pcc`: `len(y_true) == 0` 返回 `nan`
- `calc_r2`: `len(y_true) == 0` 返回 `nan`

**零方差防护**:
- `calc_pcc`: `np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8` 返回 `0.0`
- `calc_r2`: `ss_tot == 0` 返回 `0.0`

**MAPE 阈值**: 使用 `threshold=0.5`，只对绝对值大于 0.5 的 ground truth 计算 MAPE。这是交通流预测中的常见做法，避免小流量值导致的百分比爆炸。

**结论**: 通过。

**潜在问题**:
- **严重程度: 低** `calc_pcc` 在零方差情况下返回 `0.0`，可能掩盖数据质量问题。建议返回 `nan` 以示区分（当 `y_true` 或 `y_pred` 的方差为零时，相关系数无定义）。但 `0.0` 作为保守值也是合理的。

---

### 2.9 自定义 fit 模型兼容性

所有 5 个自定义 fit 的模型（EVL、TimeAlign、ProtoTS、DAN、STSSDL）均:

1. **正确实现了 `fit(self, train_X, train_Y, val_X, val_Y)` 接口**，与 `BaseModel` 和 `Evaluator` 的调用方式完全兼容。
2. **内部构建 DataLoader**，使用 `build_dataloader`。
3. **包含完整的训练循环**：epoch 迭代、进度条（tqdm）、早停、学习率调度、checkpoint 保存与加载。
4. **训练使用自定义 loss，验证使用标准 MSE/L1**，与标准 Trainer 的验证方式一致。
5. **在 fit 结束时加载最佳 checkpoint**，确保 `predict` 使用最佳模型状态。

**EVL**: 自定义 loss = MSE + lambda * EVL（极端值 loss）。验证时使用标准 MSE。

**TimeAlign**: 自定义 loss = L1_pred + L1_recon + lambda * align_loss。验证时使用 L1 loss。

**ProtoTS**: 自定义 loss = L1 + lambda * H(w)（熵正则化）。验证时使用 L1 loss。

**DAN**: 自定义 loss = loss_far + loss_near + loss_final（距离加权 MSE）。验证时使用 MSE。

**STSSDL**: 自定义 loss = L1_pred + alpha * contrastive + beta * deviation。验证时使用 L1 loss。

**结论**: 所有自定义 fit 模型与标准 Evaluator 和 Batch Runner 完全兼容。每个模型的 fit 都是独立的、自包含的训练循环，不依赖 Trainer 类。

**潜在问题**:
- **严重程度: 中** 自定义 fit 与标准 Trainer 之间存在大量代码重复。每个自定义 fit 都重复实现了几乎相同的模板代码（build_dataloader、optimizer、scheduler、ckpt_path、early_stop、save/load、print 格式）。这增加了维护成本。如果修改某处逻辑（如早停策略、日志格式），需要同步修改 6 个地方（1 个 Trainer + 5 个自定义 fit）。
- **严重程度: 低** 自定义 fit 中的 `per_node` reshape 逻辑（`X.permute(0, 2, 1).reshape(B * N, H, 1)`）与 `TorchModel.forward()` 中的逻辑重复。不过这是设计上的权衡，因为自定义 fit 需要访问 Y 的中间表示。

---

## 3. 关键问题汇总

### 3.1 严重问题

无。

### 3.2 中等问题

| 编号 | 问题 | 影响 | 建议 |
|------|------|------|------|
| M1 | 自定义 fit 与 Trainer 间大量代码重复 | 维护成本高，修改需同步多处 | 提取公共训练循环组件（如 `TrainingLoop` mixin），让自定义 fit 只覆盖 loss 计算部分 |

### 3.3 轻微问题

| 编号 | 问题 | 影响 | 建议 |
|------|------|------|------|
| L1 | checkpoint 相对路径 | 不同 CWD 运行时可能不一致 | 考虑使用绝对路径或基于项目根目录的路径 |
| L2 | `model.predict()` 无进度条 | 大规模测试时不直观 | 在 `TorchModel.predict()` 中添加 tqdm |
| L3 | `calc_pcc` 零方差返回 0.0 | 可能掩盖数据质量问题 | 返回 `nan` 更语义化，或在日志中记录警告 |
| L4 | `_config_matches` 不检查 `test_steps` | 数据更新时可能不跳过旧结果 | 添加 `test_steps` 到匹配条件 |
| L5 | `run_ha_baseline.py` 硬编码 `output_dir` | 与其他脚本不一致 | 添加 `--output_dir` 参数 |

---

## 4. 总体评价

### 4.1 优点

1. **训练循环设计良好**: 标准 Trainer 包含完整的训练循环，有 epoch 级和 batch 级进度条，早停、学习率调度、梯度裁剪、checkpoint 管理一应俱全。

2. **模型保存与加载逻辑完整**: 在验证 loss 下降时保存最佳 checkpoint，训练结束后加载回模型，路径使用 `{model_name}_best.pt` 命名，不同模型不会冲突。

3. **评测流程严谨**: 评测分 full/anomaly/normal 三个子集，使用归一化-反归一化管道，支持跨城市传输场景。

4. **结果报告完备**: JSON 包含完整的元信息（模型、城市、时间戳、指标），Reporter 打印格式清晰。

5. **批量运行器健壮**: 支持跳过已完成实验、错误隔离、模型独立运行，编译 Markdown 摘要。

6. **HA baseline 特殊处理正确**: 通过 `isinstance` 检查，跳过归一化和训练，直接调用 `predict_with_timestamps`。

7. **指标计算防护到位**: 空数组、零方差、MAPE 阈值均有防护。

8. **自定义 fit 模型兼容性良好**: 所有 5 个自定义 fit 模型均正确实现了 `BaseModel.fit()` 接口，与 Evaluator 和 Batch Runner 完全兼容。

### 4.2 改进建议

1. **重构训练循环**（优先级: 高）: 将 Trainer 中的训练循环模板提取为可复用的组件，减少自定义 fit 中的代码重复。可以考虑:
   - 提取 `run_training_loop` 通用函数，接受 `train_step_fn` 和 `val_step_fn` 回调
   - 或使用 mixin 模式，让自定义模型只需覆盖 `compute_loss` 方法

2. **添加 `predict` 进度条**（优先级: 中）: 在 `TorchModel.predict()` 中添加 tqdm 进度条，提升大规模测试时的用户体验。

3. **统一 checkpoint 路径**（优先级: 中）: 建议将 checkpoint 路径与 output_dir 关联，或使用基于项目根目录的绝对路径。

4. **添加实验日志**（优先级: 中）: 考虑使用 `logging` 模块替代 `print`，支持同时输出到文件和控制台，便于事后追溯。

5. **添加 GPU 内存监控**（优先级: 低）: 在训练循环中可选地输出 GPU 内存使用情况，帮助调试 OOM 问题。

### 4.3 总体结论

STCross 项目的训练与评测基础设施**整体完备且设计良好**。训练循环、模型保存加载、评测流程、结果报告、批量实验、指标计算等核心链路均正常工作，没有发现严重缺陷。主要改进空间在于代码复用（减少自定义 fit 与标准 Trainer 之间的重复）和用户体验细节（predict 进度条、路径管理）。基础设施可以支撑完整的实验需求。