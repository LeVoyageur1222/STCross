# STCross 数据处理流水线与代码架构审查报告

> 审查日期：2026-06-25
> 审查范围：数据预处理流水线、模型架构、训练/评测框架、指标计算
> 审查文件：17 个核心文件

---

## 一、审查范围

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据处理 | `data_preprocess/dataset.py` | 数据划分、滑窗构建、子集索引 |
| 数据处理 | `data_preprocess/normalizer.py` | PerNodeStandardScaler 标准化器 |
| 数据处理 | `data_preprocess/torch_dataset.py` | PyTorch Dataset/DataLoader 封装 |
| 数据处理 | `data_preprocess/build_flow_data.py` | 原始 .pkl 流量数据提取 |
| 数据处理 | `data_preprocess/build_adjacency.py` | 相关性邻接矩阵构建 |
| 数据处理 | `data_preprocess/build_holiday_mask.py` | 节假日掩码构建 |
| 数据处理 | `data_preprocess/build_weather_mask.py` | 极端天气掩码构建 |
| 模型架构 | `model/base.py` | BaseModel 抽象基类 |
| 模型架构 | `model/torch_base.py` | TorchModel 基类（含 forward reshape） |
| 模型架构 | `model/registry.py` | 模型注册表 |
| 训练 | `train/trainer.py` | Trainer 和 TrainConfig |
| 评测 | `eval/config.py` | EvalConfig 评测配置 |
| 评测 | `eval/evaluator.py` | Evaluator 评测器 |
| 评测 | `eval/reporter.py` | Reporter 结果报告器 |
| 实验 | `experiments/run_all.py` | 批量运行器 |
| 实验 | `experiments/run_model.py` | 单模型运行器 |
| 指标 | `metrics/metrics.py` | 评估指标（MAE, RMSE, MAPE, PCC, R2） |

---

## 二、数据流完整路径分析

### 2.1 数据流全景图

```
原始 .pkl 数据
     │
     ▼
build_flow_data.py          ─── 提取 flow_{city}.npy ───┐
build_holiday_mask.py       ─── 提取 mask_{city}_holiday_*.npy ─┐
build_weather_mask.py       ─── 提取 mask_{city}_all_*.npy ─────┤
build_adjacency.py          ─── 提取 adj_{city}_corr_*.npy ─────┤
                                                              │
                     ┌────────────────────────────────────────┘
                     ▼
              Evaluator.run()
                     │
                     ├─ 1. load_flow(target_city)
                     ├─ 2. load_anomaly_mask(target_city)
                     ├─ 3. split_data(T, train_steps, val_steps)
                     │      → (0, train_steps), (train_steps, train_steps+val_steps),
                     │        (train_steps+val_steps, T)
                     │
                     ├─ 4. PerNodeStandardScaler.fit(target_flow[train_s:train_e])
                     │
                     ├─ 5. [同城] target_scaler.transform(target_flow)
                     │        → create_sliding_windows() → train_X, train_Y, val_X, val_Y
                     │
                     ├─ 5. [跨城] source_scaler.fit(source_flow[train_s:train_e])
                     │        → source_scaler.transform(source_flow)
                     │        → create_sliding_windows() → train_X, train_Y, val_X, val_Y
                     │
                     ├─ 6. model.fit(train_X, train_Y, val_X, val_Y)
                     │        → TorchModel.fit() → DataLoader → Trainer.train()
                     │
                     ├─ 7. target_scaler.transform(target_flow)
                     │        → create_sliding_windows() → test_X
                     ├─ 7. create_sliding_windows(target_flow_raw) → raw_test_Y, test_labels
                     │
                     ├─ 8. model.predict(test_X) → preds_norm
                     ├─ 9. target_scaler.inverse_transform(preds_norm) → preds_raw
                     │
                     └─ 10. metrics.calc_* (raw_test_Y vs preds_raw) → results
```

### 2.2 关键数据流验证

**同城场景（source == target）：**
- 标准化：target_scaler.fit(train 区间) → target_scaler.transform(全部) → target_scaler.inverse_transform(预测)
- 训练/验证/测试数据全部经过同一个 scaler 的 transform，inverse 也使用同一个 scaler
- 正确：训练数据标准化 → 模型训练 → 模型预测 → 逆标准化 → 原始尺度指标

**跨城场景（source != target）：**
- 训练标准化：source_scaler.fit(source train 区间) → source_scaler.transform(source 全部)
- 测试标准化：target_scaler.fit(target train 区间) → target_scaler.transform(target 全部)
- 逆标准化：target_scaler.inverse_transform(preds_norm)
- 模型在 source 标准化数据上训练，在 target 标准化数据上测试
- 这是跨城迁移的核心挑战：分布偏移（distribution shift）

---

## 三、各组件审查结论

### 3.1 `data_preprocess/dataset.py` -- 数据划分与滑窗构建

**评分：通过**

- `split_data()` 函数生成三个连续的、无重叠的时间区间：`[0, train)`, `[train, train+val)`, `[train+val, T)`
- 时间顺序严格保持，train/val/test 之间无重叠
- `create_sliding_windows()` 正确实现了滑窗逻辑：窗口 `[i, i+H)` 为输入，`[i+H, i+H+P)` 为输出
- `anomaly_min_ratio` 参数控制样本标签生成的阈值，设计合理
- `get_subset_indices()` 提供了 full/anomaly/normal 三种子集划分

**潜在关注点：**
- `create_sliding_windows()` 中的 `anomaly_min_ratio` 参数在实验中被固定为 0.5，但注释中给出了 0.0/0.5/1.0 三种语义。建议在论文中明确说明选择 0.5 的理由，并考虑做消融实验。

### 3.2 `data_preprocess/normalizer.py` -- PerNodeStandardScaler

**评分：通过，但有一个重要发现**

- 每个节点独立计算 mean 和 std，沿时间轴（axis=0）聚合
- 对近零 std 的节点进行了保护性处理（设为 1.0），避免除零错误
- `fit()` 方法返回 `self`，支持链式调用

**关键正确性检查：**

在 `evaluator.py` 中，`fit()` 的调用方式是：
```python
target_scaler = PerNodeStandardScaler().fit(
    target_flow[train_s:train_e].astype(np.float64))
```

这确保了 scaler 仅在训练集上拟合，然后 `transform()` 应用于全部数据（包括验证集和测试集）。这是正确的做法，避免了数据泄露。

**潜在问题（低严重度）：**
- 对近零 std 节点统一设为 1.0 的 clip 处理，在 `inverse_transform` 时会导致这些节点永远输出 0（因为归一化后值为 0，乘以 1.0 + mean 后只是 mean 值）。这是合理的设计选择，但论文中应提及。

### 3.3 `data_preprocess/torch_dataset.py` -- PyTorch Dataset/DataLoader

**评分：通过，有一个值得注意的细节**

- `TrafficFlowDataset` 将 numpy 数组转换为 float32 tensor，正确
- `build_dataloader()` 中 `num_workers=0` 避免了多进程数据加载的潜在问题，但可能成为性能瓶颈
- `pin_memory=torch.cuda.is_available()` 合理

**关键正确性检查：**

在 `torch_base.py` 的 `fit()` 方法中：
```python
train_loader = build_dataloader(train_X, train_Y, ..., shuffle=True)
val_loader = build_dataloader(val_X, val_Y, ..., shuffle=False)
```

- 训练集 `shuffle=True`：正确，打乱训练样本顺序有助于 SGD 收敛
- 验证集 `shuffle=False`：正确，评估时不需要打乱
- 预测时 `predict()` 方法不通过 DataLoader，而是直接切片遍历，保证顺序

**注意：** `shuffle=True` 在时间序列预测中是一个微妙的设计选择。虽然训练样本之间存在时间重叠（因为滑窗步长为 1），但 shuffle 并不会导致数据泄露，因为每个样本的输入-输出对是完备的，且验证集和测试集与训练集在时间上不重叠。shuffle 主要影响的是批次内的梯度估计，不影响模型对时间依赖性的学习能力（因为时序信息已经在单个样本的 H 步窗口内编码）。

### 3.4 `data_preprocess/build_flow_data.py` -- 流量数据构建

**评分：通过**

- 从 .pkl 文件正确提取 `Node['TrafficNode']` 作为流量数据
- 支持按日期范围裁剪，通过时间步长计算索引
- 输出 .npy 文件，格式统一

**注意：** 硬编码的时间范围裁剪（2014-10-01 到 2017-09-30）与 EvalConfig 中的配置一致。如果修改配置，需要同步更新此脚本并重新运行。

### 3.5 `data_preprocess/build_adjacency.py` -- 邻接矩阵构建

**评分：通过，设计合理**

- 使用训练集流量数据计算 Pearson 相关系数矩阵
- 正确处理了死节点（std=0）
- 使用 Top-K 稀疏化（k=10）和对称化（max(A, A^T)）
- 输出连通分量统计，便于验证图结构质量

**关键正确性检查：** 邻接矩阵仅从训练集构造（`flow[train_s:train_e]`），未使用验证集或测试集信息。这是正确的。

### 3.6 `data_preprocess/build_holiday_mask.py` -- 节假日掩码构建

**评分：通过**

- 覆盖了美国联邦假日、选举日、城市级额外假日
- 正确处理了周末调休（周六调至周五、周日调至周一）
- 与流量数据使用相同的时间范围（2014-10-01 到 2017-09-30）和频率（60min）

**注意：** 掩码构建不依赖任何流量数据，仅基于日历规则，因此不存在数据泄露。

### 3.7 `data_preprocess/build_weather_mask.py` -- 天气掩码构建

**评分：通过**

- 定义了 7 种极端天气类型（暴雨、暴雪、风暴、极端高温、极端低温、低能见度、极端紫外线）
- 时间片重叠判断逻辑正确（valid_time < slice_end AND expire_time > slice_start）
- 掩码构建不依赖流量数据，仅基于外部天气数据

**注意：** 极端紫外线（extreme_uv）和极端高温/低温（extreme_heat/extreme_cold）类型已定义但未在实际使用中启用（`extreme_weather_types = ['heavy_rain', 'heavy_snow', 'storm', 'low_visibility']`）。这是合理的设计选择，但建议在论文中说明。

### 3.8 `model/base.py` -- BaseModel 抽象基类

**评分：通过**

- 清晰定义了 `fit()`, `predict()`, `name` 三个接口
- 使用 numpy 数组作为接口约定，与 PyTorch 实现解耦
- 简单但有效的抽象

### 3.9 `model/torch_base.py` -- TorchModel 基类

**评分：通过，设计精良**

- 同时继承 `BaseModel` 和 `nn.Module`，桥接了 numpy 接口和 PyTorch 模块
- `input_mode` 机制（"per_node" vs "all_nodes"）优雅地处理了 channel-independent 和 graph-based 模型的差异
- `forward()` 中的 reshape 逻辑正确：
  - `per_node`: `(B,H,N) → (B*N,H,1) → _net_forward → (B*N,P,1) → (B,P,N)`
  - `all_nodes`: `(B,H,N) → _net_forward → (B,P,N)`
- `predict()` 方法正确处理了 eval 模式、设备转移和大批量推理

**潜在关注点：**
- `predict()` 中使用的 `batch_size = self.train_config.batch_size * 2` 是硬编码的 2x 放大。在 GPU 内存充足的情况下合理，但在内存受限时可能导致 OOM。建议改为可配置参数。

### 3.10 `model/registry.py` -- 模型注册表

**评分：通过**

- 使用装饰器模式实现模型注册，扩展性好
- 15 个模型统一通过 `(EvalConfig, TrainConfig)` 参数构建
- 模型工厂函数中延迟导入（`from model.xxx import ...`），避免不必要的模块加载

**循环导入检查：** `registry.py` 在顶部导入了 `HA` 模型和两个配置类，但各模型工厂函数内部使用延迟导入。未发现循环导入风险。`model/__init__.py` 仅导入 `BaseModel` 和 `HAModel`，不导入 `registry.py`，避免了循环依赖。

### 3.11 `train/trainer.py` -- Trainer 和 TrainConfig

**评分：通过，功能完善**

- `TrainConfig` 使用 dataclass，集中管理所有训练超参数
- 支持早停（early stopping）、学习率调度（ReduceLROnPlateau）、梯度裁剪
- 最佳模型通过 `model.state_dict()` 保存和加载
- 设备管理（auto/cuda/mps/cpu）灵活

**潜在问题（低严重度）：**
- `Trainer.train()` 中 `torch.load(..., weights_only=True)` 在 PyTorch >= 2.0 中是正确的默认行为，但 `weights_only=True` 可能在某些自定义模型类上出错。如果未来模型包含非张量状态，需要调整。
- 没有处理训练过程中梯度爆炸/NaN 的检测机制。如果训练不稳定，可能导致静默失败。

### 3.12 `eval/config.py` -- EvalConfig 评测配置

**评分：通过，设计良好**

- 使用 dataclass 集中管理评测配置
- 路径生成方法（`flow_path()`, `holiday_mask_path()`, `weather_mask_path()`, `adj_path()`）确保命名一致性
- 时间分割参数（train_steps=15336, val_steps=5160）与注释中的月份对照清晰

**注意：** 15336 steps = 639 天（每步 60 分钟），从 2014-10-01 到 2016-06-30，确实是 639 天。验证正确。

### 3.13 `eval/evaluator.py` -- Evaluator 评测器

**评分：通过，但有以下关键发现**

这是整个数据流的核心协调器。以下逐步骤分析其正确性：

**步骤 1-3：加载数据、掩码、时间划分**
- 正确。时间划分使用 `split_data()`，保证 train/val/test 不重叠。

**步骤 4：Scaler 拟合**
- `target_scaler.fit(target_flow[train_s:train_e])`：仅在训练集上拟合，正确。
- 同城场景：`target_scaler.transform(target_flow)` 应用于全部数据，正确。

**步骤 5：跨城场景特殊处理**
- 跨城时，训练数据使用 `source_scaler` 标准化（在 source 训练集上拟合），测试数据使用 `target_scaler` 标准化（在 target 训练集上拟合）。
- 这里的关键设计是：模型在 source 分布上训练，在 target 分布上测试，这是跨城迁移的固有挑战。
- **重要：** 代码注释明确说明 "The model sees a distribution shift, which is exactly the cross-city challenge we measure"。这个设计是正确的。

**步骤 6：模型训练**
- `model.fit(train_X, train_Y, val_X, val_Y)` 传递训练和验证集。

**步骤 7：测试集构建**
- `raw_test_Y` 从原始（未标准化）流量构建，用于最终指标计算
- `test_X` 从标准化流量构建，用于模型预测
- 这样避免了 scaler 的反向传播影响指标计算

**步骤 8-9：预测与逆标准化**
- `preds_norm = model.predict(test_X)`：模型在标准化数据上预测
- `preds_raw = target_scaler.inverse_transform(preds_norm)`：逆标准化到原始尺度
- 对于同城场景：训练和测试使用同一个 scaler，逆标准化无效（因为模型在 target 标准化数据上训练，在 target 标准化数据上测试，预测值本身就是 target 标准化尺度，逆标准化恢复原始尺度）
- 对于跨城场景：模型在 source 标准化数据上训练，在 target 标准化数据上测试。预测值在 target 标准化尺度上，逆标准化是正确的。

**步骤 10：指标计算**
- 在原始尺度上计算 MAE, RMSE, MAPE, PCC, R2，正确。

**潜在关注点（中严重度）：**

跨城场景中存在一个微妙的设计问题。模型在 source 标准化数据上训练，在 target 标准化数据上测试。由于 source 和 target 的节点数量 N 可能不同，使用 `source_scaler` 标准化后训练的模型，其输出空间的统计特性与 `target_scaler` 的标准化空间可能不同。这是跨城迁移研究的固有挑战，但代码中应该明确这一点。

具体来说：如果 source 和 target 的 N 不同，`source_scaler` 和 `target_scaler` 的 mean/std 向量维度不同，因此模型在 source 标准化空间中学习到的映射，在 target 标准化空间中可能不适用。这是跨城迁移的核心研究问题，代码设计合理。

### 3.14 `eval/reporter.py` -- Reporter 结果报告器

**评分：通过**

- 格式化打印结果表格，清晰展示各子集指标
- JSON 保存结果，便于后续分析
- 时间戳命名避免文件名冲突

### 3.15 `experiments/run_all.py` -- 批量运行器

**评分：通过**

- 支持多模型 x 多城市设定的批量实验
- 包含结果缓存机制（检查已有结果文件，跳过已完成的实验）
- 异常处理：单个实验失败不影响其他实验
- 生成 summary.md 汇总表

**潜在问题（低严重度）：**
- 结果缓存机制仅检查最近一个匹配文件，如果结果过时（例如代码更新后重新运行），需要手动删除旧结果文件。建议添加 `--force` 或 `--no-cache` 参数。

### 3.16 `experiments/run_model.py` -- 单模型运行器

**评分：通过**

- 命令行接口清晰，参数可配置
- 与 `run_all.py` 共享 `EvalConfig`, `TrainConfig`, `MODEL_REGISTRY`

### 3.17 `metrics/metrics.py` -- 评估指标

**评分：通过**

- 实现了 5 个标准指标：MAE, RMSE, MAPE, PCC, R2
- MAPE 使用阈值过滤（忽略接近零的真值），避免除零
- PCC 使用 `scipy.stats.pearsonr`
- 各指标均有空值和退化情况的处理

---

## 四、数据泄露风险评估

### 4.1 时间划分泄露检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| train/val/test 时间无重叠 | 通过 | `split_data()` 返回连续不重叠区间 |
| 滑窗不跨越划分边界 | 通过 | `create_sliding_windows()` 在指定区间内滑动 |
| 验证集信息未用于训练 | 通过 | 验证集仅用于早停和 LR 调度，不参与梯度更新 |
| 测试集信息未用于任何调优 | 通过 | 测试集仅在最终评估时使用 |

### 4.2 标准化泄露检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Scaler 仅在训练集上 fit | 通过 | `target_scaler.fit(flow[train_s:train_e])` |
| 验证/测试集使用同一 scaler transform | 通过 | 正确，不包含验证/测试集统计信息 |
| 逆标准化使用正确的 scaler | 通过 | 始终使用 `target_scaler.inverse_transform()` |

### 4.3 邻接矩阵泄露检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 邻接矩阵仅从训练集构造 | 通过 | `build_correlation_adj(flow_train)` 仅接收训练集 |
| 邻接矩阵不包含验证/测试集信息 | 通过 | 验证/测试集流量不参与相关计算 |

### 4.4 掩码泄露检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 节假日掩码基于日历规则 | 通过 | 不依赖任何流量数据 |
| 天气掩码基于外部气象数据 | 通过 | 不依赖任何流量数据 |

### 4.5 综合评估

**未发现数据泄露风险。** 代码在时间划分、标准化、邻接矩阵构建等关键环节均遵循了正确的时间序列数据划分原则。

---

## 五、架构设计评价

### 5.1 模块分层

```
experiments/          ← 实验编排层（run_all.py, run_model.py, run_ha_baseline.py）
    │
    ├── eval/         ← 评测层（config, evaluator, reporter）
    │
    ├── model/        ← 模型层（base, torch_base, registry, 15个具体模型）
    │
    ├── train/        ← 训练层（TrainConfig, Trainer）
    │
    ├── metrics/      ← 指标层（MAE, RMSE, MAPE, PCC, R2）
    │
    └── data_preprocess/  ← 数据层（dataset, normalizer, torch_dataset, build_*）
```

**评价：分层清晰，依赖方向合理。** 上层依赖下层，下层不依赖上层。没有跨层反向依赖。

### 5.2 依赖关系图

```
experiments
  ├── eval.{EvalConfig, Evaluator, Reporter}
  ├── model.registry.MODEL_REGISTRY
  └── train.TrainConfig

eval.Evaluator
  ├── data_preprocess.{dataset, normalizer}
  ├── metrics.metrics
  ├── model.base.BaseModel
  └── eval.config.EvalConfig

model.torch_base.TorchModel
  ├── model.base.BaseModel
  ├── train.trainer.{Trainer, TrainConfig}
  └── data_preprocess.torch_dataset.build_dataloader

train.Trainer
  └── (独立，仅依赖 PyTorch 标准库)
```

**评价：依赖关系清晰，无循环依赖。** `model.registry` 导入 `model.ha` 和 `eval.config`/`train.trainer`，但各模型工厂使用延迟导入避免重量级模块的提前加载。`model/__init__.py` 仅导入 `BaseModel` 和 `HAModel`，不导入 `registry`，避免了 `model/__init__.py → registry → model/*` 的潜在循环。

### 5.3 配置管理

- **TrainConfig**：管理训练超参数（epochs, batch_size, lr, early_stop, lr_schedule, grad_clip 等）
- **EvalConfig**：管理评测配置（数据路径、时间划分、历史/预测长度、子集定义等）
- 两个配置类相互独立，职责明确
- 配置通过参数传递，不在模块内部硬编码

**评价：配置集中管理，设计良好。**

### 5.4 扩展性

- 添加新模型：创建模型类继承 `TorchModel`，实现 `_net_forward()` 和 `name`，在 `registry.py` 中注册
- 添加新指标：在 `metrics/metrics.py` 中添加函数，在 `evaluator.py` 中调用
- 添加新城市：扩展 `EXPERIMENT_MATRIX`，准备对应的数据文件
- 添加新 mask：参考 `build_holiday_mask.py` 创建新脚本，在 `dataset.py` 的 `load_anomaly_mask()` 中整合

**评价：扩展性良好。**

---

## 六、发现的问题及严重程度

### 严重（Critical）：无

### 中等（Medium）：1 个

**M1. 跨城场景中 scaler 不匹配的隐式假设**

- **位置：** `eval/evaluator.py` 第 62-74 行
- **描述：** 跨城迁移时，模型在 source 标准化数据上训练，在 target 标准化数据上测试。由于 source 和 target 的节点数和流量分布不同，source_scaler 和 target_scaler 产生不同的标准化空间。虽然代码注释承认了分布偏移，但模型在训练时学到的是 source 标准化空间中的映射，测试时却应用于 target 标准化空间。这可能导致性能下降，但这也是跨城迁移研究的核心挑战。
- **建议：** 在论文中明确讨论分布偏移的影响，并考虑添加一个 baseline：在 target 标准化数据上直接训练（即同城场景）作为性能上界。

### 低（Low）：5 个

**L1. predict() 中 batch_size 硬编码 2x 放大**

- **位置：** `model/torch_base.py` 第 75 行
- **描述：** `batch_size = self.train_config.batch_size * 2` 硬编码。在 GPU 内存受限时可能导致 OOM。
- **建议：** 添加 `predict_batch_multiplier` 到 `TrainConfig`，或在 `predict()` 中动态调整。

**L2. 训练中无 NaN/Inf 检测**

- **位置：** `train/trainer.py` `_run_epoch()` 方法
- **描述：** 如果训练过程中出现梯度爆炸导致 NaN loss，Trainer 不会检测也不会终止，而是继续训练直到早停触发。
- **建议：** 在 `_run_epoch()` 中添加 `if torch.isnan(loss) or torch.isinf(loss): raise RuntimeError(...)` 检测。

**L3. 结果缓存机制无强制重跑选项**

- **位置：** `experiments/run_all.py` 第 45-56 行
- **描述：** 如果代码更新后需要重新运行实验，必须手动删除旧结果文件。
- **建议：** 添加 `--force` 或 `--no-cache` 命令行参数。

**L4. build_flow_data.py 硬编码日期范围**

- **位置：** `data_preprocess/build_flow_data.py` 第 93-105 行
- **描述：** `__main__` 中硬编码了 `start_date="2014-10-01"` 和 `end_date="2017-09-30"`。如果修改 `EvalConfig` 中的日期范围，需要同步更新此脚本。
- **建议：** 将日期范围提取为命令行参数或配置文件。

**L5. 极端天气类型定义但未使用**

- **位置：** `data_preprocess/build_weather_mask.py` 第 9-45 行
- **描述：** `extreme_heat`, `extreme_cold`, `extreme_uv` 已定义但未在 `__main__` 中启用。如果这是有意为之，应在论文中说明原因。
- **建议：** 添加注释说明未启用的原因，或在论文中讨论。

---

## 七、总体评价

### 7.1 优点

1. **数据流设计正确**：原始数据 → 标准化 → 滑窗 → DataLoader → 模型 → 预测 → 逆标准化 → 指标计算，整个流程的每一步都经过仔细设计，未发现数据泄露。

2. **时间序列划分严格**：train/val/test 在时间上严格连续且无重叠，标准化器仅在训练集上拟合，邻接矩阵仅从训练集构造。

3. **架构分层清晰**：数据层、模型层、训练层、评测层、实验层各司其职，依赖方向自上而下，无循环依赖。

4. **配置集中管理**：`TrainConfig` 和 `EvalConfig` 使用 dataclass 集中管理所有超参数。

5. **模型注册机制**：`MODEL_REGISTRY` 装饰器模式使添加新模型非常简单，且支持延迟导入。

6. **跨城迁移设计**：同城和跨城场景共用同一套代码框架，通过 `source_city` 和 `target_city` 配置区分，设计巧妙。

7. **异常处理**：批量运行器中单个实验失败不影响其他实验，且有详细的 traceback 输出。

8. **指标计算严谨**：MAPE 有阈值过滤，PCC 有退化处理，R2 有除零保护。

### 7.2 改进建议

1. **添加数据验证层**：在数据加载后添加基本的完整性检查（如负值检测、缺失值检测、维度一致性验证）。

2. **添加实验追踪**：考虑集成 MLflow 或简单的实验日志，记录每次运行的超参数、指标和模型路径。

3. **添加单元测试**：关键模块（如 `split_data`, `create_sliding_windows`, `PerNodeStandardScaler`）应添加单元测试，确保数据划分和标准化的正确性。

4. **考虑添加数据增强**：对于跨城迁移场景，考虑在 source 训练时添加噪声或域适应技术。

5. **文档化数据版本**：建议在 `EvalConfig` 或数据目录中添加数据版本号，确保实验结果可复现。

### 7.3 总结

STCross 项目的数据处理流水线和代码架构设计扎实，核心数据流正确无误，未发现数据泄露风险。代码分层清晰，模块间依赖关系合理，扩展性良好。发现的 6 个问题均为中低严重度，不影响实验结果的正确性。建议在论文中明确讨论跨城迁移中的分布偏移问题，并考虑添加上述改进建议中的功能。

---

*审查人：AI Code Reviewer*
*审查日期：2026-06-25*