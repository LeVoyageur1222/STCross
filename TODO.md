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

### 1.3 PatchTST

> 文件：`model/patchtst.py` | 实现方式：适配开源
> 代码源：https://github.com/yuqinie98/PatchTST

- [ ] 提取核心模块，适配为 `TorchModel`，`input_mode="per_node"`
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models PatchTST --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

### 1.4 TimesNet

> 文件：`model/timesnet.py` | 实现方式：适配开源
> 代码源：https://github.com/thuml/TimesNet

- [ ] 提取 TimesBlock，适配为 `TorchModel`，`input_mode="per_node"`
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models TimesNet --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

---

## Phase 2：Group B Baselines（时空图方法）

> **目标**：完成 STGNN 方法，每个模型完成后立即运行 4 种实验设置。
> **依赖**：Phase 0 ✅（邻接矩阵已构建）
> **可与 Phase 1 并行推进**

### 2.0 邻接矩阵 ✅

- [x] `adj_NYC_corr_k10.npy` (820×820, 9608 edges)
- [x] `adj_CHI_corr_k10.npy` (585×585, 9696 edges)

### 2.1 STGCN

> 文件：`model/stgcn.py` | 实现方式：适配开源
> 代码源：https://github.com/VeritasYin/STGCN_IJCAI-18

- [ ] ChebConv + TemporalConv + ST-Conv Block
- [ ] 继承 `TorchModel`，`input_mode="all_nodes"`，构造函数接受 adj
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models STGCN --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

### 2.2 DCRNN

> 文件：`model/dcrnn.py` | 实现方式：适配开源
> 代码源：https://github.com/liyaguang/DCRNN

- [ ] DiffusionConvGRU + Seq2Seq
- [ ] 继承 `TorchModel`，`input_mode="all_nodes"`
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models DCRNN --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

### 2.3 GWNET

> 文件：`model/gwnet.py` | 实现方式：适配开源
> 代码源：https://github.com/nnzhan/Graph-WaveNet

- [ ] 扩散卷积 + WaveNet + 自适应邻接（可配置 adaptive=True/False）
- [ ] 继承 `TorchModel`，`input_mode="all_nodes"`
- [ ] 注册到 `model/registry.py`
- [ ] **用户运行**：`python experiments/run_all.py --models GWNET --settings all`
- [ ] 结果收集 → 更新 RESULTS.md

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

### 3.2 ConFormer (ST11) — 待代码开源

- [ ] 跟踪开源状态
- [ ] 若开源：适配 + 评测
- [ ] 若未开源：标记为无法复现

### 3.3 STEVE (ST13) — 待代码开源

- [ ] 跟踪开源状态
- [ ] 若开源：适配 + 评测
- [ ] 若未开源：标记为无法复现

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
