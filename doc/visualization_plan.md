# 可视化实验设计：V1-V3

> 日期：2026-07-06
> 关联代码：`visualization/viz_v1_*.py`、`viz_v2_*.py`、`viz_v3_*.py`
> 前置：`experiments/retrain_for_viz.py`（重训 viz 用 checkpoint）
> 输出目录：`results_viz/`

---

## 0. 总体原则

1. **只加载、不训练**：V1/V2/V3 脚本内不训练任何模型，全部加载 `checkpoints_viz/` 下已训好的 checkpoint 做推理。唯一训练是前置的 `retrain_for_viz.py`（STCross α=0.3 + DAN，存独立目录 `checkpoints_viz/`，不覆盖主结果 `checkpoints/`）。
2. **checkpoint 来源**：现有 `checkpoints/STCross_NYC_best.pt` 是主结果 α=0.01 那次；网格/消融的 α=0.3 最优 checkpoint 未保存。为贴合论文最优配置（α=0.3/K=8/γ=0.1/β=0.01）并让 V2 容量正则更强、不变性更干净，统一重训一个 α=0.3 checkpoint 供 V1-V3 使用。
3. **复用既有接口**：不重新实现数据流，全部复用 `eval/evaluator.py` 的 `_compute_nbl_full`/`_slice_nbl`、`data_preprocess/dataset.py` 的加载/窗口函数、`data_preprocess/normalizer.py`、`model/stcross.py` 的 `encode`/`predict_with_decomposition`、`model/torch_base.py` 的 `load_checkpoint`/`predict`、`model/ha.py` 的 `predict_with_timestamps`。
4. **画图规范**：与 `visualization/plot_grid_sensitivity.py`、`plot_ablation.py` 一致——matplotlib Agg、Times New Roman、300 dpi、serif 数学字体、2 位小数。
5. **V5（反事实测试）已删除**：原 V5 需新增 `predict_with_swapped_zs()` 接口，本轮不做；如后续需要再单独开列。

---

## 1. 前置：重训 viz checkpoint

**脚本**：`experiments/retrain_for_viz.py`
**作用**：用 `Evaluator.train(model, "NYC")` 只训 NYC source（不跑全 4 设置，省时），保存到 `checkpoints_viz/`：
- `STCross_NYC_best.pt`（α=0.3, K=8, γ=0.1, β=0.01）
- `DAN_NYC_best.pt`（V3 基线，默认）

**运行**：
```bash
python experiments/retrain_for_viz.py
```
预计 20-40 min（STCross + DAN 各一次 NYC 训练，early_stop_patience=10）。

---

## 2. V1：偏差原型 pattern（DIP 可解释性）

### 动机
展示 APB（Anomaly Proto Bank）学到的 K=8 个可迁移异常偏差形状，证明 DIP 在归一化偏差空间学到了有语义、差异化的异常原型（而非退化成相同向量）。这是 DIP"可迁移异常类型表征"的直接可视化。

### 方法
1. 加载 STCross（α=0.3, `checkpoints_viz/STCross_NYC_best.pt`）。
2. 直接取权重 `model.apb.pattern.detach().numpy()` → shape (8, 12)，即 8 个原型的 12 步 Δ̂_type 曲线（归一化偏差空间，城市无关）。
3. 在 CHI 测试集异常样本（~398 个）上 `model.encode()` 取 z_a，过 APB 距离（手动 `‖z‖²+‖e‖²−2z·eᵀ`，规避 MPS backward bug）取 argmin 最近原型，统计每个原型的被选频次。

### 输出
- `results_viz/v1_prototype_patterns.png`：2×4 子图，每子图一个原型的 12 步曲线（共享 y 轴范围便于对比），零基线虚线。
- `results_viz/v1_prototype_usage.png`：柱状图，x=8 个原型，y=被选占比（CHI 异常样本）。

### 预期结果
- 原型曲线**形状差异化**：例如"全天正向偏差"（节假日出行增多）、"早高峰负向偏差"（节假日通勤消失）、"短时负脉冲"（极端天气抑制）、"近似零波动"（正常模式）。
- 使用分布**非均匀**：少数原型主导异常样本（如 2-3 个原型覆盖 >70% 异常），说明原型有分工而非均匀退化。
- 若在归一化空间学习，原型形状应城市无关——可附带在 NYC 异常上重复统计对比（可选扩展）。

### 证伪条件（诚实记录）
- 若所有原型曲线形状雷同（方差极小）→ APB 退化，DIP 的"差异化异常类型"claim 不成立，需在文档记录并讨论（可能 K 过大或 L_proto 过弱）。
- 若使用分布极度集中（1 个原型占 >95%）→ 原型库未充分利用，APB 退化为单原型，需记录。

---

## 3. V2：Z_a/Z_s 城市不变性（DDAC 解耦验证）

### 动机
验证 DDAC 的核心 claim：Z_a 城市不变（可迁移），Z_s 城市特定，raw Δ 作为上界（城市可分）。这是"偏差空间解耦"的定量 + 定性双重验证。设计文档 §12 的三级保证中，Z_a 幅度不变性由方差归一化（确定性）保证，不依赖 α；Z_a 信息量受限由 L_cap（α）保证。

### 方法
1. 加载 STCross（α=0.3）。构造 NYC（in-distribution）+ CHI（cross-city）两组测试数据：各自 `load_flow`/`load_anomaly_mask`/`split_data`/`PerNodeStandardScaler`(fit on 各自 train)/`_compute_nbl_full`/滑动窗口。各采 ~400 异常 + ~400 正常样本（rng=42 可复现）。
2. `model.encode(X, b_input, b_future)` → z_a (B*N, 64), z_s (B*N, 64)。raw_delta = (X − b_input).reshape(−1, H) → (B*N, 12)（行序 b*N+n 与 encode 的 per-node reshape 一致）。
3. **域分类器（定量）**：对 raw Δ / Z_a / Z_s 各训一个 `LogisticRegression` 预测城市标签（NYC=0/CHI=1），80/20 stratified 切分报 accuracy。
4. **t-SNE（定性）**：z_a、z_s 各 ~2000 行/city 做 `TSNE(n_components=2, init='pca', perplexity=30)`，按城市着色。

### 输出
- `results_viz/v2_domain_accuracy.png`：柱状图，x=raw Δ / Z_a / Z_s，y=accuracy，参考线 50%（随机）与 90%（上界）。
- `results_viz/v2_tsne_za.png`：z_a 的 t-SNE，NYC/CHI 着色。
- `results_viz/v2_tsne_zs.png`：z_s 的 t-SNE。

### 预期结果
- raw Δ 城市分类 acc **>90%**（上界：原始偏差含城市幅度/模式，可分）。
- Z_a acc **≈50%**（城市不变：DIP 成功，归一化+容量正则挤出城市信息）。
- Z_s acc **>70%**（城市特定：DDAC 分工，CAD 拿到城市表征）。
- t-SNE：Z_a 两城市点**高度重叠**；Z_s 两城市点**分离**。

### 证伪条件（诚实记录）
- 若 Z_a acc 显著 >50%（如 >65%）→ 城市不变性不成立，需记录并讨论：可能 α（容量正则）过弱、或 std-mask 未实现导致近零偏差样本引入城市噪声。对应 `doc/ablation_review_synthesis.md` 前沿教授的"t-SNE 不足以验证城市不变性，必须补 domain classifier"——本设计已补 domain classifier 定量指标。
- 若 Z_s acc <60% → 城市特定分支未学到城市信息，DDAC 分工 claim 受质疑。
- 三者关系应满足 raw Δ > Z_s > Z_a，否则解耦方向异常，需记录。

---

## 4. V3：预测曲线 Case Study（STCross 分解）

### 动机
在"全异常/异常占比尽量高"的区间展示 STCross 的分解预测（Ŷ = NBL + Δ̂_type + Δ̂_adj）优于 baseline，直观说明三组件如何协作。节假日和极端天气各选一个区间，且区间内 baseline 表现差、STCross 表现好，效果最明显。

### 候选区间（CHI 测试期 2017-02-01~09-29，已探查）
- **节假日**（mask 连续段 ≥6h）：Lincoln's Birthday(2/12-13, 48h)、Presidents' Day(2/20)、Memorial Day(5/29)、Independence Day(7/04)、Labor Day(9/04)。
- **天气**（mask 连续段 ≥6h）：2017-02-06(19h)、2017-07-21(12h) 及若干短时极端天气段。脚本自动扫描 `mask_CHI_holiday` 和 `mask_CHI_all`（weather）的连续 run（≥6h）。

### 方法
1. 加载 STCross(α=0.3, `checkpoints_viz/`) + HA（CHI 上重算 lookup，无需 checkpoint，可 `--no-ha` 关闭）+ 一个或多个学习型 baseline（命令行 `--baselines NAME [NAME...]` 指定，**默认为三类最优：LSTM+ModernTCN+DAN**）。
   - Group A（时序）最优 **LSTM**（`checkpoints/`，复用主结果 checkpoint）
   - Group C（分布鲁棒）最优 **ModernTCN**（`checkpoints_viz/`，retrain 重训）
   - Group D（极端事件）最优 **DAN**（`checkpoints_viz/`，retrain 重训）
   - 最优依据：`results/RESULTS.md` NYC→CHI full 子集，按 R²+PCC 平衡口径（与项目自身分析一致）。ProtoTS/STSSDL 虽 anomaly MAE 更低但 R² 弱/不稳定，未选；可经 `--baselines` 手动加入对比。
2. 对每个候选 run [r_start, r_end]：
   - 计算"滚动 1-step-ahead"样本范围 i ∈ [r_start−H, r_end−H]（夹到测试集有效样本），step-0 预测覆盖 [r_start, r_end]。
   - STCross `predict_with_decomposition` + 各 baseline `predict` + HA `predict_with_timestamps`，反标准化到原始尺度（HA 已是原始尺度）。
   - 算各模型 step-0 MAE（active 节点 mean，active = CHI train 期均值 >0.5）。
3. **选区间**：节假日、天气各选 `(最强 baseline MAE − STCross MAE)` 最大的（STCross 即便对最强 baseline 也赢得最多的区间）。打印全候选 MAE 表（含每个 baseline + margin 列）供人工核查。
4. **画图**（每区间一张，共 2 张）：
   - 上 panel：滚动 1-step-ahead 曲线——true vs STCross vs HA vs 各 baseline，对 active 节点 mean 聚合；标题含区间日期 + 各模型 MAE。
   - 下 panel：STCross 分解（代表样本，取全异常样本范围的中间）——NBL baseline + Δ̂_type + Δ̂_adj 堆叠面积 + STCross pred 虚线 + true 实线。

### Baseline 配置（代码内 + 命令行）
脚本顶部 `BASELINE_CONFIG` 字典维护每个 baseline 的 checkpoint 目录：
```python
BASELINE_CONFIG = {
    "DAN":      "checkpoints_viz",   # retrain_for_viz.py 重训
    "EVL":      "checkpoints",       # 主结果 checkpoint
    "LSTM":     "checkpoints",
    "DLinear":  "checkpoints",
    "PatchTST": "checkpoints",
}
```
- **命令行选 baseline**：`--baselines NAME [NAME...]`（默认三类最优 LSTM+ModernTCN+DAN，可任意子集）
- **命令行覆盖路径**：`--ckpt-dir NAME=PATH`（可重复，如 `--ckpt-dir DAN=checkpoints`）
- **代码内加 baseline**：在 `model/registry.py` 注册后，往 `BASELINE_CONFIG` 加一行 `"NAME": "ckpt_dir"`
- HA 始终作为 NBL 参考线（非参数化，无 checkpoint）；`--no-ha` 关闭

### 输出
- `results_viz/v3_case_study_holiday.png`
- `results_viz/v3_case_study_weather.png`
- 命令行打印全候选 MAE 表（每个 baseline + HA + margin 列），供人工核查区间选择

### 预期结果
- 选中区间上 **STCross MAE < 基线 ≤ HA**（异常子集聚合已示 STCross anomaly MAE 0.6292 < DAN 0.7577；HA 0.625 接近 STCross）。
- 分解图：NBL 提供正常基线形状，Δ̂_type 提供可迁移异常形状修正（节假日区间应呈典型节假日偏差形状，与 V1 原型对应），Δ̂_adj 提供城市幅度调整，三者相加≈STCross pred，贴近 true。
- 滚动曲线：STCross 在异常时段跟踪 true 的能力优于基线（尤其节假日早高峰下降、天气短时抑制处）。

### 证伪条件（诚实记录）
- 若所有候选区间 STCross MAE 都不优于最强 baseline → 案例研究选不出"STCross 赢"的区间，说明在硬异常上 STCross 无优势，需记录（与主结果 anomaly 子集结论矛盾时优先信主结果聚合数据）。
- 若分解图中 Δ̂_type 与 Δ̂_adj 量级远小于 NBL → 分解组件贡献微弱，"偏差空间分解"价值受限，需记录。
- 若 STCross pred ≠ baseline + Δ̂_type + Δ̂_adj → 实现错误（应相等），需排查。

### 实现注意
- `flow` 是 2D (T, N)，构造 (L, H, N) 窗口用 `np.stack([arr[i:i+H] for i in idx])`，不用 fancy indexing。
- HA 输出已是原始尺度，**不**反标准化；STCross/各 baseline 输出是归一化空间，需 `inverse_transform`。反标准化线性，故 `inv(baseline)+inv(delta_type)+inv(delta_adj) = inv(y_pred)`，分解在原始尺度仍成立。
- active 节点聚合用 mean（非 sum），避免量纲过大。
- 多 baseline 时上 panel 线条较多，每个 baseline 用 `BASELINE_COLORS` 里固定颜色区分；超过 5 个用默认色循环。

---

## 5. 运行流程（详细）

### 5.0 环境前置

- **Python ≥ 3.9**（项目实测 3.10/3.12 均可）。
- 依赖：`torch numpy pandas scipy tqdm einops scikit-learn`。V2 用 `sklearn.manifold.TSNE` + `sklearn.linear_model.LogisticRegression`（已验证 sklearn 1.7.2 可用）。安装：`pip install -r requirements.txt && pip install scikit-learn`。
- **必须从项目根目录运行**：所有路径（`checkpoints/`、`checkpoints_viz/`、`data/processed_data/`、`results_viz/`）均相对项目根。所有脚本已内置 `sys.path.insert(0, 项目根)`，**无需设 `PYTHONPATH`**，直接 `python xxx.py` 即可。
- 数据：`data/processed_data/` 下 NYC/CHI 的 flow/holiday/weather mask 必须已生成（项目已就绪）。

### 5.1 步骤（项目根目录执行）

| 步骤 | 命令 | 产物 | 耗时 | 依赖 |
|---|---|---|---|---|
| 0 前置 | `python experiments/retrain_for_viz.py` | `checkpoints_viz/{STCross,ModernTCN,DAN}_NYC_best.pt`（LSTM 复用 `checkpoints/`） | 30-60min | 仅一次 |
| 1 V1 | `python visualization/viz_v1_prototype_patterns.py` | `results_viz/v1_prototype_patterns.png` + `v1_prototype_usage.png` | 秒级 | 步骤 0 |
| 2 V2 | `python visualization/viz_v2_city_invariance.py` | `results_viz/v2_domain_accuracy.png` + `v2_tsne_za.png` + `v2_tsne_zs.png` | 分钟级 | 步骤 0 |
| 3 V3 | `python visualization/viz_v3_prediction_case_study.py` | `results_viz/v3_case_study_{holiday,weather}.png` + MAE 表 | 分钟级 | 步骤 0 |

> 步骤 1-3 互相独立，只依赖步骤 0（retrain 生成的 checkpoint），可任意顺序跑。

### 5.2 命令示例

```bash
# 0. 重训 viz checkpoint（α=0.3 最优 STCross + DAN，仅 NYC source，不覆盖主结果）
python experiments/retrain_for_viz.py

# 1. V1 偏差原型 pattern
python visualization/viz_v1_prototype_patterns.py

# 2. V2 Z_a/Z_s 城市不变性
python visualization/viz_v2_city_invariance.py

# 3. V3 预测曲线 case study（默认三类最优 baseline：LSTM+ModernTCN+DAN）
python visualization/viz_v3_prediction_case_study.py
#   自选子集
python visualization/viz_v3_prediction_case_study.py --baselines DAN EVL
python visualization/viz_v3_prediction_case_study.py --baselines LSTM ModernTCN DAN DLinear PatchTST EVL
#   覆盖某 baseline 的 checkpoint 路径（可重复）
python visualization/viz_v3_prediction_case_study.py --baselines DAN EVL --ckpt-dir DAN=checkpoints --ckpt-dir EVL=checkpoints
#   关闭 HA 参考线
python visualization/viz_v3_prediction_case_study.py --no-ha
```

所有输出到 `results_viz/`。

### 5.3 检查点（自验）

- 步骤 0 后：`ls checkpoints_viz/` 应有 `STCross_NYC_best.pt` + `DAN_NYC_best.pt`；可 `python -c "import torch; sd=torch.load('checkpoints_viz/STCross_NYC_best.pt',map_location='cpu',weights_only=True); print(sd['apb.embed'].shape)"` 确认 `apb.embed` shape=(8,64)（结构正确）。
- 步骤 1-3：确认 `results_viz/` 下对应 PNG 已生成。

### 5.4 故障排查

| 报错 | 原因 / 修复 |
|---|---|
| `ModuleNotFoundError: No module named 'eval'` | 未从项目根运行，或脚本缺 `sys.path.insert`。已修复（所有 viz/retrain 脚本均内置）。确认 cwd=项目根。 |
| `FileNotFoundError: ... STCross_NYC_best.pt` | 步骤 0（retrain）未跑。先跑 `python experiments/retrain_for_viz.py`。 |
| `FileNotFoundError: ... DAN_NYC_best.pt` | V3 用 DAN 但 retrain 未生成，或 DAN 在别的目录。用 `--ckpt-dir DAN=<dir>` 指向，或 `--baselines` 去掉 DAN。 |
| V2 `AssertionError: Z_s is None` | 加载的 STCross checkpoint 是 `use_city_branch=False` 变体。确认用步骤 0 的完整 STCross checkpoint。 |
| V3 区间选不出 / STCross 不赢 | 命令行会打印全候选 MAE 表，人工核查；可换 `--baselines` 子集或换区间（修改 `find_runs` 的 `MIN_RUN`）。 |
| 训练 OOM / 慢 | retrain 训练 3 个模型（STCross+ModernTCN+DAN），CPU 较慢（~30-60min）。有 GPU 自动用。 |

---

## 6. 与评审文档的关系

`doc/ablation_review_synthesis.md` 前沿教授指出"t-SNE 不足以验证城市不变性，必须补 domain classifier accuracy"。V2 已采纳：t-SNE（定性）+ LogisticRegression 域分类器（定量）双重验证。其余评审 P0 项（拆分 A1/A3 单变量、公平对照 A5、NBL 消融 A6、H1 Wasserstein、证伪条件表）不属于本轮 V1-V3 范围，见 TODO 5.5/5.7/6.1 待办。

## 7. 假设

- **V3 baseline**：用户已备齐所有 baseline checkpoint。脚本通过 `--baselines NAME [NAME...]` 选任意子集对比，每个 baseline 的 checkpoint 目录在脚本顶部 `BASELINE_CONFIG` 维护、可用 `--ckpt-dir NAME=PATH` 覆盖。DAN 默认指向 `checkpoints_viz/`（retrain_for_viz.py 重训），其余指向 `checkpoints/`（主结果）。HA 始终作为 NBL 参考线（无 checkpoint，`--no-ha` 关闭）。
- V1-V3 使用 α=0.3 checkpoint（用户决策重训）。若改用主结果 α=0.01 checkpoint，把 `load_stcross`/`load_learned_baseline` 的 `checkpoint_dir` 改为 `"checkpoints"`、`stcross_alpha=0.01` 即可。
