# 代码审查问题修复计划

> 基于 5 份博士生审查报告 (doc/review_synthesis.md) 的后续修复计划
> 更新日期: 2026-06-25

---

## 已修复问题

### P0 (阻塞性) — 全部已修复

| 编号 | 问题 | 修复方式 | 状态 |
|------|------|---------|------|
| H1 | GWNET 输出形状错误 | 经实测验证，skip 时间维度已收缩到 1，squeeze(-1) 工作正常 — **误报** | ✅ 已验证 |
| H2 | 交叉城市 Group B N 维度不匹配 | 已验证 4 个模型全部崩溃，从 registry 移除，Phase 2 标记为废弃 | ✅ 已修复 |
| H3 | DAN 架构简化 (FC) | 重写为 CNN+LSTM 编码器/解码器 (134K params) | ✅ 已修复 |
| H4 | ST-SSDL 架构简化 (FC) | 重写为 LSTM 编码器 + Wq 投影 (64K params) | ✅ 已修复 |

### P1 (高优先级) — 已修复

| 编号 | 问题 | 修复方式 | 状态 |
|------|------|---------|------|
| M4 | EALGAP QKV Softmax 激活 | Softmax → GELU (中间层)，Softmax 保留在最终层 | ✅ 已修复 |
| M5 | EALGAP instance z-score | 文档化为已知近似 | ✅ 已修复 |
| M9 | EALGAP PDF 动态计算 | 文档化为已知近似 | ✅ 已修复 |
| M1 | 自定义 fit 与标准 Trainer 间大量代码重复 | 提取 `TrainingLoop` 公共类，自定义 fit 只需提供 `train_step_fn` 和 `val_step_fn` 回调 | ✅ 已修复 |

---

## 待修复问题

### P2 (中等优先级) — 2 项待修复

| 编号 | 问题 | 文件 | 严重程度 | 建议修复 |
|------|------|------|---------|---------|
| M2 | **STGCN 输出层与论文偏差** — 模型已废弃，无需修复 | `model/stgcn.py` | N/A | 模型已从 registry 移除 |
| M3 | ~~**TimeAlign RevIN 状态管理**~~ | `model/timealign.py` | ✅ 已修复 | 重构为两个独立 RevIN 实例 |
| M6 | **ST-SSDL 对比损失** — 当前使用 triplet margin loss，论文可能使用 InfoNCE 风格对比损失 | `model/stssdl.py` | LOW | 验证论文原始公式，若不同则修改。预计 2-3 小时 |
| M7 | **ST-SSDL historical anchoring** — 当前使用 `x.mean(dim=1)` (instance mean)，论文使用 running mean | `model/stssdl.py` | LOW | 评估是否可在 per-node 框架下实现 EMA。预计 1-2 小时 |
| M8 | **交叉城市无目标城市验证集** — 早停基于源城市验证损失 | `eval/evaluator.py` | LOW | 论文中讨论此局限性即可，无需代码修改 |

### P3 (低优先级) — 13 项待修复 (4 项已修复)

| 编号 | 问题 | 文件 | 建议 |
|------|------|------|------|
| L1 | DCRNN 无 scheduled sampling | `model/dcrnn.py` | 模型已废弃 |
| L2 | TimesNet MPS 设备强制回退 CPU | `model/timesnet.py` | GPU 服务器运行则无影响 |
| L3 | STGODE 双分支共用同一邻接矩阵 | `model/stgode.py` | 模型已废弃 |
| L4 | PatchTST 短序列仅 5 patches | `model/patchtst.py` | H=12 固有局限，可考虑增大 H |
| L5 | DLinear kernel_size=3 对 H=12 效果有限 | `model/dlinear.py` | 可考虑 kernel_size=5 |
| L6 | `model.predict()` 无进度条 | `model/torch_base.py` | 添加 tqdm 包装 |
| L7 | Checkpoint 路径为相对路径 | `train/trainer.py` | 使用 os.path.abspath |
| L8 | `calc_pcc` 零方差返回 0.0 而非 nan | `metrics/metrics.py` | 返回 nan 更语义化 |
| L9 | `_config_matches` 不检查 `test_steps` | `experiments/run_all.py` | 添加 test_steps 检查 |
| L10 | ~~`run_ha_baseline.py` 硬编码 `output_dir`~~ | `experiments/run_ha_baseline.py` | ✅ 已修复 | 改为 `--output_dir` 命令行参数，默认 `results` |
| L11 | `predict()` 中 batch_size 硬编码 2x 放大 | `model/torch_base.py` | 可配置化 |
| L12 | ~~训练中无 NaN/Inf 检测~~ | `train/trainer.py` | ✅ 已修复 | 添加 `torch.isnan(loss).any()` 检测，遇到 NaN/Inf 时跳过该 batch |
| L13 | ~~结果缓存机制无强制重跑选项~~ | `experiments/run_all.py` | ✅ 已修复 | 添加 `--force` 参数，跳过缓存检查强制重跑 |
| L14 | `build_flow_data.py` 硬编码日期范围 | `data_preprocess/build_flow_data.py` | 改为可配置参数 |
| L15 | 极端天气类型已定义但未启用 | `data_preprocess/build_weather_mask.py` | 文档化或移除 |
| L16 | ~~EVL LSTM 仅使用最后一层隐藏状态~~ | `model/evl.py` | ✅ 已修复 | 改为使用所有 LSTM 输出的均值 (`mean(dim=1)`) |
| L17 | TimeAlign `_patchify` 使用 `unfold` | `model/timealign.py` | 验证是否有时间步丢失 |

---

## 修复优先级建议

### 建议修复 (论文撰写前)

| 序号 | 问题 | 预计工时 | 负责人 |
|------|------|---------|--------|
| 1 | L11: predict() batch_size 可配置化 | 30min | 训练管线的学生 |
| 2 | L14: build_flow_data.py 日期范围参数化 | 30min | 数据预处理的学生 |

### 可选修复 (时间允许时)

| 序号 | 问题 | 预计工时 |
|------|------|---------|
| 3 | M6/M7: ST-SSDL 对比损失 + Historical anchoring | 3-5h |
| 4 | L4-L17: 其他 LOW 问题 | 各 10-30min |

---

## 当前模型清单 (12 个)

| # | Group | 模型 | 文件 | 参数量 | 状态 |
|---|-------|------|------|--------|------|
| 1 | A | HA | `model/ha.py` | 0 | ✅ |
| 2 | A | LSTM | `model/lstm.py` | 51K | ✅ |
| 3 | A | DLinear | `model/dlinear.py` | 312 | ✅ |
| 4 | A | PatchTST | `model/patchtst.py` | 71K | ✅ |
| 5 | A | TimesNet | `model/timesnet.py` | 1.2M | ✅ |
| 6 | C | TimeAlign | `model/timealign.py` | 148K | ✅ |
| 7 | C | ProtoTS | `model/protots.py` | 6K | ✅ |
| 8 | C | ModernTCN | `model/moderntcn.py` | 39K | ✅ |
| 9 | D | EVL | `model/evl.py` | 74K | ✅ |
| 10 | D | EALGAP | `model/ealgap.py` | 244K | ✅ |
| 11 | D | DAN | `model/dan.py` | 134K | ✅ |
| 12 | D | STSSDL | `model/stssdl.py` | 64K | ✅ |

---

## 总结

- **4 个 HIGH 问题**: 3 个已修复，1 个 (GWNET) 经实测验证为误报（模型已废弃）
- **9 个 MEDIUM 问题**: 5 个已修复（M1/M3/M4/M5/M9），3 个随 Group B 废弃（M2/M8 部分），仅 M6/M7 待修复
- **17 个 LOW 问题**: 4 个已修复（L10/L12/L13/L16），13 个待修复
- **当前状态**: 12 个模型代码就绪，可以开始正式实验