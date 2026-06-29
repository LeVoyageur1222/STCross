# STCross: 偏差空间解耦的跨城市异常客流预测

> 统一设计文档（v7）
> 日期：2026-06-29
> 投稿目标：AAAI 时空数据挖掘 track
> v7 修订：模块重新命名（3-5字母功能缩写，不引用原论文名）

---

## 模块命名总览

| 缩写 | 全称 | 功能 |
|------|------|------|
| **NBL** | Normal Baseline Lookup | 正常基线查找表（非学习，数据统计） |
| **AEnc** | Anomaly Encoder | 异常类型编码器（变分编码+容量正则） |
| **CEnc** | City Encoder | 城市特定表征编码器 |
| **APB** | Anomaly Prototype Bank | 异常偏差原型库+模式匹配 |
| **CAD** | City Adjustment Decoder | 城市特定偏差调整解码器 |
| **MIC** | Mutual Information Constraint | 分支间互信息约束（冗余去除） |
| **DIP** | Deviation-Informed Prototyping | 创新点一：偏差空间+容量正则+原型匹配 |
| **DDAC** | Deviation-space Disentanglement And Conditioning | 创新点二：三层解耦+城市调整+混杂缓解 |

---

## 1. 动机与核心 Insight

### 1.1 问题定义

跨城市异常客流预测：在 source 城市（如 NYC）训练，直接迁移到 target 城市（如 CHI）预测节假日/极端天气时段的客流。核心挑战：异常模式如何跨城市迁移？

### 1.2 核心 Insight

**交通客流 = 城市特定正常基线 + 城市无关异常偏差**

```
Ŷ = NBL[future] + Δ̂
    ↑                ↑
    城市特定(不迁移)  城市无关(可迁移)
    数据统计(非学习)  模型学习
```

- **NBL**：同时段正常-only 历史均值（排除异常的训练数据），纯数据统计，任何城市可直接计算
- **Δ = X - NBL**：偏差信号，消去了城市绝对流量尺度，语义城市无关
- **Δ̂**：模型在偏差空间学习的偏差预测，可跨城市迁移

### 1.3 Baseline 实验支撑

| Baseline 发现 | 设计回应 |
|--------------|---------|
| DAN 在绝对值空间分解最优但不可迁移 | 在偏差空间分解 → 天然可迁移 |
| HA 在 CHI 同城强势 | NBL 是"正常-only HA"，更纯净 |
| STSSDL 偏差学习在 anomaly MAE 有优势 | 用偏差 + normal-only 基线 + 容量正则 |
| 大城市→小城市迁移更易 | 偏差模式城市无关 → 迁移天然支持 |

### 1.4 两个创新点及协同关系

**创新点一（DIP）**：偏差空间容量约束的可迁移异常原型。在偏差空间施加容量正则，Z_a 只保留对预测最关键的异常类型信息，通过原型匹配实现跨城市可迁移。

**创新点二（DDAC）**：偏差空间的异常-城市分离与条件化。通过输入级方差归一化 + 容量级正则 + 冗余级 MIC 的递进式解耦，分离异常类型（Z_a，可迁移）和城市特定表征（Z_s，自适应），并提供城市特定偏差调整。

**协同关系**：DIP 提供可迁移的 Z_a 表示（"表示能迁移"），DDAC 确保 Z_a 只含可迁移成分、城市特定信息由 Z_s 自适应处理（"表示够纯净"）。两者构成"可迁移 + 纯净"的双重保证。

---

## 2. NBL — 正常基线查找

### 2.1 定义

```
NBL[node, weekday, hour] = mean(flow[node, t]
    for t in training_period
    WHERE not is_anomaly(t) AND t.weekday == weekday AND t.hour == hour)
```

- 形状 (N, 7, 24)，纯数据统计，无学习参数
- 排除异常样本 → 基线更纯净（vs HA 用含异常的全量均值）

### 2.2 三级回退

```
Level 1: NBL[node, weekday, hour]   # 正常-only 同 slot (≥5样本)
Level 2: NBL[node, hour]            # 去 weekday 维度
Level 3: NBL[node]                  # 全局正常均值
```

### 2.3 目标城市获取（零训练）

NBL 是数据统计，target 城市的训练期流量和异常标签在框架中已有，直接计算即可。

### 2.4 与 RevIN 的区分

RevIN（Kim et al., ICLR 2022）在表示层做 learnable affine 变换并反归一化以稳定训练。STCross 的 NBL + 方差归一化在数据层做确定性统计/除法且**不反归一化**——目的是强制 Z_a 只接收形状信息，不是稳定分布。

---

## 3. 创新点一：DIP — 偏差空间容量约束的可迁移异常原型

### 3.1 偏差计算与方差归一化

```
Δ = X - NBL[input_window]                 # 偏差
Δ_norm = Δ / (std(Δ) + ε)                # 方差归一化: 消除幅度, 保留形状
```

- 正常时段 Δ ≈ 0；异常时段 Δ ≠ 0
- Δ_norm 只保留偏差**形状**（何时升降、几个标准差），消除**幅度**
- 形状是城市无关的（"节假日早高峰下降 1σ"在 NYC 和 CHI 一致）

**std(Δ) 计算**：在 H 维度上算每个样本的标准差。当 std(Δ) < τ（如 0.01）时，mask 该样本的 Z_a 损失，避免归一化爆炸。

### 3.2 容量正则（AEnc 的变分编码）

**动机**：归一化后 Δ_norm 仍可能含城市特定形状细节。需要约束 Z_a 的信息容量，迫使它只保留最通用的异常类型信息。

**直觉**：在"笔记字数受限"的压力下，Z_a 优先保留对所有城市通用的异常模式（信息效率高），丢弃城市特定细节（信息效率低）。

**实现**：AEnc 用变分编码 + KL 约束作为容量正则：

```
Z_a ~ N(μ_a, σ_a²)          # 重参数化采样
L_cap = KL(Z_a ‖ N(0,1))   # 容量正则: 限制 Z_a 信息量
```

**诚实说明**：KL 到 N(0,1) 与 β-VAE 同构，是 I(Z_a; Δ_norm) 的变分上界，不是严格的信息瓶颈最优解。定位为**容量正则**，效果通过消融验证。

### 3.3 APB — 异常原型匹配

```
Z_a = AEnc(Δ_norm)                        # LSTM + 变分编码
w = softmax(-dist(Z_a, APB.embed))        # K 个原型匹配
Δ̂_type = Σ w_i · APB.pattern_i            # 偏差原型模式加权和
```

APB 在归一化偏差空间学习，每个原型代表一种可迁移的异常偏差形状（如"节假日早高峰下降""极端天气全天抑制""正常波动"）。因在归一化空间学习，原型天然城市无关。

### 3.4 可迁移性

**假设 H1**：偏差分布 P(Δ | event_type) 在城市间近似一致。

若 H1 成立，source 城市训练的 AEnc + APB 可直接用于 target 城市。因为 Δ_norm 已消除城市基线和幅度，且容量正则压缩了残留城市细节。

**H1 实证检验**：报告 P(Δ|event) 在 NYC vs CHI 的 Wasserstein 距离（§8）。

---

## 4. 创新点二：DDAC — 偏差空间的异常-城市分离与条件化

### 4.1 为什么需要 DDAC

DIP 提供了可迁移的 Z_a，但：
1. Z_a 仍可能含残留城市形状细节（容量正则不保证 selective disentanglement）
2. 缺少城市特定偏差调整机制（如 NYC 周五晚额外降的幅度）

DDAC 通过三层递进机制解决这两个问题。

### 4.2 城市特定表征的完整分析

偏差 Δ 中不可迁移的城市特定信息包括：幅度、时间形态、空间异质性、交互效应、季节调制。Z_s 分支（CEnc + CAD）负责捕获这些信息，提供城市特定偏差调整 Δ̂_adj。

### 4.3 三层解耦架构

| 层 | 机制 | 约束关系 | 作用 | 保证 |
|---|------|---------|------|------|
| **输入级** | 方差归一化 | Z_a 输入 vs Z_s 输入 | 确定性消除幅度 | 强（不可逆） |
| **容量级** | L_cap (KL) | Z_a ↔ 输入 | 限制 Z_a 信息量 | 中（容量正则） |
| **冗余级** | MIC | Z_a ↔ Z_s | 分支不冗余 | 中（互信息上界） |

```
偏差 Δ
  ├─ Δ_norm = Δ/std(Δ) ──→ AEnc ──────→ Z_a (异常类型, 可迁移)
  │                        + L_cap       → APB匹配 → Δ̂_type
  │
  └─ Δ (原始) ──────────→ CEnc ──────→ Z_s (城市特定表征)
                                       → CAD ────→ Δ̂_adj
  │
  MIC: I(Z_s; Z_a) → 0  (分支冗余去除)
  │
  Δ̂ = Δ̂_type + Δ̂_adj
  Ŷ = NBL[future] + Δ̂
```

### 4.4 MIC — 互信息约束

MIC 最小化 I(Z_s; Z_a)：
- ✅ 保证：Z_s 和 Z_a 统计独立（分支间无冗余，分工明确）
- ❌ **不保证**：Z_a 城市不变性（一个变量可以与 Z_s 独立但仍含城市信息）

城市不变性由**输入级（归一化）+ 容量级（L_cap）**保证，不由 MIC 保证。MIC 的角色是确保 Z_s 捕获 Z_a 无法捕获的**互补信息**（城市特定表征）。

**实现**：两阶段优化——阶段 1 训练 MIC 估计器 q_φ 估计互信息上界（每 3-5 步更新 1 次），阶段 2 最小化互信息（每步更新编码器）。

### 4.5 CAD — 城市调整解码器

```
Δ̂_adj = CAD(Z_s)    # FC: latent_dim → P
```

- 从 Z_s 预测城市特定偏差调整
- **防漂移**：加范数惩罚 L_adj = ‖Δ̂_adj‖²，逼 CAD 只做"小幅度城市调整"，不被 L_pred 压成兜底项

### 4.6 CEnc 的跨城市迁移

CEnc 学习的是通用函数"如何从偏差序列提取特征"（参数通用），不是城市特定参数。偏差空间 Δ 语义城市一致（Δ=-0.3 在任何城市都表示"低于正常 30%"），因此 CEnc 可处理任何城市数据。Z_s_chi 之所以是 CHI 特定的，是因为输入 Δ_chi 是 CHI 的数据。

### 4.7 混杂缓解（受 Pearl 启发）

架构设计受 Pearl 后门调整的直觉启发：Z_s（城市特定表征）是潜在混杂因素，Z_a（异常类型）是关心的效应。三层机制减少 Z_a 中的城市混杂信息。

**诚实定位**：由于(1)不同城市的异常事件分布不同（City→Event 关系存在），(2) Z_a/Z_s 是学习得到的隐变量，不声称形式化因果保证。定位为**启发式混杂缓解**，有效性通过跨城迁移实验实证验证。

### 4.8 单城市弱不变性与双模式

**张力承认**：IRM/DANN 需多环境信号，单城市训练下"城市不变性"缺乏参照系。

**STCross 定位**：架构化不变性（方差归一化 + L_cap），非学习型不变性（GRL/IRM）。提供"弱保证"：Z_a 不含幅度（确定性）+ Z_a 信息量受限（容量正则）。可迁移性是经验验证的。

**双模式**：
- Mode A（单城市 zero-shot）：架构化弱不变，主实验
- Mode B（多城市联合，可选）：加 IRM 约束，学习型强不变

---

## 5. 整体架构

```
输入: X (B, H, N) + 异常标签 + 时间戳

Step 1: NBL = 数据统计 (查表, 非学习)
  B_input = NBL[input_window]
  B_future = NBL[future_window]

Step 2: 偏差 + 归一化
  Δ = X - B_input
  Δ_norm = Δ / (std(Δ) + ε)     # 小 std 时 mask Z_a 损失

Step 3: 双分支编码
  Z_a ~ AEnc(Δ_norm)             # 异常类型 (归一化+L_cap → 弱不变)
  Z_s = CEnc(Δ)                  # 城市特定表征 (完整偏差)

Step 4: 偏差预测
  w = softmax(-dist(Z_a, APB))   # 原型匹配 (仅用 Z_a)
  Δ̂_type = Σ w_i · APB.pattern_i # 可迁移异常类型偏差
  Δ̂_adj = CAD(Z_s)              # 城市特定调整
  Δ̂ = Δ̂_type + Δ̂_adj

Step 5: 最终预测
  Ŷ = B_future + Δ̂

Step 6: 损失 (4 核心 + 1 稳定性)
  L = L_pred + α·L_cap + β·L_mic + γ·L_proto + ζ·L_adj
  [+ 可选: L_dev, L_ent 进消融]
  [+ Mode B: L_IRM]
```

---

## 6. 损失函数

### 6.1 核心损失（4 项）

| 损失 | 公式 | 作用 |
|------|------|------|
| L_pred | MSE(Ŷ, Y) | 预测准确性 |
| L_cap | KL(Z_a ‖ N(0,1)) | 容量正则: 限制 Z_a 信息量, 挤出城市细节 |
| L_proto | TripletMargin(Z_a, P_pos, P_neg) | 原型判别性: 不同异常类型可区分 |
| L_mic | MIC(Z_s, Z_a) | 冗余去除: Z_s 与 Z_a 分工明确 |

### 6.2 稳定性正则（1 项）

| 损失 | 公式 | 作用 |
|------|------|------|
| L_adj | ‖Δ̂_adj‖² | 防 CAD 漂移: 逼其只做小幅度城市调整 |

### 6.3 消融损失（2 项，验证后决定保留）

| 损失 | 公式 | 作用 |
|------|------|------|
| L_dev | MSE(‖Z_a-P*‖, \|Δ̂-Δ_true\|) | 偏差距离-误差一致性 (可解释性) |
| L_ent | -Σw_i log(w_i) | 稀疏原型激活 |

### 6.4 可选（Mode B）

| 损失 | 公式 | 作用 |
|------|------|------|
| L_IRM | Σ_e L_e(Z_a) | 多环境不变风险最小化 |

---

## 7. 跨城市迁移

```
Source 训练 (NYC):
  NBL_nyc = 数据统计
  Δ_nyc = X_nyc - NBL_nyc; Δ_norm_nyc = Δ_nyc / std(Δ_nyc)
  Z_a = AEnc(Δ_norm_nyc)  → APB匹配 → Δ̂_type
  Z_s = CEnc(Δ_nyc)       → CAD → Δ̂_adj
  训练: AEnc, CEnc, CAD, APB, MIC 估计器

Target 推理 (CHI, 零训练):
  NBL_chi = 数据统计 (从 CHI 训练数据+异常标签)
  Δ_chi = X_chi - NBL_chi; Δ_norm_chi = Δ_chi / std(Δ_chi)
  Z_a^chi = AEnc(Δ_norm_chi)       # 复用! 归一化保证城市不变
  Z_s^chi = CEnc(Δ_chi)            # 复用! 通用函数处理 CHI 数据
  w^chi = softmax(-dist(Z_a^chi, APB))  # 复用原型!
  Δ̂_type = Σ w_i · APB.pattern_i        # 可迁移: 异常类型形状
  Δ̂_adj = CAD(Z_s^chi)                  # 自适应: CHI 的表征调整
  Ŷ_chi = NBL_chi[future] + Δ̂_type + Δ̂_adj
```

---

## 8. 评测公平性与假设验证

### 8.1 公平对照

STCross 用 anomaly_mask 计算 NBL，baseline 不用 → 信息不对称。

| 实验 | NBL 来源 | baseline 也用 | 目的 |
|------|---------|-------------|------|
| 主实验 | normal-only HA | STCross 独有 | 完整性能 |
| 公平对照 | normal-only HA | ✅ baseline 也在偏差空间 | 证明增益来自架构 |
| 标签消融 | 随机 mask | STCross 独有 | 验证不依赖标签质量 |

### 8.2 假设 H1 验证

对每类 event ∈ {holiday, weather, normal}，报告 P(Δ|event) 在 NYC vs CHI 的 Wasserstein 距离。距离小 → 可迁移性有支撑；距离大 → 诚实报告，强调 Z_s+CAD 的自适应补偿。

---

## 9. 参数量

| 组件 | 参数量 |
|------|--------|
| AEnc (LSTM+变分, hidden=64, 2层) | ~58K |
| CEnc (LSTM, hidden=64, 2层) | ~50K |
| CAD (FC, 64→P) | ~1K |
| APB (K=8, 嵌入+模式) | ~0.6K |
| MIC 估计器 (64→32→64) | ~4K |
| **总计** | **~114K** |

---

## 10. 与 Baseline 的区分

| 维度 | DAN | STSSDL | ProtoTS | STEVE | **STCross** |
|------|-----|--------|---------|-------|------------|
| 预测空间 | 绝对值 | 绝对值 | 绝对值 | 绝对值 | **偏差+基线** |
| 基线 | 无 | instance mean | 无 | 无 | **NBL (normal-only)** |
| 原型 | 无 | 无 | 绝对曲线 | 隐式向量 | **APB (偏差形状+容量约束)** |
| 不变性机制 | 无 | 无 | 无 | GRL(需多城市) | **归一化+L_cap(单城市)** |
| 城市调整 | 无 | 无 | 无 | 无 | **CAD** |
| 可迁移性 | 部分 | 部分 | 部分 | ❌ N-dependent | **容量正则+假设H1** |
| 混杂缓解 | 无 | 无 | 无 | 无 | **启发式(受Pearl启发)** |

---

## 11. 消融实验

### 11.1 核心消融

| 消融 | 验证的 claim |
|------|-------------|
| 偏差空间 vs 绝对值空间 | 偏差空间分解的必要性 |
| NBL: normal-only vs full HA vs instance mean | 正常-only 基线的价值 |
| 归一化 vs 无归一化 | 输入级分离的效果 |
| L_cap vs 无 L_cap | 容量级约束的效果 |
| MIC vs 无 MIC | 冗余级约束的效果 |

### 11.2 三层组合消融

| 组合 | 验证 |
|------|------|
| 无归一化 + 无L_cap + 无MIC | 基线(单编码器) |
| 归一化 only | 输入级单独效果 |
| 归一化 + L_cap | 输入+容量 |
| 归一化 + L_cap + MIC | 完整三层 |
| 归一化 + L_cap + cos正交 | MIC vs 简化版 |

### 11.3 组件消融

| 消融 | 验证 |
|------|------|
| 仅 Δ̂_type vs 仅 Δ̂_adj vs 两者 | CAD 角色 |
| K=2/4/8/16 | 原型数敏感性 |
| L_dev on/off | 偏差一致性 |
| L_ent on/off | 稀疏激活 |
| Mode A vs Mode B | 单城市 vs 多城市 |

### 11.4 公平性与假设

| 消融 | 验证 |
|------|------|
| baseline + 偏差空间 vs baseline | 增益来自架构还是信息 |
| 随机 mask vs anomaly_mask | NBL 是否依赖标签质量 |
| P(Δ\|event) 跨城 Wasserstein | 假设 H1 检验 |
| NYC→CHI vs CHI→NYC | 迁移方向对称性 |

---

## 12. 理论保证（简洁版）

| 保证级别 | 性质 | 机制 |
|---------|------|------|
| **确定性保证** | Z_a 不含幅度 | 方差归一化（不可逆操作） |
| **框架保证** | Z_a 信息量受限; Z_s⊥Z_a | L_cap 容量正则; MIC 互信息上界 |
| **经验验证** | 可迁移性; 混杂减少 | 跨城迁移实验; P(Δ\|event) 分布距离 |

不过度声称，明确区分确定性/框架/经验三个层级。

---

## 13. 代码实现

> 文件：`model/stcross.py`

### 模块结构（类层次）

| 组件 | 类/函数 | 职责 |
|------|---------|------|
| NBL | `compute_nbl()` / `lookup_nbl()` (模块级函数) | 正常基线统计与查表（三级回退） |
| AEnc | `VariationalLSTM(nn.Module)` | LSTM + 变分编码 (μ, σ, Z_a) |
| CEnc | `CityLSTM(nn.Module)` | LSTM 城市表征编码 (Z_s) |
| APB | `AnomalyProtoBank(nn.Module)` | 原型嵌入 + 模式 + 匹配 |
| CAD | `CityAdjDecoder(nn.Module)` | FC 城市调整解码 (Δ̂_adj) |
| MIC | `MICEstimator(nn.Module)` | 互信息约束估计器 |
| 主模型 | `STCrossModel(TorchModel)` | 主模型，含 `_net_forward()`, `fit()`, `predict_with_baseline()` |

### 关键实现细节

- **APB 手动距离计算**：使用 `||z||² + ||e||² - 2*z@e.T` 替代 `torch.cdist`，规避 MPS 后端 backward bug
- **MIC 两阶段优化**：每 3 步更新一次 q_φ 估计器，每步更新编码器（AEnc/CEnc）
- **自定义 `fit()`**：为 main params 和 MIC 估计器分别配置 optimizer
- **`predict_with_baseline()`**：NBL-aware 推理接口，确保 Ŷ = NBL[future] + Δ̂_type + Δ̂_adj
- **注册**：在 `model/registry.py` 中注册为 `"STCross"`

### 参数量

约 ~123K（AEnc ~58K + CEnc ~50K + CAD ~1K + APB ~0.6K + MIC ~4K + 其余）。
