# 消融方案三位教授审阅综合报告

> 审阅对象：`doc/ablation_plan.md`
> 审阅人：大导（宏观）+ 工程教授（实践）+ 前沿教授（理论）
> 日期：2026-06-29

---

## 一、三位教授的共识（5 项一致）

| # | 问题 | 大导 | 工程教授 | 前沿教授 |
|---|------|:---:|:---:|:---:|
| 1 | **A2 打包消融（归一化+L_cap）违反单一变量** | 🔴 | 🔴 | 🔴 |
| 2 | **A3 打包消融（CEnc+CAD+MIC）违反单一变量** | 🔴 | 🔴 | 🔴 |
| 3 | **缺失公平对照实验（baseline+偏差空间）** | 🔴 | 🔴 | 🔴 |
| 4 | **缺失 H1 假设检验（Wasserstein 距离）** | 🔴 | — | 🔴 |
| 5 | **缺失 NBL 设计消融（normal-only vs full HA）** | 🔴 | — | — |

## 二、各视角独特发现

### 大导（宏观）
- A1 实现描述模糊（X_encoder 未定义）
- A2 的 std-mask 处理未说明
- 迁移方向对称性（NYC→CHI vs CHI→NYC）遗漏
- 总评："方向正确、骨架在、但血肉不够"

### 工程教授（实践）
- 每个变体的**具体代码改动方案**（A1: 改2行, A2: 改1行, A3: 加flag+guard, A4: 替换模块）
- **S2/S3 必须补 α=0 和 β=0**（边界证据）
- **VariationalLSTM 需加 logvar.clamp(-10,10)** 防止高 α 时 NaN
- **V2/V3 需新增接口**：`encode()` 提取 Z_a/Z_s，`predict_with_decomposition()` 返回分解预测
- **patience 应分变体配置**：A2=20, S3=15, 其余=10
- **A3 的 mic_params 空列表需加守护**，否则 mic_optimizer.step() 报错
- **K=1 作为退化基线**（单原型退化为 FC）
- 总评："可实现性 OK，但 A2 必须拆分，公平对照必须补"

### 前沿教授（理论）— 最严厉
- **A1 也违反单一变量**：同时改了空间（偏差→绝对值）和 IB（保留 vs 移除）
- **A4 预期与理论矛盾**：DIP claim 原型是可迁移性载体，去掉应"显著变差"而非"略升"
- **S3（β/MIC）与 v7 叙事自相矛盾**：v7 已把 MIC 降级为"冗余去除非城市不变"，列为敏感性参数反而抬高其地位
- **S4（warmup）是工程旋钮非理论旋钮**，应移入 appendix
- **t-SNE 不足以验证城市不变性**：必须补 domain classifier accuracy + HSIC + 条件 Wasserstein
- **缺失反事实测试**（交换 Z_s/Z_a）— DDAC 解耦的最强可证伪实验
- **缺失 Mode A vs Mode B 对比** — "架构化足够"的 claim 无法实证
- **缺失证伪条件** — 所有预期都是"变差"，没有"若不变差则 claim 被证伪"
- **NBL future mask 因果泄露未讨论** — 不是工程不公平，是因果层面的信息泄露
- **ablation_plan 与 theoretical_framework 脱节** — 验证组件不验证理论
- 总评："理论写得严谨，但实验没验证理论，AAAI 直接被拒的典型 pattern"

## 三、综合风险排序

| 排名 | 风险 | 来源 | 严重度 |
|------|------|------|--------|
| 1 | A2/A3 打包消融，因果不可分 | 三人一致 | ⛔ 致命 |
| 2 | 缺公平对照（baseline+偏差空间） | 三人一致 | ⛔ 致命 |
| 3 | 消融验证组件不验证理论（与 theoretical_framework 脱节） | 前沿 | ⛔ 致命 |
| 4 | 缺证伪条件（全部"变差"无"若不变差则证伪"） | 前沿 | ⛔ 致命 |
| 5 | NBL future mask 因果泄露未讨论 | 前沿 | ⛔ 致命 |
| 6 | t-SNE 不足以验证城市不变性 | 前沿 | ⚠️ 高 |
| 7 | 缺 H1 Wasserstein 检验 | 大导+前沿 | ⚠️ 高 |
| 8 | A4 预期与 DIP 理论矛盾 | 前沿 | ⚠️ 高 |
| 9 | S3/S4 与 v7 叙事不符 | 前沿 | ⚠️ 高 |
| 10 | 缺反事实测试 + Mode A vs B | 前沿 | ⚠️ 中 |
| 11 | S2/S3 缺 α=0/β=0 边界 | 工程教授 | ⚠️ 中 |
| 12 | logvar clamp / patience 分变体 / 接口改造 | 工程教授 | 🔵 低 |

## 四、修改建议（综合三人意见）

### 必修（P0）

1. **拆分 A2**：A2a（仅去归一化，保留 L_cap）+ A2b（仅去 L_cap，保留归一化）
2. **拆分 A3**：A3a（去 CEnc+CAD，保留 MIC）+ A3b（去 MIC，保留 CEnc+CAD）
3. **补公平对照 A5**：DAN/ModernTCN + 偏差空间（NBL），证明增益来自架构
4. **补 H1 检验**：P(Δ|event) 跨城 Wasserstein 距离（可视化，非训练）
5. **补 NBL 消融 A6**：normal-only vs full HA vs instance mean
6. **明确 NBL 不消费 future anomaly label**：NBL 是训练期统计+时间戳查表
7. **A4 预期改为"跨城迁移显著变差"**：与 DIP claim 对齐
8. **补证伪条件表**：每个 claim → 证伪条件 → 实验编号

### 建议修（P1）

9. **V2 补 domain classifier accuracy**：线性 probe 分类城市，准确率接近 50% 才是城市不变
10. **S2/S3 补 α=0/β=0 边界点**
11. **S3 替换为 τ（std mask 阈值）敏感性**；S4 移入 appendix
12. **补反事实测试**（V5）：交换 Z_s/Z_a，验证解耦
13. **VariationalLSTM 加 logvar.clamp(-10,10)**
14. **patience 分变体**：A2=20, S3=15

### 可选（P2）

15. 补 Mode A vs Mode B 对比
16. 补迁移方向对称性（CHI→NYC）
17. 补 I(Z_a;City) vs I(Z_a;Event) MI 估计
18. K=1 退化基线

### 修订后实验量

| 类别 | 原方案 | 修订后 |
|------|--------|--------|
| 消融 | 4 变体 × 2 setting = 8 | 6 变体 × 2 setting = 12 (+公平对照 2) |
| 参数敏感性 | 4 参数 ~14 | 3 参数 + α=0/β=0 边界 ~12 |
| 可视化 | 4 项 | 5 项（+domain classifier） |
| **总计** | ~26 | **~33** |

仍远低于原始 50+，且 claim 闭环完整。
