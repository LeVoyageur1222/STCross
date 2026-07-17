# STCross 论文大纲

> 目标会议：KDD / AAAI 主会研究论文  
> 参考材料：`writing/ref paper/IJCAI_26_MAP_STAR_main.pdf`、`writing/ref paper/ST13_Seeing the Unseen Learning Basis Confounder Representationsfor Robust Traffic Prediction.pdf`、`doc/model_design.md`、`TODO.md`、`results/RESULTS.md`  
> 写作目标：形成一篇围绕跨城市异常客流预测的英文论文大纲，目录细化到 `x.x` 小节。

---

## 0. 参考论文结构与写作风格

> 路径说明：当前仓库中的参考论文目录实际为 `writing/ref paper/`，其中包含 `IJCAI_26_MAP_STAR_main.pdf` 与 `ST13_Seeing the Unseen Learning Basis Confounder Representationsfor Robust Traffic Prediction.pdf`。本文大纲按 `IJCAI_26_MAP_STAR_main.pdf` 这类 IJCAI 方法论文的常见结构进行章节划分，并结合 ST13 的交通预测、混杂因素、解耦表示与鲁棒评估写法调整本文叙事。

### 0.1 参考论文到本文结构的映射

| 参考论文写法 | 本文对应章节 | 迁移方式 |
|---|---|---|
| 从实际城市预测痛点切入，再提出方法动机 | `1. Introduction` | 从异常客流和跨城市迁移难题切入，引出“正常基线 + 异常偏差”的核心 insight。 |
| 先定义问题与符号，再展开方法 | `3. Preliminaries` | 明确定义 source/target city、full/anomaly 子集、NBL 与 deviation space。 |
| 方法章节先给 overview，再拆模块 | `4. Methodology` | 先给 STCross 总体框架，再分解 NBL、DIP、DDAC、MIC、training objective 与 inference。 |
| 实验章节按设置、主结果、消融、分析展开 | `5. Experiments` | 主实验聚焦 `NYC→CHI` 与 `CHI→NYC`，并同时报告 full/anomaly；随后给消融、敏感性、可视化与公平性讨论。 |
| 结论前加入机制解释或讨论 | `6. Discussion` | 解释为什么偏差空间有助于迁移，并主动收敛因果/不变性相关 claim。 |

### 0.2 整体结构参考

参考 `IJCAI_26_MAP_STAR_main.pdf` 的方法型论文组织方式，本文建议采用以下主线：

1. **Introduction**：从实际城市异常事件预测需求切入，提出现有方法在跨城市迁移下的不足，再给出核心 insight 和贡献。
2. **Related Work**：按问题、方法和技术组件分组，而不是简单按模型罗列。
3. **Preliminaries / Problem Formulation**：先定义符号、数据、跨城市设置和异常评测口径。
4. **Methodology**：用 overview figure 串起完整框架，再逐个解释关键模块。
5. **Experiments**：先给实验设置，再给主结果、消融、敏感性、可视化和讨论。
6. **Conclusion**：总结方法价值、实证发现和未来工作。

### 0.3 用语风格参考

参考两篇论文的整体风格，建议本文保持以下表达方式：

- 多使用问题驱动句式：`However, existing methods ...`, `This raises a key question: ...`, `To address this challenge, we propose ...`。
- 方法命名采用短缩写，并在首次出现时给出全称，例如 `NBL`, `DIP`, `DDAC`, `APB`, `MIC`。
- 每个模块都用“motivation → design → effect”的顺序展开，避免只堆公式。
- 对理论或因果相关表述保持克制，用 `inspired by`, `encourages`, `regularizes`, `empirically validates`，避免过度声称严格因果保证。
- 实验分析要围绕 claim 展开：主结果验证有效性，消融验证组件必要性，敏感性验证鲁棒性，可视化验证可解释性与解耦效果。

### 0.4 两篇参考论文对本文的具体借鉴点

- `IJCAI_26_MAP_STAR_main.pdf`：借鉴其“问题动机 → 形式化定义 → 模块化方法 → 系统实验”的主线组织方式，使本文从跨城市异常预测问题自然过渡到 STCross 的 NBL、DIP 与 DDAC 设计。
- `ST13_Seeing the Unseen...pdf`：借鉴其对交通预测中混杂因素、分布偏移、表示解耦和鲁棒评测的叙事方式，但本文避免采用 N-dependent confounder bank，而强调 N-independent 的偏差空间原型与城市条件调整。
- 本文写法应弱化严格因果承诺，强调“启发式混杂缓解”和“经验验证”，与 ST13 的强混杂建模形成区分。

---

## 1. Introduction

### 1.1 Background: Anomalous Urban Flow Forecasting

说明城市客流预测在交通调度、公共安全、运力规划中的价值，并强调异常场景的重要性。异常场景包括节假日、极端天气等，它们会打破常规通勤模式，使普通时序预测模型显著失准。

建议核心句式：

- `Accurate urban flow forecasting under anomalous events is crucial for resilient transportation systems.`
- `Unlike routine traffic, anomalous flows are sparse, heterogeneous, and strongly affected by city-specific baselines.`

### 1.2 Challenge: Cross-City Transfer under Anomalies

引出本文核心问题：source 城市训练，target 城市评测。强调 NYC 与 CHI 的节点规模、流量尺度、城市结构不同，直接迁移面临困难。

需要突出三个挑战：

- 城市间正常客流基线差异大，绝对值空间难以迁移。
- 异常样本稀疏，节假日/天气事件在不同城市分布不均。
- 传统图模型常含 `N-dependent` 参数或邻接矩阵，面对 NYC 820 节点与 CHI 585 节点时无法直接迁移。

### 1.3 Key Insight: Baseline Is City-Specific, Deviation Is Transferable

提出本文核心 insight：

```text
Urban flow = city-specific normal baseline + city-agnostic anomalous deviation
Ŷ = NBL[future] + Δ̂
```

解释 `NBL` 捕获城市特定正常模式，`Δ` 捕获相对正常状态的异常偏差。相比在绝对客流空间学习，在偏差空间学习更接近可迁移的异常语义。

### 1.4 Our Approach: STCross

简要介绍 STCross 的两个核心组件：

- `DIP`：Deviation-Informed Prototyping，在偏差空间学习容量受限的可迁移异常原型。
- `DDAC`：Deviation-space Disentanglement And Conditioning，通过输入级归一化、容量级正则和互信息约束分离异常类型表示与城市特定表示。

强调最终预测为 `NBL + transferable deviation + city adjustment`。

### 1.5 Contributions

建议贡献写成 3 点：

1. 提出跨城市异常客流预测的新框架 STCross，将预测分解为城市特定正常基线和城市无关异常偏差。
2. 设计 DIP 与 DDAC，分别实现可迁移异常原型学习和偏差空间中的异常-城市解耦。
3. 在 NYC 与 CHI 的双向跨城市迁移实验中系统评估，主实验覆盖 `NYC→CHI` 和 `CHI→NYC` 的 full/anomaly 子集，结果优于 12 个 baseline，并通过消融、敏感性和可视化验证关键设计。

---

## 2. Related Work

### 2.1 Spatiotemporal Traffic Forecasting

综述传统时空预测模型和深度时序模型，包括 LSTM、DLinear、PatchTST、TimesNet、ModernTCN 等。重点指出它们通常在单城市或同分布设置下评测，对异常场景和跨城市迁移关注不足。

### 2.2 Graph-based Traffic Forecasting and Transferability Limitations

讨论 STGCN、DCRNN、GWNET、STGODE 等图模型。强调图模型能建模空间依赖，但常依赖固定节点集合、邻接矩阵或节点嵌入，因此在不同城市节点数不一致时难以 zero-shot 迁移。

### 2.3 Robust and Transferable Time Series Forecasting

讨论 TimeAlign、ProtoTS、ModernTCN 等与分布偏移、原型学习、可迁移时序预测相关的方法。突出它们的优势，但指出它们大多没有显式分离城市正常基线与异常偏差。

### 2.4 Extreme and Anomalous Event Forecasting

讨论 EVL、EALGAP、DAN、ST-SSDL 等异常/极端事件预测方法。指出这些方法能增强异常时段预测，但多数仍在绝对值空间建模，缺少针对跨城市异常迁移的结构化设计。

### 2.5 Disentanglement and Prototype Learning

讨论原型学习、表示解耦、互信息约束、自监督/因果启发式混杂缓解。这里可以引用 ST13 的思路作为相关工作，但要明确本文不同点：STCross 不学习 N-dependent confounder bank，而是在偏差空间学习 N-independent 的异常原型和城市条件调整。

---

## 3. Preliminaries

### 3.1 Notations

定义主要符号：

- `X_c ∈ R^{T × N_c}`：城市 `c` 的历史客流序列。
- `N_c`：城市 `c` 的节点数，NYC 为 820，CHI 为 585。
- `H`：输入窗口长度，本文为 12。
- `P`：预测窗口长度，本文为 12。
- `M_c`：异常掩码，由 holiday OR weather 得到。
- `Y_c`：未来客流标签。

### 3.2 Cross-City Anomalous Flow Forecasting

定义训练与测试设置：在 source city `s` 上训练模型，在 target city `t` 上直接评测。主实验聚焦两个跨城 setting：

- `NYC→CHI`
- `CHI→NYC`

论文正文主表建议只放这两个跨城 setting，并同时报告 full 与 anomaly 子集。

### 3.3 Normal Baseline and Deviation Space

定义 `NBL`：对训练期 normal-only 样本按 node、weekday、hour 统计正常基线。定义偏差：

```text
Δ = X - NBL[input]
Y = NBL[future] + Δ_future
```

强调 `NBL` 是非学习统计量，可在 target city 训练期直接计算，因此支持 zero-shot target inference。

### 3.4 Evaluation Subsets and Metrics

定义 full/anomaly 两个评测子集：

- `full`：全部测试样本。
- `anomaly`：预测窗口内异常比例达到 `anomaly_min_ratio=0.5` 的样本。

主实验建议报告 `MAE`, `RMSE`, `PCC`, `R²`，其中正文重点分析 `MAE` 与 `R²`，附表可包含 `MAPE`。

---

## 4. Methodology

### 4.1 Overview of STCross

给出整体框架图。图中应包含：

```text
X + timestamp + anomaly mask
→ NBL lookup
→ deviation Δ and normalized deviation Δ_norm
→ AEnc + APB for transferable anomaly deviation
→ CEnc + CAD for city-specific adjustment
→ MIC for redundancy reduction
→ Ŷ = NBL[future] + Δ̂_type + Δ̂_adj
```

本节用一段话概括 STCross 如何从 source 城市学习异常偏差，并在 target 城市复用模型参数。

### 4.2 Normal Baseline Lookup

介绍 `NBL` 的 normal-only 统计方式和三级回退机制：

1. `node + weekday + hour`
2. `node + hour`
3. `node global mean`

说明该模块不含可学习参数，目的是隔离城市特定正常模式，而不是作为最终预测器。

### 4.3 Deviation-Informed Prototyping

介绍 DIP 的核心流程：

```text
Δ = X - NBL[input]
Δ_norm = Δ / (std(Δ) + ε)
Z_a = AEnc(Δ_norm)
w = softmax(-dist(Z_a, APB.embed))
Δ̂_type = Σ w_i · APB.pattern_i
```

本节重点解释三点：

- 方差归一化消除城市幅度差异，保留异常形状。
- 变分编码和 `L_cap` 限制 `Z_a` 信息容量，压缩残留城市细节。
- APB 在偏差空间学习可解释的异常模式原型。

### 4.4 Deviation-space Disentanglement and Conditioning

介绍 DDAC：

```text
Z_a = AEnc(Δ_norm)  # transferable anomaly type
Z_s = CEnc(Δ)       # city-specific deviation context
Δ̂_adj = CAD(Z_s)
```

说明 `Z_a` 与 `Z_s` 的分工：`Z_a` 用于跨城可迁移异常类型，`Z_s` 用于目标城市偏差幅度和局部调整。

### 4.5 Mutual Information Constraint

介绍 `MIC` 的作用：减少 `Z_a` 和 `Z_s` 的冗余，鼓励二者学习互补信息。表述上应强调 `MIC` 不单独保证城市不变性，城市不变性主要来自输入级方差归一化和容量正则。

建议写法：

- `The MIC term does not enforce invariance by itself; instead, it discourages redundant information between the transferable and city-specific branches.`

### 4.6 Training Objective

给出总损失：

```text
L = L_pred + α L_cap + β L_mic + γ L_proto + ζ L_adj
```

逐项解释：

- `L_pred`：预测损失。
- `L_cap`：容量正则。
- `L_mic`：分支冗余约束。
- `L_proto`：原型判别性。
- `L_adj`：防止城市调整分支漂移。

### 4.7 Cross-City Inference

描述 source 训练和 target 推理流程。强调 target city 不进行参数微调，仅重新计算 target 的 NBL，然后复用 AEnc、APB、CEnc、CAD 和 MIC 相关训练后的参数。

### 4.8 Complexity and Model Size

简要说明模型规模约 123K 参数，属于中等规模，明显小于 TimesNet 等大模型。强调 STCross 是 channel-independent / N-independent，因此天然支持不同节点数城市间迁移。

---

## 5. Experiments

### 5.1 Experimental Setup

介绍数据、划分和训练设置：

- 城市：NYC 820 节点，CHI 585 节点。
- 时间范围：2014-10-01 至 2017-09-29。
- 粒度：60min。
- 划分：Train 15336，Val 5160，Test 5784。
- 窗口：`H=12 → P=12`。
- 异常定义：holiday OR weather，`anomaly_min_ratio=0.5`。
- 标准化：Per-node StandardScaler，fit on target city train split。

### 5.2 Baselines

按组介绍 12 个 baseline：

- Group A：LSTM、DLinear、PatchTST、TimesNet。
- Group C：TimeAlign、ProtoTS、ModernTCN。
- Group D：EVL、EALGAP、DAN、ST-SSDL。
- HA 可作为统计基线或附表结果，不建议放入跨城主表，因为 HA 无训练迁移概念。

同时说明 Group B 图模型为何不作为主 baseline：NYC 和 CHI 节点数不同，N-dependent 图结构无法 zero-shot 跨城迁移。

### 5.3 Main Results on Cross-City Transfer

这是正文最重要的实验小节。主表建议只保留用户指定的两个跨城 setting：

- `NYC→CHI`
- `CHI→NYC`

每个 setting 同时报告 full 和 anomaly 子集。建议表格结构如下：

| Model | NYC→CHI Full MAE | NYC→CHI Full R² | NYC→CHI Anom. MAE | NYC→CHI Anom. R² | CHI→NYC Full MAE | CHI→NYC Full R² | CHI→NYC Anom. MAE | CHI→NYC Anom. R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LSTM |  |  |  |  |  |  |  |  |
| DLinear |  |  |  |  |  |  |  |  |
| PatchTST |  |  |  |  |  |  |  |  |
| TimesNet |  |  |  |  |  |  |  |  |
| TimeAlign |  |  |  |  |  |  |  |  |
| ProtoTS |  |  |  |  |  |  |  |  |
| ModernTCN |  |  |  |  |  |  |  |  |
| EVL |  |  |  |  |  |  |  |  |
| EALGAP |  |  |  |  |  |  |  |  |
| DAN |  |  |  |  |  |  |  |  |
| ST-SSDL |  |  |  |  |  |  |  |  |
| STCross |  |  |  |  |  |  |  |  |

若版面允许，可加入 `PCC`；若版面紧张，正文主表只放 `MAE/R²`，`RMSE/MAPE/PCC` 放 appendix。

本节分析重点：

- STCross 在双向跨城迁移中 full 子集均显著优于 baseline。
- STCross 在 anomaly 子集上尤其提升 `R²/PCC`，说明不仅降低误差，还能更好解释异常波动。
- `NYC→CHI` 与 `CHI→NYC` 都有效，说明方法不是单向迁移技巧。

### 5.4 Same-City Results as Auxiliary Evidence

将 `NYC→NYC` 和 `CHI→CHI` 作为辅助结果，放正文小表或 appendix。该小节用于说明 STCross 不仅适用于跨城，也不牺牲同城预测能力。

建议正文只给一句概括：STCross 在四个 setting 的 full 指标上均最优；详细同城结果放 appendix。

### 5.5 Ablation Study

消融实验围绕核心 claim 组织：

- `w/o DevSpace`：去掉偏差空间、归一化和容量正则。
- `w/o ProtoBank`：APB 替换为普通预测头。
- `w/o CityBranch`：去掉 CEnc、CAD 和 MIC。
- `w/o MIC`：去掉互信息约束。

正文主表建议报告 `NYC→CHI anomaly`，因为这是最能体现跨城异常预测价值的设置。若后续整理完成，可补充 `CHI→NYC anomaly` 到 appendix。

### 5.6 Hyperparameter Sensitivity

展示 `α`, `K`, `γ` 的网格搜索结果：

- `α ∈ {0.02, 0.05, 0.1, 0.3}`
- `K ∈ {4, 8, 16, 32}`
- `γ ∈ {0.05, 0.1, 0.2, 0.4}`

说明最终采用 `α=0.3, K=8, γ=0.1, β=0.01`。本节重点不是堆所有图，而是说明 STCross 在合理参数范围内稳定，并解释为什么 `K=8` 能平衡表达能力和过拟合风险。

### 5.7 Visualization and Interpretability

本节对应 V1/V2/V3：

- V1：APB 学到的 8 个偏差原型 pattern 与使用分布。
- V2：`Z_a/Z_s` 城市不变性，使用 t-SNE 和 domain classifier accuracy。
- V3：节假日和极端天气 case study，展示 `NBL + Δ̂_type + Δ̂_adj` 的分解预测。

建议每个可视化只讲一个 claim：

- V1 证明原型不是退化的。
- V2 证明 `Z_a` 更城市不变，`Z_s` 更城市特定。
- V3 证明分解预测在典型异常段更可解释。

### 5.8 Fairness and Hypothesis Validation

回应潜在审稿质疑：STCross 用 anomaly mask 构造 normal-only NBL，而 baseline 没有直接使用 anomaly mask。建议包含两个补充分析：

- NBL 公平对照：baseline + deviation-space 或 full HA vs normal-only HA。
- H1 验证：比较 NYC 与 CHI 在 holiday/weather/normal 下的 `P(Δ|event)` Wasserstein 距离。

本节可以放正文后半或 appendix，取决于实验是否完成。

---

## 6. Discussion

### 6.1 Why Deviation Space Helps Transfer

从结果回到机制，解释为什么 `NBL + Δ` 比绝对值空间更适合跨城市迁移。重点强调：城市规模、节点数量和平均流量差异主要存在于基线中，而异常相对偏差形状更接近城市无关语义。

### 6.2 What STCross Does Not Claim

主动收敛 claim：

- 不声称严格因果识别。
- 不声称 MIC 单独保证城市不变性。
- 不声称所有异常事件都共享完全相同的跨城偏差分布。
- 当前主要验证 NYC 与 CHI，更多城市需要未来扩展。

这类诚实讨论有助于降低 AAAI/KDD 审稿人对“因果过度声称”和“泛化不足”的质疑。

### 6.3 Practical Implications

说明 STCross 的实践价值：

- 目标城市不需要重新训练深度模型，只需计算 NBL。
- N-independent 结构支持不同节点规模城市迁移。
- 原型和分解预测提供一定可解释性，便于异常交通管理。

---

## 7. Conclusion

### 7.1 Summary of Findings

总结本文提出 STCross，将跨城市异常客流预测分解为正常基线查找和偏差空间建模。实验表明 STCross 在 `NYC→CHI` 与 `CHI→NYC` 的 full/anomaly 子集上均取得领先结果。

### 7.2 Future Work

未来方向可包括：

- 扩展到更多城市和更多异常事件类型。
- 引入弱监督或无监督异常标签，减少对 anomaly mask 的依赖。
- 探索多城市联合训练下的强不变性约束。
- 将 STCross 应用于其他城市时空预测任务，如出行需求、道路速度或能源负荷。

---

## 8. 建议图表清单

### 8.1 Main Figures

- **Figure 1**：问题动机图，展示绝对值空间跨城差异大、偏差空间更可迁移。
- **Figure 2**：STCross 整体框架图。
- **Figure 3**：V1 原型 pattern 与使用分布。
- **Figure 4**：V2 `Z_a/Z_s` 城市不变性。
- **Figure 5**：V3 典型异常时段 case study。

### 8.2 Main Tables

- **Table 1**：跨城主结果，`NYC→CHI` 和 `CHI→NYC`，同时报告 full/anomaly。
- **Table 2**：消融实验，优先 `NYC→CHI anomaly`。
- **Table 3**：同城结果或参数量对比，可视版面放正文或 appendix。

### 8.3 Appendix Tables

- 完整四 setting 指标：`NYC→NYC`, `CHI→CHI`, `NYC→CHI`, `CHI→NYC`。
- 完整指标：`MAE`, `RMSE`, `MAPE`, `PCC`, `R²`。
- 超参数网格搜索详细结果。
- 公平性对照与 H1 Wasserstein 检验。

---

## 9. 推荐论文标题备选

### 9.1 偏 KDD 风格

**STCross: Deviation-Space Prototyping for Cross-City Anomalous Urban Flow Forecasting**

特点：直接、数据挖掘味更强，突出跨城市、异常客流和偏差空间原型。

### 9.2 偏 AAAI 风格

**Learning Transferable Anomaly Deviations for Cross-City Urban Flow Forecasting**

特点：更强调方法和可迁移表示，弱化系统名。

### 9.3 偏解释性风格

**Normal Baselines and Anomaly Deviations: A Disentangled Framework for Cross-City Urban Flow Forecasting**

特点：突出核心 insight，适合强调机制解释。
