# STCross 模型设计文档工程审阅报告

> 审阅人视角：工程项目实践 / 实现可行性 / 训练稳定性 / 实验严谨性
> 审阅对象：`doc/model_design.md` (v4)
> 参考代码：`model/torch_base.py`、`eval/evaluator.py`
> 日期：2026-06-29

作为带过多个学生、亲手写过不少"看起来很美、调起来很崩"的模型的过来人，我对这份设计文档的总体评价是：**想法扎实、理论自洽，但工程上踩坑点比文档愿意承认的多**。下面按八个维度展开，不空谈、只讲落地。

---

## 1. 实现可行性

逐模块拆：

- **B_normal 计算**：纯 numpy/pandas 统计即可，无技术难点。但要实现为可缓存、可热加载的预处理产物（每个城市一个 `.npy`），不要塞进 DataLoader 在线算——每条样本都查表会拖垮训练吞吐。
- **Encoder_a / Encoder_s（双 LSTM 分支）**：标准实现。注意 per_node 模式下 `_net_forward` 收到的是 `(B*N, H, 1)`，每个节点独立跑 LSTM，这意味着原型匹配、CLUB 估计器都要在 `B*N` 维度上做，显存压力大（见 §3）。
- **变分编码 + IB（KL → N(0,1)）**：标准 VAE 重参数化，无难点。但 `σ_a` 用 `softplus + 1e-4` 加下界，避免 KL 里 `log σ²` 数值爆炸。
- **CLUB 估计器**：这是**最容易翻车**的模块。CLUB 原始论文（Cheng et al. 2020）的两阶段交替优化在单 GPU 上调试很烦：阶段 1 训练 `q_φ` 时 detach `Z_s/Z_a`、阶段 2 训练编码器时 detach `q_φ`，标准的实现是手动构造两个 `optimizer.step()` 交替调用。**关键工程难点**：CLUB 估计器极易过拟合到当前 batch 的边缘分布，导致 `q_φ` 训练太快、编码器来不及响应，表现为 MI 估计值上下震荡而真实互信息根本没下降。建议每 N 步编码器更新才更新 1 步 `q_φ`（N=3~5），并且给 CLUB loss 加梯度裁剪。
- **偏差原型 P + pattern**：`nn.Embedding(K, d)` + `nn.Linear(d, P)` 即可，无难点。TripletMargin 损失需要正负样本采样策略——文档没写**怎么选 P_pos / P_neg**，这是实现时的真实坑：是按事件类型？按 w 的 top-2？建议明确。
- **Decoder_adj**：1 层 FC，简单。但要确保它**不被 L_pred 直接压成"Δ̂_adj ≈ Y - B_normal - Δ̂_type"**（即变成兜底项，把 Z_a 的信息挤出去）。这是消解耦最常见的失败模式：约束太弱时 Decoder_adj 直接接管全部偏差预测，Z_a + 原型形同虚设。

**总体**：全部可在 PyTorch 实现，难点集中在 **CLUB 的两阶段优化稳定性**和 **Decoder_adj 的角色漂移**。

---

## 2. 训练稳定性

六项损失 `L_pred + L_IB + L_mi + L_proto + L_dev + L_ent` 联合优化，**α/β/γ/δ/ε 五个超参**，这是典型的不稳定配置：

- **CLUB 两阶段**：见 §1，必踩。两阶段优化在深度学习中历来脆弱（参考 GAN、BiGAN），需要 early stopping、warmup、梯度裁剪全套上。
- **方差归一化 `Δ_norm = Δ / std(Δ)`**：这是**最严重的数值隐患**。当某节点某窗口的 Δ 本身接近 0（正常时段），`std(Δ)` 趋近 0，归一化后变成噪声/爆炸。文档加了 `ε` 但没说量级。实测中 `std(Δ)` 在 per_node 粒度可能极小（住宅区深夜），`ε=1e-8` 也救不回来，会把噪声放大几个数量级灌进 Encoder_a。**强烈建议**：用全局 std 或滑动窗口 std 替代逐样本 std，并对 `std(Δ) < τ` 的窗口直接 mask 掉 Z_a 的损失（视为无异常类型信号）。
- **L_dev 形式可疑**：`MSE(||Z - P*||, |Δ̂ - Δ_true|)` 把"表示空间距离"和"输出空间误差"对齐——量纲不一致、量级不一致，β 调起来很敏感。建议改成归一化的相关性损失或 rank loss。
- **L_ent 稀疏激活**：熵正则容易把 w 推向 one-hot，导致梯度只流过 1 个原型、其他原型学不到东西（原型坍塌）。需要加均匀性约束（参考 SWAV 的 sinkhorn）或 EQ-normalization。

**建议**：先冻结 CLUB 和 IB，单独训 `L_pred + L_proto + L_ent` 验证主路径能跑通、能收敛，再逐项打开。**不要一上来六项全开**。

---

## 3. 计算效率

- 双分支 LSTM：参数 ~114K（文档估算合理），但**FLOPs 接近 baseline 的 2 倍**（两条 LSTM 串行+CLUB 网络+原型匹配）。
- per_node 模式下 effective batch = `B * N`，NYC 是 820 节点，B=32 → effective 26240。CLUB 估计器 forward/backward 在这个 batch 规模上每步要跑两次（阶段 1 + 阶段 2），训练吞吐预估**比 DAN（134K）慢 2.5–3 倍**。
- 推理时 CLUB 和 IB 可以关掉，但双分支 + 原型匹配仍然在，推理约 **1.5–2× baseline**。可以接受，但要在论文里诚实报告 wall-clock time。

**风险**：`predict()` 里 `max_effective_batch=32768` 是 `torch_base.py` 已有的保护，NYC 下 batch_size 会被压到 40 左右，推理 IO 开销大。建议提前测一下 NYC→CHI 的端到端推理时长。

---

## 4. 接口兼容性

`TorchModel` 基类已经把 per_node 的 reshape 处理好了，`_net_forward` 收到 `(B', H, 1)` 返回 `(B', P, 1)`。但 STCross 的 `_net_forward` 需要的不只是 X——它还需要 **B_normal[input_window]** 和 **B_normal[future_window]**（查表），以及方差归一化的 std。

这意味着 STCross **不能简单继承默认 `fit()`**，因为默认 `fit()` 通过 `build_dataloader` 只把 `(X, Y, mask)` 喂进去，B_normal 查表需要时间戳信息（weekday/hour），而当前 DataLoader 流水线**没有透传时间戳**。两个出路：

1. **重写 `fit()`**：在 STCross 内部把 B_normal 预计算好，按 sample index 查表注入 batch。复杂度中等，但要改 `build_dataloader` 或自己包一层 Dataset。
2. **在 `_net_forward` 内部用预计算的 B_normal tensor**：通过 `self._b_normal_lookup` 缓存，根据 batch 在 dataset 中的索引查。但 `forward()` 是无状态的，拿不到 sample index——这条路走不通。

**结论**：必须自定义 `fit()`，并且要在 `Evaluator.train()` 里把 source 城市的 `(train_flow, anomaly_mask, timestamps)` 一并准备好传给 model。`evaluator.py` 当前只传 `(train_X, train_Y, val_X, val_Y)`，**接口需要扩展**，或者 STCross 自己在 `fit()` 开头重新 `load_flow + load_anomaly_mask` 一次（冗余但可行，因为 cache 机制存在）。

推理时同理：`Evaluator.evaluate()` 调 `model.predict(test_X)`，但 STCross 还需要 target 城市的 B_normal。**这是接口缺口**，需要 STCross 在 `predict()` 内部根据 target 城市重新算 B_normal，或由 Evaluator 注入。建议改 Evaluator 注入路径，保持 model 无状态。

---

## 5. B_normal 实现细节

- **同时段正常-only 均值**：逻辑正确，但"同时段"指 `(weekday, hour)` 还是 `(weekday, hour, minute)`？freq_min 决定，30min 频率下 hour 粒度只有 48 个 slot，足够；15min 频率下要按 96 个 slot。文档没明确，**建议按数据频率定义 slot 数**。
- **稀疏 slot 回退**：< 5 样本回退到"该节点全局正常均值"——可行但粗暴。更好的做法是回退到 `(hour,)` 维度（去掉 weekday），再不够才回退到全局。三级回退更鲁棒。
- **目标城市 B_normal**：文档声称"训练期流量和异常标签都已有"，对照 `evaluator.py` 确实如此（`load_flow` + `load_anomaly_mask` 在 evaluate 中被调用）。但 **B_normal_target 必须在 `predict()` 之前算好**，而 `predict()` 当前签名只收 `X: np.ndarray`。这是真实的工程缺口——需要在 `Evaluator` 层把 B_normal 算好后注入 model（比如 `model.set_target_baseline(B)`），或在 `predict()` 内重新加载。前者更干净。
- **节假日定义一致性**：source 和 target 的 holiday/weather mask 必须用同一套日历+天气规则生成，否则偏差分布城市间不一致，迁移假设直接破。这点文档没强调，**实测时务必核对**。

---

## 6. 评测公平性

**这是最严重的实验设计问题**。STCross 用 B_normal = "排除异常的同时段均值"，而 B_normal 的计算**依赖 anomaly_mask**（holiday/weather 标签）。也就是说：

- STCross 在**训练期**用异常标签筛掉异常样本算基线 → 基线更纯净；
- baseline（DAN/STSSDL/ProtoTS）**不使用异常标签**做任何预处理。

这构成**信息不对称的不公平优势**。即使 STCross 在推理时不直接看 future 的异常标签（future B_normal 也是用 train 期统计），但 train 期用了异常标签这件事本身，就让 STCross 拥有了 baseline 没有的先验。

**控制方案（任选其一，建议都做）**：

1. **公平对照**：给所有 baseline 也提供"normal-only HA"作为输入预处理器（即让 baseline 也在偏差空间操作），看 STCross 的增益是否还来自架构本身。
2. **消融**：STCross 用 **full HA（含异常）** 作为 B_normal 的变体，与 normal-only B_normal 对比，证明纯度带来的贡献是可量化的、而非占大头。
3. **标签消融**：STCross 用随机 mask 替代 anomaly_mask 算 B_normal，看性能是否塌方——如果塌方，说明信息泄露严重；如果不塌方，说明 B_normal 设计鲁棒。

论文里**必须**有一节专门讨论这个公平性问题，否则审稿人一击致命。

---

## 7. 消融实验设计评估

§10 列的消融基本覆盖了组件，但**缺几个关键对照**：

- **缺 B_normal 公平性消融**：见 §6，必须有"normal-only vs full HA vs instance mean"作为基线的对比，且要给 baseline 也喂 normal-only HA。
- **缺 Decoder_adj 角色消融**：单独看"只有 Δ̂_type" vs "只有 Δ̂_adj" vs "两者相加"，验证 Decoder_adj 没有喧宾夺主。
- **缺 K（原型数）敏感性**：K=2/4/8/16，验证原型不是越多越好也不是越少越好。
- **缺 target 适配方式消融**：zero-shot vs few-shot 微调 Decoder_adj（文档 §4.8 提了但没列入消融）。
- **缺 std 归一化粒度消融**：per-sample std vs per-node global std vs sliding-window std——这是 §2 数值稳定性的关键。
- **缺 source/target 方向对称性**：NYC→CHI vs CHI→NYC，验证"大城市→小城市"和反向都 work。

§10 已有的"偏差空间 vs 绝对值空间""解耦 vs 无解耦""归一化+IB vs GRL"是核心，**优先级最高**，但上述几项也建议补上。

---

## 8. 潜在工程问题（Top 5）

1. **方差归一化数值崩溃**：`std(Δ) → 0` 时 Z_a 爆炸。建议：全局/滑动 std + 小 std 的窗口 mask 掉 Z_a 损失，加 ε=1e-4 量级而非 1e-8。

2. **CLUB 两阶段优化震荡**：`q_φ` 训太快、编码器跟不上。建议：编码器每 3–5 步才更新 1 步 `q_φ`，CLUB loss 梯度裁剪 clip_norm=1.0，并监控 MI 估计值的滑动均值而非瞬时值。

3. **Decoder_adj 角色漂移**：被 L_pred 压成兜底项，Z_a 失效。建议：给 Δ̂_adj 加范数惩罚（`||Δ̂_adj||²`）或显式约束 `|Δ̂_adj| < |Δ̂_type|`，逼 Decoder_adj 只做"小幅度城市调整"。

4. **接口缺口：B_normal 注入路径**：`TorchModel.fit/predict` 不透传时间戳和 anomaly_mask，STCross 必须自定义 `fit()` 并在内部重新加载城市数据，或扩展 `Evaluator` 注入 B_normal。后者更干净，但要改框架公共接口。

5. **per_node effective batch 过大**：NYC 下 `B*N` 可达数万，CLUB 估计器双阶段 forward 在显存上吃紧。建议：CLUB 估计器用更小的隐层（32 维已足够），并对 `B*N` 做 sub-batch 切分（只对 CLUB 这一支切分，主路径不切）。

---

## 总评

设计有想法、有理论、有 baseline 验证支撑，**作为 PhD 阶段的创新点提案是合格的**。但从"能跑出 paper 数字"到"能在 4 个城市对 6 个 baseline 稳定胜出"，中间至少还有 2–3 个月的工程调试，主要瓶颈不在写代码、而在 CLUB 训练稳定性和 B_normal 公平性论证。

**优先级建议**：

1. 先打通"无 CLUB、无 IB"的简化版（双分支 + 原型 + B_normal），验证主路径和接口；
2. 解决 §6 的公平性，**这是论文能否发出的红线**；
3. 再逐项加 IB、CLUB，每加一项跑完整消融；
4. 最后补 §7 列出的对照。

祝顺利。模型不难，难的是把"看起来都加了"变成"每一项都真的起了作用"。
