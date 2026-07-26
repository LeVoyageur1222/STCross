# ProtoDRIFT Preliminaries 与 Methodology 重构审阅报告

## 0. 本轮重构范围

本轮只关注两个文件：

- `paper_latex/content/2 Preliminaries.tex`
- `paper_latex/content/3 The Proposed Method.tex`

其他章节暂时不提出修改源码，不再讨论 Introduction、Experiment、Related Work、Conclusion 或 Appendix 的替换文本。此前关于 RQ3 只分析 $\alpha$、$K$、$\gamma$ 的要求仍然保留：方法主文应避免把 $\beta$、$\zeta$、`margin`、`std_threshold`、`warmup_epochs`、`mic_clip` 等实现常数写成与三项核心超参数同等级的设计参数。

本轮额外参考了 `writing/ref paper/3 The Proposed Method_v2.tex` 的组织方式。该参考章节的可借鉴点不是具体内容，而是写法：先用 overview 交代整体范式和关键组件关系，再按模块展开；每个模块先说明为什么需要，再给关键机制和少量必要公式；若模块内部有阶段性机制，用加粗段首组织，而不是机械罗列很多孤立公式。

## 1. 总体审阅结论

当前 `Preliminaries` 已经承担了观测定义、异常上下文定义和跨城问题定义三件事，不宜再新增一段 normal reference 定义。上一版报告建议加入 Definition 3 会让该节更长，也会把方法特有的正常基线构造提前放进问题定义，容易削弱 Preliminaries 的简洁性。更合理的做法是压缩现有定义，保留任务必要符号，把 normal reference 放回 Methodology overview 中说明。

当前 `Methodology` 的核心思想是正确的，但写法有三类问题：

1. 小节标题风格不统一。`Transferable Anomaly Prototype Design` 与 `Mutual-Information-Constrained Disentanglement` 偏“设计说明”口吻，建议改为具体组件名称，例如 `Deviation-Normalized Prototype Bank` 和 `MI-Constrained Dual-Branch Encoder`。
2. 公式数量与粒度需要控制。像 $\mathbf z_a=\mathrm{AEnc}(\cdot)$、$\widehat{\Delta}_{adj}=\mathrm{CAD}(\mathbf z_s)$ 这类很短的映射不必单独占一行公式，可合并在正文或组合公式中。
3. 方法叙述不能像模块清单。应围绕“为什么偏差空间更可迁移 → 如何用原型概括偏差模式 → 为什么需要城市条件分支 → 如何训练并跨城推理”形成递进，而不是 NBL/AEnc/APB/CEnc/CAD/MIC 逐个列模块。

## 2. 参考方法章节的风格提炼

`writing/ref paper/3 The Proposed Method_v2.tex` 的方法部分有以下结构特征，适合借鉴到 ProtoDRIFT：

- **Overview 先行。** 开头用一段集中说明整体架构、关键组件和迁移范式，避免读者进入公式前不知道模块关系。
- **模块标题是组件名。** 例如 `Heterogeneous Subgraph Sampler`、`Patch Flow Encoder`、`Multi-view Pattern-bank Subgraph Encoder`，标题本身是模块类型，不强调 `Design`。
- **每节内部按逻辑阶段组织。** 当一个模块内部有两个阶段，用 `\textbf{...}` 段首提示，例如 message passing 和 adapter，而不是继续细分很多 subsection。
- **公式服务于关键机制。** 只对边权、频域变换、注意力融合、总目标等核心计算给展示公式；其他简单映射放入句子。
- **公式后立即解释符号。** 每个公式紧跟 where 句解释新符号，避免定义悬空。
- **训练目标收束而非再讲模块。** Optimization 节负责把前文机制统一到 objective，不重复长篇动机。

对应到 ProtoDRIFT，建议 `Methodology` 使用如下结构：

```text
Methodology
Overview
3.1 Deviation-Normalized Prototype Bank
    Normal-reference deviation space
    Prototype-based anomaly deviation
3.2 MI-Constrained Dual-Branch Encoder
    Dual-branch representation encoding
    Redundancy control and city-conditioned adjustment
3.3 Objective and Optimization
3.4 Cross-City Inference
```

其中 3.1 和 3.2 都使用具体组件名；内部用加粗段首分出相对独立的子机制，满足“有比较分离的子模块就分开叙述”的要求，但不会把章节切得过碎。

## 3. Preliminaries 审阅与修改建议

### 3.1 当前问题

当前 `Preliminaries` 的主要问题不是缺内容，而是已经略长。它包括：

- 城市流量观测定义。
- 异常上下文和未来窗口异常样本定义。
- 跨城市异常预测问题定义。

这三项足够支撑 Methodology。不要再加入正常基线、偏差空间、NBL 回退机制等方法专属内容。否则 Preliminaries 会从“任务形式化”变成“方法预告”，篇幅和职责都会变重。

### 3.2 推荐策略

- 保留 $\mathbf X^{(c)}$、$\mathbf Y$、$H$、$P$、$\mathbf m^{(c)}$、$a_t^{(c)}$、$c_s/c_t$、$\mathcal D_s/\mathcal D_t$、$f_\theta$。
- 删除冗长解释句，尤其是对异常样本含义的重复解释。
- 不新增 normal reference 定义。
- 把 `\mathbf{m}^{(c)} \in {0,1}^{T_c}` 修正为 `\mathbf{m}^{(c)} \in \{0,1\}^{T_c}`。
- `\mathcal{D}_{s}` 的样本中当前包含历史异常 mask，但代码的核心模型并不直接把 mask 作为输入；如果保留 mask，是为了定义异常上下文和评估子集，需要在文字中避免暗示模型依赖 mask 作为预测输入。建议 problem formulation 中训练函数仍写成 $f_\theta(\mathbf X)$，而将 $\mathbf m$ 表述为 split/evaluation context。

## 4. Methodology 审阅与修改建议

### 4.1 超参数显著性控制

方法主文建议仅显式解释与 RQ3 分析范围一致的三项参数作用：

- $\alpha$：capacity regularization strength。
- $K$：prototype number。
- $\gamma$：prototype regularization strength。

以下参数应隐去或放入实现细节，不应在方法主公式中以独立系数出现：

- $\beta$：MIC 固定辅助权重。
- $\zeta$：city adjustment magnitude 固定辅助权重。
- `margin`：prototype separation 固定 margin。
- `std_threshold`：归一化数值稳定项。
- `warmup_epochs`、`mic_clip`：训练稳定策略。

因此总目标建议写成：

```latex
\mathcal L = \mathcal L_{\mathrm{pred}} + \alpha\mathcal L_{\mathrm{cap}} + \gamma\mathcal L_{\mathrm{proto}} + \mathcal L_{\mathrm{dec}},
```

其中 $\mathcal L_{\mathrm{dec}}$ 吸收互信息去冗余与城市调整幅度约束，避免把 $\beta$、$\zeta$ 变成论文主超参。正文中可说明 auxiliary weights are fixed across all experiments，但不在方法公式中展开。

### 4.2 符号定义需要补齐但不要增加过多公式

建议补齐以下符号：

- $\mathbf B^{(c)}$：城市正常参照，来自非异常历史同星期/小时统计。
- $\boldsymbol\Delta^{(c)}$：相对正常参照的输入偏差。
- $s_t^{(c)}$：输入窗口级偏差尺度，对应代码中 `delta.std(dim=1)+std_threshold`，不要写成城市级 $\sigma_\Delta^{(c)}$。
- $\mathbf e_k$、$\mathbf r_k$：原型嵌入和原型未来偏差模式。
- $d_k$：$\mathbf z_a$ 到第 $k$ 个原型的平方欧氏距离。
- $w_k$：prototype assignment weight。
- $\mathbf z_a$、$\mathbf z_s$：异常语义表示与城市条件表示。
- $\widehat I(\mathbf z_a;\mathbf z_s)$：可优化互信息估计，不是严格不变性保证。
- $\mathcal L_{\mathrm{cap}}$、$\mathcal L_{\mathrm{proto}}$、$\mathcal L_{\mathrm{dec}}$：需要给出含义。

但不建议为每个短映射都单独给公式。例如：

- `AEnc` 和 `CEnc` 可以在一句话里写：`The anomaly and city-conditioned branches encode \tilde{\Delta} and \Delta into \mathbf z_a and \mathbf z_s, respectively.`
- `CAD` 可以在正文写：`CAD maps \mathbf z_s to \widehat{\Delta}_{adj}`。
- 预测公式、prototype attention、训练目标是需要展示的核心公式。

### 4.3 篇幅分布建议

- `Deviation-Normalized Prototype Bank`：建议拆成两个相对独立的段首：`Normal-reference deviation modeling` 和 `Prototype encoding and retrieval`。前者解释 $\Delta$ 与 $\widetilde\Delta$ 如何构造，后者明确 $\mathbf z_a$ 由 AEnc 从 $\widetilde\Delta$ 得到，并说明该 $\mathbf z_a$ 会同时用于 prototype retrieval 和后续 MI-constrained decomposition。
- `MI-Constrained Dual-Branch Encoder`：需要继续明确双分支架构，但不要重新定义另一个 $\mathbf z_a$。建议写成“the same anomaly representation $\mathbf z_a$”，再补充 raw deviation 经 CEnc 得到 $\mathbf z_s$，CAD 由 $\mathbf z_s$ 生成城市调整项。该节建议用 `\textbf{Dual-branch representation split.}` 和 `\textbf{Information-constrained adjustment.}` 两个段首组织。
- `Objective and Optimization`：短一些，给最终目标和三项核心超参说明。
- `Cross-City Inference`：保持短。跨城推理逻辑简单，不需要长篇重复训练模块；一段文字加预测公式即可。

## 5. 建议替换后的 `2 Preliminaries.tex`

```latex
\section{Preliminaries}

\textbf{Urban Flow Observations.} Given a set of cities $\mathcal{C}$, each city $c\in\mathcal{C}$ contains $N_c$ spatial units and a historical flow sequence $\mathbf{X}^{(c)}=[\mathbf{x}^{(c)}_1,\ldots,\mathbf{x}^{(c)}_{T_c}]\in\mathbb{R}^{T_c\times N_c}$, where $\mathbf{x}^{(c)}_t\in\mathbb{R}^{N_c}$ denotes the flow state at time step $t$. With a history length $H$ and a prediction length $P$, a forecasting sample is denoted as $(\mathbf{X}^{(c)}_{t-H+1:t},\mathbf{Y}^{(c)}_{t+1:t+P})$, where $\mathbf{Y}^{(c)}_{t+1:t+P}=\mathbf{X}^{(c)}_{t+1:t+P}$.

\textbf{Anomalous Flow Context.} Let $\mathbf{m}^{(c)}\in\{0,1\}^{T_c}$ denote the anomaly context sequence of city $c$, where $m^{(c)}_t=1$ indicates that time step $t$ is affected by holidays, extreme weather, large-scale events, or other external disturbances. For a forecasting origin $t$, the future-window anomaly indicator is
\begin{equation}
    a^{(c)}_t=\mathbb{I}\left(\sum_{\tau=t+1}^{t+P}m^{(c)}_\tau>0\right).
\end{equation}
Samples with $a^{(c)}_t=1$ form the anomaly subset, while all samples together form the full evaluation subset.

\textbf{Problem Formulation.} Given a source city $c_s$ and a target city $c_t$ with $c_s\neq c_t$, the two cities may differ in node numbers, flow scales, periodic patterns, and anomaly distributions. Let $\mathcal{D}_s=\{(\mathbf{X}^{(c_s)}_{t-H+1:t},\mathbf{Y}^{(c_s)}_{t+1:t+P})\}_{t\in\mathcal{T}_s}$ denote the source-city training samples and $\mathcal{D}_t=\{(\mathbf{X}^{(c_t)}_{t-H+1:t},\mathbf{Y}^{(c_t)}_{t+1:t+P})\}_{t\in\mathcal{T}_t}$ denote the target-city test samples. The goal is to learn a forecasting function on the source city and directly apply it to the target city:
\begin{equation}
\begin{gathered}
    \theta^*=\arg\min_\theta\sum_{t\in\mathcal{T}_s}\ell\left(f_\theta(\mathbf{X}^{(c_s)}_{t-H+1:t}),\mathbf{Y}^{(c_s)}_{t+1:t+P}\right),\\
    \widehat{\mathbf{Y}}^{(c_t)}_{t+1:t+P}=f_{\theta^*}(\mathbf{X}^{(c_t)}_{t-H+1:t}).
\end{gathered}
\end{equation}
Under this setting, the key challenge is to transfer forecasting knowledge to target-city anomalous periods when cross-city distribution discrepancies and anomaly scarcity coexist.
```

### 5.1 这版 Preliminaries 的变化

- 没有新增 normal reference 段落，避免变长。
- 将 Definition 标题压缩为三个加粗段首，篇幅更紧。
- 保留异常 mask 只用于 anomaly subset 定义，不再暗示模型必须以 mask 为输入。
- 公式数控制为 2 个，比原文更集中。

## 6. 建议替换后的 `3 The Proposed Method.tex`

```latex
\section{Methodology}

This section presents ProtoDRIFT, a deviation-referenced prototype framework for cross-city anomalous urban flow forecasting. As illustrated in the overall architecture, ProtoDRIFT shifts transfer from the raw-flow space to the deviation space relative to each city's normal reference. The motivation is that raw urban flows entangle long-term commuting regularities, city-specific scales, and anomaly-induced perturbations, whereas deviations from a normal reference more directly describe how an event changes the normal operating state. ProtoDRIFT is built around two components: a \textit{Deviation-Normalized Prototype Bank}, which encodes normalized deviations and retrieves recurring anomaly patterns from a compact prototype bank, and an \textit{MI-Constrained Dual-Branch Encoder}, which complements the shared anomaly pattern with a city-conditioned response adjustment while discouraging representational redundancy. The final prediction combines the target city's normal reference, the prototype-based anomaly deviation, and the city-conditioned adjustment:
\begin{equation}
\begin{aligned}
    \widehat{\mathbf{Y}}^{(c)}_{t+1:t+P}
    &=\mathbf{B}^{(c)}_{t+1:t+P}
    +\widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{type}}
    +\widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{adj}}.
\end{aligned}
\end{equation}
Here $\mathbf{B}^{(c)}$ denotes the city-specific normal reference, $\widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{type}}$ is the transferable anomaly deviation retrieved from the prototype bank, and $\widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{adj}}$ is a city-conditioned residual adjustment. The two deviation terms share the same forecasting target but play different roles: the prototype term captures anomaly semantics intended to transfer across cities, while the adjustment term absorbs city-conditioned response differences that remain after normal-reference subtraction.

\subsection{Deviation-Normalized Prototype Bank}

The first component addresses the scale mismatch and anomaly scarcity that make direct raw-flow transfer unreliable. Instead of asking the prototype bank to model absolute traffic volumes, ProtoDRIFT first removes each city's regular operating state and then performs prototype retrieval in a normalized deviation space. This design makes the prototype bank focus on anomaly-induced shapes rather than city-specific baselines.

\textbf{Normal-reference deviation modeling.} For each city, ProtoDRIFT estimates a normal reference $\mathbf{B}^{(c)}$ from non-anomalous historical observations. In implementation, this reference is computed by aggregating normal samples with the same day-of-week and hour-of-day attributes, with fallback to coarser normal averages when a slot is sparse. Given an input window, the model removes the normal component and normalizes the residual by its window-level scale:
\begin{equation}
\begin{gathered}
    \boldsymbol{\Delta}^{(c)}_{t-H+1:t}
    =\mathbf{X}^{(c)}_{t-H+1:t}-\mathbf{B}^{(c)}_{t-H+1:t}, \\
    \widetilde{\boldsymbol{\Delta}}^{(c)}_{t-H+1:t}
    =\frac{\boldsymbol{\Delta}^{(c)}_{t-H+1:t}}{s^{(c)}_t}.
\end{gathered}
\end{equation}
where $s^{(c)}_t$ is the standard deviation of the input-window deviation with a small fixed stability constant. This window-level normalization follows the implementation and avoids introducing a separate city-level deviation statistic. After this step, the model keeps both views of the residual signal: $\widetilde{\boldsymbol{\Delta}}^{(c)}$ is used to learn transferable anomaly semantics, while $\boldsymbol{\Delta}^{(c)}$ is retained for the city-conditioned branch introduced later.

\textbf{Prototype encoding and retrieval.} The normalized deviation is encoded by the anomaly encoder into an anomaly representation $\mathbf{z}_a$. This representation is the same $\mathbf{z}_a$ used by the subsequent MI-constrained decomposition; it is not redefined in the next subsection. In the implementation, the anomaly encoder is a variational LSTM, which outputs the latent code together with its distribution parameters and is regularized by a KL-based capacity term $\mathcal{L}_{\mathrm{cap}}$. The encoding step can be written as
\begin{equation}
\begin{gathered}
    \mathbf{z}_a=\mathrm{AEnc}\left(\widetilde{\boldsymbol{\Delta}}^{(c)}_{t-H+1:t}\right), \\
    \mathcal{L}_{\mathrm{cap}}=\mathrm{KL}\left(q(\mathbf{z}_a\mid\widetilde{\boldsymbol{\Delta}}^{(c)}_{t-H+1:t})\,\|\,p(\mathbf{z}_a)\right).
\end{gathered}
\end{equation}
Here $\mathrm{AEnc}(\cdot)$ denotes the anomaly encoder, $q(\mathbf{z}_a\mid\cdot)$ is the variational posterior, and $p(\mathbf{z}_a)$ is a simple latent prior. The capacity term discourages $\mathbf{z}_a$ from carrying excessive city-specific details, making it more suitable for prototype matching.

On top of $\mathbf{z}_a$, ProtoDRIFT maintains an anomaly prototype bank $\mathcal{P}=\{(\mathbf{e}_k,\mathbf{r}_k)\}_{k=1}^{K}$, where $\mathbf{e}_k$ is the $k$-th prototype embedding and $\mathbf{r}_k\in\mathbb{R}^{P}$ is its future deviation pattern. The prototype bank uses soft retrieval rather than hard assignment, allowing an anomalous window to be represented as a mixture of several canonical deviation patterns:
\begin{equation}
\begin{gathered}
    d_k=\|\mathbf{z}_a-\mathbf{e}_k\|_2^2, \\
    w_k=\frac{\exp(-d_k)}{\sum_{j=1}^{K}\exp(-d_j)}, \\
    \widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{type}}=\sum_{k=1}^{K}w_k\mathbf{r}_k.
\end{gathered}
\end{equation}
Here $d_k$ is the squared Euclidean distance from $\mathbf{z}_a$ to prototype $k$, $w_k$ is the corresponding assignment weight, and $K$ controls the granularity of the prototype bank. A fixed-margin prototype regularizer $\mathcal{L}_{\mathrm{proto}}$ further encourages the selected prototype to be separated from nearby alternatives; its strength is controlled by $\gamma$.

\subsection{MI-Constrained Dual-Branch Encoder}

The prototype bank captures shared anomaly shapes, but prototype deviations alone cannot fully determine how a target city responds to an event. Even after normal-reference subtraction, deviations may still contain city-specific magnitude, local response delay, and event-coverage effects. ProtoDRIFT therefore complements the prototype path with a city-conditioned branch while constraining the two branches to avoid learning the same information.

\textbf{Dual-branch representation split.} The transferable branch reuses the anomaly representation $\mathbf{z}_a$ defined in the Deviation-Normalized Prototype Bank, where normalized deviations are encoded for prototype retrieval. In parallel, the city-conditioned branch takes the unnormalized deviation as input and encodes it into $\mathbf{z}_s$, preserving local response magnitude that may be suppressed by normalization. The city adjustment decoder then maps $\mathbf{z}_s$ to the residual adjustment:
\begin{equation}
\begin{gathered}
    \mathbf{z}_s=\mathrm{CEnc}\left(\boldsymbol{\Delta}^{(c)}_{t-H+1:t}\right), \\
    \widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{adj}}=\mathrm{CAD}(\mathbf{z}_s).
\end{gathered}
\end{equation}
Together, $\mathbf{z}_a$ and $\mathbf{z}_s$ form a dual-branch representation: $\mathbf{z}_a$ drives prototype-based transferable anomaly prediction, whereas $\mathbf{z}_s$ supports city-conditioned response adjustment. $\mathrm{CEnc}(\cdot)$ denotes the city-conditioned encoder, and $\mathrm{CAD}(\cdot)$ denotes the city adjustment decoder. This split lets the prototype bank focus on transferable anomaly types, while the city-conditioned branch handles response variations that are useful for prediction but less suitable as shared anomaly semantics.

\textbf{Information-constrained adjustment.} Without additional constraints, the city-conditioned branch could absorb most predictive information and weaken the role of the prototype bank. ProtoDRIFT therefore regularizes the dependence between $\mathbf{z}_a$ and $\mathbf{z}_s$ and constrains the magnitude of the adjustment output:
\begin{equation}
\begin{aligned}
    \mathcal{L}_{\mathrm{dec}}
    &=\widehat{I}(\mathbf{z}_a;\mathbf{z}_s)
    +\left\|\widehat{\boldsymbol{\Delta}}^{(c)}_{\mathrm{adj}}\right\|_2.
\end{aligned}
\end{equation}
The first term is an optimizable mutual-information estimate that reduces overlap between the anomaly and city-conditioned representations, and the second term prevents the city adjustment branch from overwhelming the transferable prototype deviation. This constraint should be interpreted as a practical redundancy-reduction mechanism rather than a strict proof of city invariance. Its auxiliary weights are fixed across experiments and are treated as implementation settings rather than independent factors in the method analysis.

\subsection{Objective and Optimization}

The forecasting loss is the mean squared error between the predicted and ground-truth future flows. Combining forecasting accuracy with representation capacity, prototype separation, and decomposition regularization, ProtoDRIFT optimizes
\begin{equation}
\begin{aligned}
    \mathcal{L}
    &=\mathcal{L}_{\mathrm{pred}}
    +\alpha\mathcal{L}_{\mathrm{cap}}
    +\gamma\mathcal{L}_{\mathrm{proto}}
    +\mathcal{L}_{\mathrm{dec}}.
\end{aligned}
\end{equation}
In this objective, $\mathcal{L}_{\mathrm{pred}}$ fits the final flow prediction, $\alpha$ weights the capacity regularization on the anomaly representation, $K$ determines the granularity of the prototype bank, and $\gamma$ weights the prototype regularizer. The remaining numerical constants are kept fixed to support stable optimization and are reported as implementation settings when necessary. This objective keeps the main forecasting path concise while still linking each regularization term to a specific modeling role.

\subsection{Cross-City Inference}

After training on a source city $c_s$, ProtoDRIFT directly transfers the learned anomaly encoder, prototype bank, city-conditioned encoder, and adjustment decoder to a target city $c_t$. The target city only needs its own normal reference to construct deviation inputs. The target prediction is
\begin{equation}
\begin{aligned}
    \widehat{\mathbf{Y}}^{(c_t)}_{t+1:t+P}
    &=\mathbf{B}^{(c_t)}_{t+1:t+P}
    +\widehat{\boldsymbol{\Delta}}^{(c_t)}_{\mathrm{type}}
    +\widehat{\boldsymbol{\Delta}}^{(c_t)}_{\mathrm{adj}}.
\end{aligned}
\end{equation}
This procedure does not require the source and target cities to share the same node set, node embeddings, or graph topology. The target-city adaptation mainly occurs through its normal reference and deviation representation, while the learned prototype bank and encoders reuse source-city anomaly knowledge in the deviation space.
```

## 7. 这版 Methodology 的变化说明

### 7.1 `Deviation-Normalized Prototype Bank` 被拆成两个内部段首

新版在该小节内部增加两个相对独立的段首：

- `Normal-reference deviation modeling`：解释正常参照、输入偏差、窗口级归一化，以及为什么同时保留 normalized deviation 和 raw deviation。
- `Prototype encoding and retrieval`：补足 prototype encoder 细节，明确 $\mathbf z_a$ 由 AEnc 处理 $\widetilde\Delta$ 得到，并由同一个 $\mathbf z_a$ 完成 prototype retrieval。

这样可以避免该节只讲 prototype bank 而没有解释 prototype representation 来源的问题。

### 7.2 `z_a` 的前后一致性被明确

新版明确说明：`Deviation-Normalized Prototype Bank` 中得到的 $\mathbf z_a$ 与 `MI-Constrained Dual-Branch Encoder` 中使用的 $\mathbf z_a$ 是同一个 anomaly representation。后者不再重复书写 $\mathbf z_a=\mathrm{AEnc}(\widetilde\Delta)$，而是用文字说明沿用上一节定义的 $\mathbf z_a$，避免读者误以为存在第二次定义或第二个 anomaly representation。

### 7.3 `MI-Constrained Dual-Branch Encoder` 架构更完整

新版保留两个段首：

- `Dual-branch representation split`：说明 $\mathbf z_a$ 沿用上一节定义并负责 prototype retrieval，$\mathbf z_s$ 由 raw deviation 经 CEnc 得到并保留城市响应幅度，CAD 再输出 $\widehat\Delta_{adj}$。
- `Information-constrained adjustment`：解释 MI 估计和 adjustment norm 如何防止两个分支冗余，以及为什么该约束是实用的 redundancy reduction 而非严格城市不变性证明。

### 7.4 篇幅适度扩充但保持结构集中

新版 Methodology 建议源码在不改变主线的情况下扩充了以下细节：

- overview 中增加两个 deviation terms 的角色区分。
- prototype 小节补充 AEnc、variational posterior、capacity term 和 soft retrieval 的解释。
- MI 小节补充 raw/normalized deviation 分工、city response magnitude、local response delay、event coverage 等动机。
- objective 小节补充每个损失项的建模角色。

扩充后的篇幅更接近正式方法节，不再像模块清单，但仍然避免把 `Cross-City Inference` 扩成新的复杂模块。

### 7.5 公式格式保持多行结构

建议版 Methodology 保留 8 个核心公式：

1. 总体预测分解。
2. 偏差和窗口级归一化。
3. AEnc 编码与 capacity regularization。
4. 原型距离、原型权重与原型偏差输出。
5. 城市条件编码与城市调整解码。
6. 解耦正则 $\mathcal{L}_{\mathrm{dec}}$。
7. 总训练目标。
8. 目标城市推理公式。

其中包含多个表达式的公式均使用 `gathered` 或 `aligned` 多行组织，避免单行并列公式过长导致排版问题。

### 7.6 超参数表述更学术化

新版不使用预设式的“关键/核心超参数”表述，而改为说明三者在目标函数和原型库中的具体作用：

- $\alpha$ weights the capacity regularization。
- $K$ determines the granularity of the prototype bank。
- $\gamma$ weights the prototype regularizer。

其他数值常数表述为 fixed implementation settings，避免在 Methodology 中制造额外敏感性分析义务。

### 7.7 符号定义更完整

新版 Methodology 明确解释：

- $\mathbf B^{(c)}$ 是 normal reference。
- $s_t^{(c)}$ 是窗口级偏差尺度，对齐代码实现。
- $\mathbf z_a$ 是由 AEnc 从 normalized deviation 得到的 anomaly representation。
- $\mathbf z_s$ 是由 CEnc 从 raw deviation 得到的 city-conditioned representation。
- $\mathbf e_k$ 和 $\mathbf r_k$ 分别是 prototype embedding 和 future deviation pattern。
- $d_k$ 是平方欧氏距离。
- $w_k$ 是原型权重。
- $\widehat I(\mathbf z_a;\mathbf z_s)$ 是可优化 MI 估计，不是严格不变性保证。
- $\mathcal L_{dec}$ 吸收 MI redundancy reduction 和 adjustment magnitude control。

## 8. 方法-代码一致性核查

| 论文写法 | 代码对应 | 是否一致 |
| --- | --- | --- |
| $\Delta=X-B$ | `_net_forward`: `delta = X - B_input` | 一致 |
| 窗口级尺度 $s_t$ | `_net_forward`: `delta.std(dim=1)+std_threshold` | 一致 |
| $z_a$ 由 normalized deviation 经 AEnc 得到，并被 prototype bank 与 MI 约束共用 | `_net_forward`: `z_a, mu, logvar = self.aenc(delta_norm)`；随后 `self.apb(z_a)` 与 `self.mic(..., c['z_a'])` | 一致 |
| prototype 权重来自平方距离 softmax | `AnomalyProtoBank.forward`: `dist` 与 `softmax(-dist)` | 一致 |
| $z_s$ 来自 raw deviation | `self.cenc(delta)` | 一致 |
| city adjustment 来自 $z_s$ | `self.cad(z_s)` | 一致 |
| 总目标突出 $\alpha,K,\gamma$ 的机制作用 | 代码仍有 beta/zeta，但报告建议吸收到 auxiliary loss 定义 | 写作层面对齐 RQ3 |
| Cross-city inference 短叙述 | `predict_with_baseline` 使用目标城市 `b_input/b_future` | 一致 |

## 9. 建议执行优先级

1. 先替换 `2 Preliminaries.tex`，压缩篇幅并移除方法特有内容。
2. 再替换 `3 The Proposed Method.tex`，尤其是 `Deviation-Normalized Prototype Bank` 的两个内部段首和 `MI-Constrained Dual-Branch Encoder` 中 $\mathbf z_a$ 的一致性说明。
3. 检查全文中是否还有 `$\beta$`、`$\zeta$` 在 Methodology 主公式中出现。
4. 检查 Methodology 中的多表达式公式是否都使用 `gathered` 或 `aligned` 多行组织。
5. 检查 `Cross-City Inference` 是否仍保持短段落，不要扩展为新的设计模块。
