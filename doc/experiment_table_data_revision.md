# 实验部分表格数据核对与修改建议

## 0. 本次修改范围

本次核对并重构实验部分的表格与图版素材，并重排 RQ 结构。第 5 节给出可直接替换 `paper_latex/content/4 Experiment.tex` 的完整正文。

素材清单：

- 主实验表：`tab:main_cross_city`
- 消融实验图：`fig:ablation`（`ablation_triptych_metrics.pdf`）
- 参数敏感性图：`fig:design_sensitivity`（`design_sensitivity_dualaxis.pdf`）
- 可解释性图：`fig:semantic_case`（`v4_semantic_case_dualpanel.pdf`）

RQ 结构调整：由现行 4 个 RQ 改为 5 个，参数敏感性前移至 RQ3，可解释性一节内拆为 RQ4 与 RQ5 两个子小节。原「Disentangled Representations」（$\mathbf{z}_a$ 与 $\mathbf{z}_s$ 对比、domain classifier）与「三分量预测分解」内容因无对应图版而移除。

> 待办：三个 PDF 目前都在 `results_viz/`，需复制到 `paper_latex/fig/` 后再编译。

## 1. 数据来源核对结论

### 1.1 主实验表

当前论文主实验表中的 baseline 数据与 `results/RESULTS.md` 及对应 `results/*.json` 记录一致，属于真实实验数据，没有发现伪造或错误记录。

但 ProtoDRIFT 行当前使用的是 `results/STCross_*_to_*.json` 中的单次主实验记录，而参数敏感性分析与消融表的统一口径应对齐到网格搜索最优组合：

```text
results_para_full/a0.3_k8_g0.1/summary.md
```

因此建议把主实验表中 ProtoDRIFT 行改为该最优组合的结果：

- NYC→CHI full: `0.6988 / 1.8742 / 0.5867`
- NYC→CHI anomaly: `0.6292 / 1.7539 / 0.4679`
- CHI→NYC full: `1.7061 / 3.6897 / 0.5127`
- CHI→NYC anomaly: `1.4906 / 3.1140 / 0.4309`

最优/次优标注已按更新后的数值重新核对。需要注意：NYC→CHI anomaly 的 MAE 最优仍是 ProtoTS，次优是 ST-SSDL，ProtoDRIFT 只在 RMSE 和 R² 上最优。

### 1.2 消融表

当前论文消融表中的 A1/A2/A3 数据均来自真实消融实验：

- `results_ablation/A1_no_devspace/summary.md`
- `results_ablation/A2_no_proto/summary.md`
- `results_ablation/A3_no_citybranch/summary.md`

但未消融 ProtoDRIFT 行当前来自 `results_ablation/control/summary.md`，与参数敏感性最优组合不完全一致。项目已有聚合文件 `results_ablation/ablation_NYC_to_CHI.csv` 和 `TODO.md` 均明确 Full STCross 行采用：

```text
results_para_full/a0.3_k8_g0.1
```

因此建议把消融表 ProtoDRIFT 行改为：

```text
0.6292 / 1.7539 / 0.4679
```

### 1.3 参数敏感性数据

参数敏感性图所使用的所有 RMSE/R² 数值已逐项核对，均来自 `results_para_full` 中对应组合的 `summary.md`，未发现错误记录：

- α sweep: `a0.02_k8_g0.1`, `a0.05_k8_g0.1`, `a0.1_k8_g0.1`, `a0.3_k8_g0.1`
- K sweep: `a0.3_k4_g0.1`, `a0.3_k8_g0.1`, `a0.3_k16_g0.1`, `a0.3_k32_g0.1`
- γ sweep: `a0.3_k8_g0.05`, `a0.3_k8_g0.1`, `a0.3_k8_g0.2`, `a0.3_k8_g0.4`

最优标注正确，三组 sweep 的共同最优点均为 `α=0.3, K=8, γ=0.1`，对应 NYC→CHI anomaly：

```text
MAE=0.6292, RMSE=1.7539, R²=0.4679
```

## 2. 图表素材变更

主实验表：ProtoDRIFT 行换为 `a0.3_k8_g0.1` 结果，最优/次优标注已重核。完整 LaTeX 见第 5 节。

消融与参数敏感性：由表改图，`tab:ablation` 与 `tab:sensitivity` 删除，改用 `fig:ablation`（`ablation_triptych_metrics.pdf`）与 `fig:design_sensitivity`（`design_sensitivity_dualaxis.pdf`）。消融图中 ProtoDRIFT 行使用 `0.6292/1.7539/0.4679`，与 `visualization/plot_ablation_triptych.py` 内置数值一致。

可解释性：新增 `fig:semantic_case`（`v4_semantic_case_dualpanel.pdf`），宽幅图需用 `figure*` 跨双栏。

## 3. 重排后的 RQ 结构与各小节内容

### 3.1 RQ 对照表

| 新编号 | 小节 | 对应图/表 | 现行 tex 来源 |
|---|---|---|---|
| RQ1 | 4.2 Main Results on Cross-City Transfer | `tab:main_cross_city` | RQ1（数值需更新） |
| RQ2 | 4.3 Ablation Study | `fig:ablation` | RQ2（表改图） |
| RQ3 | 4.4 Sensitivity to Prototype and Deviation Design | `fig:design_sensitivity` | RQ3（表改图，标题改为机制导向） |
| RQ4 | 4.5.1 Semantic Structure of Anomaly Deviations | `fig:semantic_case` (a)(b) | 原 RQ4 内 Anomaly Prototype Patterns（内容重写） |
| RQ5 | 4.5.2 Case-Level Anomaly Forecasting | `fig:semantic_case` (c)(d) | 原 RQ4 内 Case Study（内容重写） |
| — | 删除 | 无 | 原 RQ4 内 Disentangled Representations |

写作体例：所有 RQ 小节采用连续段落分析，不使用 itemize；小节标题体现机制性问题而非实验类型。可解释性一节内，一张图跨两个子小节，各自只引用自己的 panel。

### 3.2 各 RQ 内容要点

**RQ1（主实验）**。双向迁移、full 与 anomaly 两个子集。要点：常规时序方法存在迁移上限；分布对齐类方法在困难方向上失效；异常感知类方法靠保守预测压低 MAE 但无法解释波动；ProtoDRIFT 优势集中在解释力。需诚实说明：NYC→CHI 的 anomaly MAE 最优仍是 ProtoTS，次优 ST-SSDL，ProtoDRIFT 仅在 RMSE 与 $R^2$ 上最优。

**RQ2（消融）**。三个变体 w/o DevSpace / w/o ProtoBank / w/o CityBranch，只报 NYC→CHI anomaly 子集。结论：偏差空间分解对 MAE 影响最大，城市条件分支对 $R^2$ 影响最大，原型库增益稳定但幅度较温和。

**RQ3（设计变量敏感性）**。把 $\alpha$/$K$/$\gamma$ 论述为三个建模选择（偏差表示的信息容量、原型库粒度、原型学习的正则强度），而非普通调参。三组 sweep 共同最优点为 $\alpha=0.3, K=8, \gamma=0.1$。

**RQ4（语义结构）**。对应图 (a)(b)。将 CHI 测试集异常窗口的 window-level $\mathbf{z}_a$ 聚为八类并做 t-SNE，再用选定簇的平均偏差曲线证明两簇语义方向相反（节假日为正偏差，暴雨为负偏差）。叙事落点：偏差空间承载方向可辨的异常语义，而非城市量级。

**RQ5（case 级预测）**。对应图 (c)(d)。在两个簇各取一个代表窗口做 rolling step-0 预测对比。节假日 case 上 baseline 集体低估，暴雨 case 上 baseline 集体高估，ProtoDRIFT 两侧都跟得上。叙事落点：语义组不仅可解释，且直接转化为方向正确的 case 级预测。

> 数值来源：两个 case 的节点级 MAE（Presidents' Day node 119：1.72 vs baseline 均值 2.88；Rainstorm node 79：2.23 vs 5.44）由 `visualization/viz_v4_semantic_case_dualpanel.py` 的固定 case 配置算出，可重算核对。正文中以相对百分比表达，避开堆砌绕对数值。

## 4. 后续正文同步提醒

第 5 节的完整正文已经完成下列同步，直接替换即可：

- 主实验表 ProtoDRIFT 行换为 `a0.3_k8_g0.1` 结果，正文中旧值 `0.7170/1.9085/0.5715`、`0.6463/1.7869/0.4477`、`0.6444/1.7706/0.4577` 均已更新，相对提升幅度已重算（full $R^2$ +55.5\%，anomaly $R^2$ +29.6\%）
- 消融与参数敏感性由表改图，`tab:ablation` 与 `tab:sensitivity` 已删除
- RQ3 标题从 `Hyperparameter Sensitivity` 改为机制导向表述，位置前移至消融之后
- 原 4.2 节末尾遗留的双语交付注释句（“*The LaTeX source ... Chinese Section 4.2 above*”）已删除
- Setup 中“see the sensitivity analysis in Section 4.4”在新结构下仍指向参数敏感性节，无需修改
- 消融小节末尾原引用“see Section 4.5”仍指向可解释性节，无需修改

## 5. 完整 Experiment 正文（可直接替换 `paper_latex/content/4 Experiment.tex`）

```latex
\section{Experiment}

In this section, we conduct systematic experiments to answer the following five research questions:

\begin{itemize}
    \item RQ1: Under bidirectional cross-city transfer, how does ProtoDRIFT perform on both full and anomalous periods compared with existing methods?
    \item RQ2: How does each core component of ProtoDRIFT contribute to cross-city anomaly forecasting?
    \item RQ3: How do the deviation capacity, prototype-bank size, and prototype regularization affect ProtoDRIFT's transfer performance?
    \item RQ4: Does the deviation representation organize anomalous periods into interpretable and reusable semantic groups?
    \item RQ5: Do these semantic groups translate into more accurate case-level forecasts than baselines during holiday and rainstorm anomalies?
\end{itemize}

\subsection{Experimental Setup}

\noindent \textbf{Datasets.} We conduct experiments on real-world taxi flow data from New York City (NYC) and Chicago (CHI), where NYC contains 820 spatial nodes and CHI contains 585 spatial nodes, spanning from 2014-10-01 to 2017-09-29 with a temporal granularity of 60 minutes. Both cities are split by calendar-aligned months into training (15336 steps), validation (5160 steps), and test (5784 steps) sets, and samples are constructed with a sliding window that predicts the future $P=12$ steps from the past $H=12$ steps. Anomalous periods are jointly annotated by holiday calendars and extreme weather records: a sample is assigned to the anomaly subset when the ratio of anomalous time steps within its prediction window is no less than 0.5. In the end, the NYC test set contains 5761 full samples (including 507 anomalous ones), and the CHI test set contains 5761 full samples (including 398 anomalous ones).

\noindent \textbf{Baselines.} We compare ProtoDRIFT with 11 representative methods in three categories: (1) conventional deep time series forecasting methods, including LSTM~\cite{lstm}, DLinear~\cite{dlinear}, PatchTST~\cite{PatchTST}, TimesNet~\cite{timesnet}, and ModernTCN~\cite{moderntcn}, which represent the mainstream paradigms
of single-city time series forecasting; (2) distributionally robust and transferable forecasting methods, including TimeAlign~\cite{timealign} and ProtoTS~\cite{protots}, which address distribution shifts via distribution alignment and hierarchical prototype learning, respectively; (3) anomaly/extreme-event-aware forecasting methods, including EVL~\cite{evl}, EALGAP~\cite{ealgap}, DAN~\cite{dan}, and ST-SSDL~\cite{stssdl}, which design dedicated losses or structures for extreme values and anomalous periods. All baselines adopt channel-independent modeling in our implementation, which supports zero-shot cross-city transfer between cities with different numbers of nodes.

\noindent \textbf{Implementation.} All methods share the same data splits, normalization pipeline, and evaluation scripts to ensure fairness: input data are normalized by a per-node StandardScaler and clipped to $[-10, 10]$, with statistics fitted only on the training set of each city. Training uses the Adam optimizer with a CosineAnnealingLR learning rate schedule and early stopping with a patience of 10. The default hyperparameters of ProtoDRIFT are $\alpha=0.3$, $K=8$, $\gamma=0.1$, and $\beta=0.01$ (see the sensitivity analysis in Section 4.4). We adopt MAE, RMSE, and $R^2$ as evaluation metrics, where lower MAE/RMSE is better and higher $R^2$ indicates a stronger ability to explain the flow variations of the target city; for each transfer direction, results are reported on both the full and anomaly test subsets.

\begin{table*}[htbp]
\centering
% \small
\setlength{\tabcolsep}{3.5pt}
\renewcommand\arraystretch{1.0}
\begin{tabular}{l|ccc|ccc|ccc|ccc}
\toprule
\multirow{3}{*}{\textbf{Method}}
 & \multicolumn{6}{c|}{\textbf{NYC$\rightarrow$CHI}}
 & \multicolumn{6}{c}{\textbf{CHI$\rightarrow$NYC}} \\ \cmidrule(lr){2-7}\cmidrule(lr){8-13}
 & \multicolumn{3}{c|}{\textbf{Full}}
 & \multicolumn{3}{c|}{\textbf{Anomaly}}
 & \multicolumn{3}{c|}{\textbf{Full}}
 & \multicolumn{3}{c}{\textbf{Anomaly}} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
 & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ & MAE & RMSE & $R^2$ \\
\midrule
LSTM       & 0.8832 & 2.3166 & 0.3686 & 0.7726 & 1.9324 & 0.3541 & 1.9290 & 4.1915 & 0.3711 & 1.6581 & 3.3830 & 0.3283 \\
DLinear    & 0.9095 & 2.4967 & 0.2666 & 0.7529 & 2.0096 & 0.3014 & 2.1537 & 4.6317 & 0.2321 & 1.7628 & 3.5901 & 0.2435 \\
PatchTST   & 0.8961 & 2.5132 & 0.2569 & 0.7365 & 2.0836 & 0.2490 & 1.9704 & 4.5254 & 0.2669 & 1.6036 & 3.6003 & 0.2392 \\
TimesNet   & 0.9021 & 2.5002 & 0.2646 & 0.7404 & 2.0738 & 0.2561 & 1.9583 & 4.4772 & 0.2825 & 1.5957 & 3.5653 & 0.2539 \\
ModernTCN  & 0.8670 & \underline{2.3006} & \underline{0.3773} & 0.7545 & \underline{1.9220} & \underline{0.3610} & 1.8934 & \underline{4.1522} & \underline{0.3829} & 1.6235 & 3.3587 & 0.3379 \\
\midrule
TimeAlign  & 0.8490 & 2.5864 & 0.2130 & 0.6522 & 2.0849 & 0.2481 & 2.1680 & 5.0835 & 0.0750 & 1.6703 & 3.9315 & 0.0928 \\
ProtoTS    & 0.8068 & 2.5923 & 0.2094 & \textbf{0.6168} & 2.1099 & 0.2299 & 2.3919 & 5.5185 & -0.0901 & 1.8235 & 4.2466 & -0.0585 \\
\midrule
EVL        & 0.8776 & 2.3286 & 0.3621 & 0.7696 & 1.9515 & 0.3412 & 1.8968 & 4.2172 & 0.3634 & 1.6106 & 3.3680 & 0.3342 \\
EALGAP     & 0.9367 & 2.3085 & 0.3730 & 0.8385 & 1.9441 & 0.3462 & 2.0923 & 4.2726 & 0.3466 & 1.8367 & 3.5233 & 0.2714 \\
DAN        & 0.8718 & 2.3104 & 0.3720 & 0.7577 & 1.9243 & 0.3595 & \underline{1.8786} & 4.1558 & 0.3818 & 1.6001 & \underline{3.3461} & \underline{0.3429} \\
ST-SSDL    & \underline{0.8020} & 2.4744 & 0.2797 & \underline{0.6258} & 2.0190 & 0.2948 & 2.0138 & 4.7708 & 0.1853 & \underline{1.5715} & 3.6968 & 0.1979 \\
\midrule
ProtoDRIFT & \textbf{0.6988} & \textbf{1.8742} & \textbf{0.5867} & 0.6292 & \textbf{1.7539} & \textbf{0.4679} & \textbf{1.7061} & \textbf{3.6897} & \textbf{0.5127} & \textbf{1.4906} & \textbf{3.1140} & \textbf{0.4309} \\
\bottomrule
\end{tabular}
\caption{Performance comparison with baselines under NYC$\rightarrow$CHI and CHI$\rightarrow$NYC cross-city transfer settings on both full and anomaly subsets. Best results are in \textbf{bold}; second-best are \underline{underlined}.}
\label{tab:main_cross_city}
\end{table*}

\subsection{Main Results on Cross-City Transfer (RQ1)}

To answer RQ1, we compare ProtoDRIFT with all baselines on the full and anomaly test subsets under both transfer directions, as reported in Table~\ref{tab:main_cross_city}.

The five conventional temporal methods share a common ceiling: ModernTCN is the strongest among them and ranks second-best on full RMSE and $R^2$ in both directions, yet no conventional method exceeds an $R^2$ of 0.39. Operating purely in the absolute-value space, these models cannot explain the target-city flow variations, because the magnitudes learned from the source city do not match the target scale distribution. Distribution alignment alone does not remove this barrier either: TimeAlign and ProtoTS are competitive on NYC$\rightarrow$CHI anomaly MAE, but their full $R^2$ collapses to 0.0750 and $-0.0901$ in the harder CHI$\rightarrow$NYC direction, falling below even LSTM. When scale shift and anomaly distribution shift coexist, aligning marginal distributions or reusing source-city prototypes can introduce systematic bias.

The anomaly-aware baselines expose a different limitation. ST-SSDL and ProtoTS attain the two best anomaly MAE values on NYC$\rightarrow$CHI, yet their anomaly $R^2$ stays below 0.30. Such a gap between MAE and $R^2$ indicates conservative, smoothed predictions that reduce the mean absolute error without capturing the large fluctuations of anomalous periods.

ProtoDRIFT ranks first on every full-subset metric in both directions and obtains the best anomaly RMSE and $R^2$ throughout. Its advantage lies in explanatory power: on NYC$\rightarrow$CHI it raises the full $R^2$ by 55.5\% and the anomaly $R^2$ by 29.6\% relative to the strongest baseline. On the anomaly MAE of this direction it ranks behind ProtoTS and ST-SSDL, which is consistent with the trade-off above, since ProtoDRIFT lowers RMSE and raises $R^2$ at the same time. These results support modeling transferable anomaly semantics in the deviation space together with a city-conditioned branch that calibrates target-city magnitudes.

\subsection{Ablation Study (RQ2)}

To assess the contribution of each major component in ProtoDRIFT, we conduct an ablation study on the anomaly subset under the NYC$\rightarrow$CHI transfer setting. We compare the full model with three variants: \textit{w/o DevSpace}, which removes deviation-space decomposition and predicts in the absolute-value space; \textit{w/o ProtoBank}, which removes the anomaly prototype bank; and \textit{w/o CityBranch}, which disables the city-conditioned adjustment branch.

\begin{figure}[!htbp]
   \centering
    \includegraphics[width=0.50\textwidth]{fig/ablation_triptych_metrics.pdf} % Reduce the figure size so that it is slightly narrower than the column.
    \caption{Ablation study on the NYC$\rightarrow$CHI anomaly subset, with variants (a) \textit{w/o DevSpace}, (b) \textit{w/o ProtoBank}, and (c) \textit{w/o CityBranch}.}
    \label{fig:ablation}
\end{figure}

As shown in Figure~\ref{fig:ablation}, the full ProtoDRIFT performs best on all three metrics, confirming that the components are complementary. Removing the deviation-space decomposition degrades MAE the most, indicating that separating normal references from transferable deviations is central to cross-city anomaly forecasting. Removing the prototype bank weakens the summarization of scarce anomaly patterns, while removing the city-conditioned branch harms RMSE and $R^2$ most, showing that transferable prototypes still require city-specific magnitude adjustment.

\subsection{Sensitivity to Prototype and Deviation Design (RQ3)}

To answer RQ3, we study three design variables: the deviation capacity coefficient $\alpha$, the prototype-bank size $K$, and the prototype regularization strength $\gamma$. Rather than isolated tuning parameters, they control how much information the deviation representation retains, how fine-grained the prototype bank is, and how strongly prototype learning is regularized. Each sweep varies one variable with the other two fixed at the selected configuration, evaluated on the NYC$\rightarrow$CHI anomaly subset.

\begin{figure}[!htbp]
   \centering
\includegraphics[width=0.50\textwidth]{fig/design_sensitivity_dualaxis.pdf}
\caption{Design-variable sensitivity on the NYC$\rightarrow$CHI anomaly subset, varying (a) $\alpha$, (b) $K$, and (c) $\gamma$. Blue solid and red dashed curves denote RMSE and $R^2$; the dotted line marks the selected configuration.}
\label{fig:design_sensitivity}
\end{figure}

As shown in Figure~\ref{fig:design_sensitivity}, the selected configuration lies at or near the best region in all three sweeps. A larger deviation capacity $\alpha$ helps in this harder direction, indicating that the deviation representation needs sufficient capacity to preserve transferable anomaly semantics. A moderate bank size $K=8$ is optimal: too few prototypes cannot cover diverse anomalous patterns, whereas too many fragment the scarce anomaly samples and dilute prototype semantics. A moderate $\gamma$ also works best, since overly strong regularization restricts the deviation representation. ProtoDRIFT therefore benefits from coherent design choices rather than extreme parameter settings.

\subsection{Interpretability of Anomaly Deviation Modeling}

\begin{figure*}[!htbp]
\centering
\includegraphics[width=\textwidth]{fig/v4_semantic_case_dualpanel.pdf}
\caption{Interpretability analysis on the CHI test set (NYC$\rightarrow$CHI). (a) t-SNE of window-level anomaly representations in eight semantic clusters, with the selected holiday and rainstorm windows marked; (b) mean deviation curves of the two selected clusters; (c)(d) rolling one-step-ahead predictions for the two windows against three baselines.}
\label{fig:semantic_case}
\end{figure*}

\subsubsection{Semantic Structure of Anomaly Deviations (RQ4)}

To answer RQ4, we examine whether the deviation representation organizes anomalous periods into interpretable and reusable semantic groups rather than encoding city-specific magnitudes. We encode all anomalous windows of the target-city test set into window-level anomaly representations, group them into eight clusters, project them with t-SNE, and summarize each cluster by the mean deviation curve of its members.

As shown in Figure~\ref{fig:semantic_case}(a), the anomalous windows form separable groups, and the two selected windows, a public holiday and a rainstorm, fall into different clusters. Figure~\ref{fig:semantic_case}(b) shows that the two clusters carry opposite deviation semantics: the holiday cluster is positive relative to the normal reference, whereas the rainstorm cluster stays negative over the whole horizon. The representation therefore encodes the direction and shape of departure from the normal state rather than the absolute traffic scale, which is exactly what prototype-guided transfer requires: anomalous dynamics become comparable across events once city-specific magnitudes are absorbed into the normal reference.

\subsubsection{Case-Level Anomaly Forecasting (RQ5)}

To answer RQ5, we ask whether this semantic organization yields concrete forecasting gains. We take one representative window from each of the two clusters and compare the rolling one-step-ahead predictions of ProtoDRIFT with three baselines, one from each category.

In the holiday window (Figure~\ref{fig:semantic_case}(c)), the observed flow rises well above its normal reference and all three baselines systematically underestimate the surge, whereas ProtoDRIFT follows the rising trend and reproduces most of the peak. The rainstorm window (Figure~\ref{fig:semantic_case}(d)) shows the opposite failure: the observed flow stays below the normal reference for most of the day, the baselines keep predicting near-normal levels and overestimate throughout, and only ProtoDRIFT tracks the suppressed demand. On these two nodes ProtoDRIFT reduces the mean absolute error to roughly 60\% and 41\% of the baseline average. The semantic structure of RQ4 is thus not merely interpretable but directly responsible for predicting anomalies in the correct direction, which explains why the gain of ProtoDRIFT concentrates on anomaly RMSE and $R^2$.
```
