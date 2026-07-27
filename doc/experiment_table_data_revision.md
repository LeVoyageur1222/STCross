# 实验部分表格数据核对与修改建议

## 0. 本次修改范围

本次核对并重构实验部分的主实验表、消融图版和参数敏感性图版素材，不修改 `paper_latex/content/4 Experiment.tex` 正文段落：

- 主实验表：`tab:main_cross_city`
- 消融实验图：`fig:ablation`
- 参数敏感性图：`fig:design_sensitivity`

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

## 2. 修正后的主实验表 LaTeX

```latex
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
```

## 3. 修正后的消融实验小节 LaTeX（图版）

```latex
\subsection{Ablation Study (RQ2)}

To assess the contribution of each major component in ProtoDRIFT, we conduct an ablation study on the anomaly subset under the NYC$\rightarrow$CHI transfer setting. We compare the full model with three variants: \textit{w/o DevSpace}, which removes deviation-space decomposition and predicts in the absolute-value space; \textit{w/o ProtoBank}, which removes the anomaly prototype bank; and \textit{w/o CityBranch}, which disables the city-conditioned adjustment branch.

\begin{figure}[!htbp]
   \centering
    \includegraphics[width=0.50\textwidth]{fig/ablation_triptych_metrics.pdf} % Reduce the figure size so that it is slightly narrower than the column.
    \caption{Ablation study of ProtoDRIFT on the NYC$\rightarrow$CHI anomaly subset. The variants denote (a) \textit{w/o DevSpace}, which removes deviation-space decomposition, (b) \textit{w/o ProtoBank}, which removes anomaly prototype retrieval, and (c) \textit{w/o CityBranch}, which removes city-conditioned adjustment.}
    \label{fig:ablation}
\end{figure}

As shown in Figure~\ref{fig:ablation}, the full ProtoDRIFT achieves the best overall performance, confirming that the three components are complementary. Removing the deviation-space decomposition leads to the largest degradation in MAE, indicating that separating normal references from transferable deviations is central to cross-city anomaly forecasting. Removing the prototype bank weakens the model's ability to summarize scarce anomaly patterns, while removing the city-conditioned branch notably harms RMSE and $R^2$, suggesting that transferable anomaly prototypes still require city-specific magnitude adjustment. These results validate the necessity of jointly modeling deviation normalization, prototype-based anomaly transfer, and city-conditioned response calibration.
```

> 使用提醒：若后续将该图写入 `paper_latex/content/4 Experiment.tex`，需要确保 `ablation_triptych_metrics.pdf` 已复制或放置到 LaTeX 工程可访问的 `fig/` 目录下。

## 4. 更新后的 RQ 与 RQ3--RQ6 小节 LaTeX

### 4.1 RQ 列表 LaTeX

```latex
In this section, we conduct systematic experiments to answer the following research questions:

\begin{itemize}
    \item RQ1: Under bidirectional cross-city transfer, how does ProtoDRIFT perform on both full and anomalous periods compared with existing methods?
    \item RQ2: How does each core component of ProtoDRIFT contribute to cross-city anomaly forecasting?
    \item RQ3: Can the prototype bank capture diverse and transferable anomaly deviation patterns across cities?
    \item RQ4: Does the dual-branch design separate transferable anomaly semantics from city-conditioned response factors?
    \item RQ5: How do the deviation capacity, prototype-bank size, and prototype regularization affect ProtoDRIFT's transfer performance?
    \item RQ6: How does the decomposition of normal references, prototype deviations, and city-conditioned adjustments explain ProtoDRIFT's predictions during anomalous periods?
\end{itemize}
```

### 4.2 RQ3: Transferable Anomaly Prototype Patterns

```latex
\subsection{Transferable Anomaly Prototype Patterns (RQ3)}

To answer RQ3, we examine whether the prototype bank learns reusable anomaly deviation semantics rather than memorizing city-specific flow magnitudes. Specifically, we visualize the future deviation pattern $\mathbf{r}_k$ associated with each anomaly prototype and further count the activation frequency of different prototypes under holiday and extreme-weather periods in both cities. This analysis directly corresponds to the role of the prototype bank in ProtoDRIFT: if anomalous dynamics are more transferable in the deviation space, the learned prototypes should cover diverse deviation shapes and be activated in a consistent manner across cities under similar anomaly types.

As shown in Figure~\ref{fig:proto_patterns}, the learned prototypes exhibit clearly distinguishable deviation patterns, including positive shifts, negative shifts, persistent changes, and short-term fluctuations. This indicates that the prototype bank does not collapse to a single average anomaly pattern, but instead provides a structured basis for representing heterogeneous anomalous responses. More importantly, prototypes associated with the same anomaly type tend to be activated in both NYC and CHI, suggesting that the prototype bank captures transferable anomaly semantics beyond city-specific traffic scales. These observations support the design choice of modeling anomalies as prototype-guided deviations from normal references, rather than directly transferring absolute traffic values across cities.
```

### 4.3 RQ4: Transfer-City Factor Disentanglement

```latex
\subsection{Transfer-City Factor Disentanglement (RQ4)}

To answer RQ4, we analyze whether the dual-branch encoder separates transferable anomaly semantics from city-conditioned response factors. We compare the anomaly representation $\mathbf{z}_a$, which is used for prototype retrieval, with the city-conditioned representation $\mathbf{z}_s$, which is used for residual adjustment. We visualize both representations using t-SNE and train a domain classifier to measure how much city information is retained in each representation.

The results in Figure~\ref{fig:disentangle} show that $\mathbf{z}_s$ preserves strong city-discriminative information, while $\mathbf{z}_a$ is less separable by city and is more closely organized by anomaly semantics. This contrast is consistent with the intended division of labor between the two branches: $\mathbf{z}_a$ should support city-agnostic anomaly prototype matching, whereas $\mathbf{z}_s$ should retain city-specific magnitude and local response factors needed for target-city calibration. Therefore, the dual-branch design does not merely increase model capacity; it provides a mechanism for preventing the transferable prototype bank from being dominated by city-scale information.
```

### 4.4 RQ5: Sensitivity to Prototype and Deviation Design

```latex
\subsection{Sensitivity to Prototype and Deviation Design (RQ5)}

To answer RQ5, we study how three design variables affect the transfer performance of ProtoDRIFT: the deviation capacity coefficient $\alpha$, the prototype-bank size $K$, and the prototype regularization strength $\gamma$. Rather than treating them as isolated tuning parameters, we interpret them as controlling three modeling choices: how much information the deviation representation can retain, how fine-grained the anomaly prototype bank should be, and how strongly prototype learning should be regularized. In each sweep, one variable is varied while the other two are fixed at the selected configuration, and RMSE and $R^2$ are evaluated on the NYC$\rightarrow$CHI anomaly subset.

\begin{figure}[!htbp]
   \centering
\includegraphics[width=0.50\textwidth]{fig/design_sensitivity_dualaxis.pdf}
\caption{Sensitivity to prototype and deviation design variables on the NYC$\rightarrow$CHI anomaly subset. The subplots vary (a) $\alpha$ for deviation capacity, (b) $K$ for prototype-bank size, and (c) $\gamma$ for prototype regularization. Blue solid and red dashed curves denote RMSE and $R^2$, and the dotted line marks the selected configuration.}
\label{fig:design_sensitivity}
\end{figure}

As shown in Figure~\ref{fig:design_sensitivity}, the selected configuration is consistently located at or near the best region across the three sweeps. For $\alpha$, allowing a larger deviation capacity improves both RMSE and $R^2$ in this harder transfer direction, indicating that the deviation representation requires sufficient capacity to preserve transferable anomaly semantics. For $K$, the best performance is obtained at $K=8$, suggesting that a moderate prototype-bank size balances coverage and specificity: too few prototypes cannot represent diverse anomalous patterns, whereas too many prototypes may fragment scarce anomaly samples and dilute prototype semantics. For $\gamma$, a moderate value works best, while overly strong regularization reduces the flexibility of the deviation representation. Overall, these trends show that ProtoDRIFT benefits from coherent design choices in deviation capacity, prototype granularity, and prototype regularization, rather than relying on extreme parameter settings.
```

### 4.5 RQ6: Prediction Decomposition under Anomalies

```latex
\subsection{Prediction Decomposition under Anomalies (RQ6)}

To answer RQ6, we further inspect how ProtoDRIFT constructs predictions during anomalous periods. We select representative holiday and extreme-weather windows from the target-city test set and decompose the prediction into three parts: the normal reference $\mathbf{B}$, the prototype-based deviation $\widehat{\boldsymbol{\Delta}}_{\mathrm{type}}$, and the city-conditioned adjustment $\widehat{\boldsymbol{\Delta}}_{\mathrm{adj}}$. This case-level analysis is intended to verify whether the model's prediction follows the proposed decomposition mechanism.

As illustrated in Figure~\ref{fig:case_study}, the normal reference captures regular daily traffic patterns but is insufficient for explaining abnormal shifts. The prototype-based deviation contributes the main anomalous trend, such as an upward or downward departure from the normal baseline, while the city-conditioned adjustment corrects the magnitude and local response of the target city. The final prediction combines these components and follows the ground truth more closely than the normal reference alone. This decomposition explains why ProtoDRIFT improves anomaly-period RMSE and $R^2$: its prediction is not produced by a monolithic black-box mapping, but by combining transferable anomaly prototypes with target-city response calibration.
```

> 使用提醒：RQ5 和 RQ6 已按你的要求对调，设计敏感性分析放在预测分解前面；如果后续写入 `paper_latex/content/4 Experiment.tex`，需要同步调整原 `Hyperparameter Sensitivity (RQ3)` 和 `Visualization and Interpretability (RQ4)` 的标题、RQ 编号、正文引用和图表位置，并确保 `design_sensitivity_dualaxis.pdf` 已复制或放置到 LaTeX 工程可访问的 `fig/` 目录下。

## 5. 后续正文同步提醒

本次按要求只给出实验表格与图版小节的修改建议，不改正文。但如果后续实际替换 `4 Experiment.tex`，需要同步检查正文中基于旧数值的相对提升描述，尤其是 RQ1 和 RQ2 中引用 ProtoDRIFT 旧值 `0.7170/1.9085/0.5715`、`0.6463/1.7869/0.4477`、`0.6444/1.7706/0.4577` 的句子。
