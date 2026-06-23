# Evaluation Frameworks in Traffic Prediction Papers (ST01–ST13)

## 1. Input/Output Settings

### Historical Timesteps (Input) & Prediction Horizon (Output)

| Paper | Model/Dataset | Input Steps | Output Steps | Time Granularity | Notes |
|-------|--------------|-------------|--------------|------------------|-------|
| ST01 (TGDL) | NYCBike / NYCTaxi | Single-step | Single-step | 60min / 30min | Inflow/outflow prediction |
| ST01 (TGDL) | PEMSD4 / PEMSD8 | Multi-step | 1–4 steps | 30 min | Up to 2 hours ahead |
| ST02 (STBP) | PEMS-Stream / CA-Stream / AIR-Stream | **12** | **12** | 5 min / 1 hr | Standard 1h→1h setting |
| ST03 (DoFlow) | Hydropower | 180 context | 30 pred | — | Domain-specific |
| ST03 (DoFlow) | Cancer Treatment | 55 context | 5 pred | — | Domain-specific |
| ST03 (DoFlow) | Synthetic DAGs | 90 context | 30 pred | — | T=120, τ=90 |
| ST04 (UrbanGraph) | UMC4/12 | T_hist steps | T_pred steps | Hourly | Not explicitly numbered; 12-hour span |
| ST05 (TimeAlign) | ETT/Weather/Elec/Traffic/Solar | {96,192,336,512,720} | {96,192,336,720} | Varies | Long-term forecasting benchmark |
| ST06 (ProtoTS) | LOF | 384 | 96 | 15 min | 96h look-back → 24h forecast |
| ST06 (ProtoTS) | EPF | 168 | 24 | 1 hour | 7-day look-back → 1-day forecast |
| ST07 (HyperD) | PEMS03/04/07/08 | **12** | **12** | 5 min | **1 hour → 1 hour** |
| ST08 (GenCast) | PEMS-Bay/PEMS07/08/METR-LA/Melbourne | 2 hours | 2 hours | 5–15 min | Past 2h → future 2h |
| ST09 (TraffiDent) | TraffiDent | Varies | Varies | 5 min | Dataset paper; post-incident window = 24 steps (2h) |
| ST10 (IGSTGNN) | Alameda/Contra Costa/Orange | **12** | **12** | 5 min | **1 hour → 1 hour** |
| ST11 (ConFormer) | Tokyo/San Diego/Bay Area/PEMS03-08 | Standard | Standard | 10–15 min | Follows standard benchmarks |
| ST12 (CAMEF) | Financial assets | Event-aligned window | 35/70/140 steps | 5 min | Half-day / 1-day / 2-day horizons |
| ST13 (STEVE) | NYCTaxi/NYCBike1/NYCBike2/BJTaxi | Standard | Standard | 30–60 min | Follows standard city-flow benchmarks |

### Most Common Setting
> **The dominant convention across traffic prediction papers is: input = 12 timesteps, output = 12 timesteps at 5-minute granularity (i.e., 1 hour of history → 1 hour of future).** This is used by ST02, ST07, ST08, ST10 and implicitly by ST11 and ST13. Long-term TSF papers (ST05, ST06) use much longer windows (96–720 steps).

---

## 2. Train/Val/Test Split Ratios

| Paper | Split Ratio | Notes |
|-------|------------|-------|
| ST01 (TGDL) | Not explicitly stated | Uses standard PEMS/NYC splits |
| ST02 (STBP) | **6 : 2 : 2** | All three stream datasets |
| ST04 (UrbanGraph) | **7 : 2 : 1** | Test set = spatially unseen blocks |
| ST05 (TimeAlign) | Standard benchmarks | Follows established ETT/Weather/etc. splits |
| ST06 (ProtoTS) | Not explicitly stated | Standard benchmarks + new LOF/EPF |
| ST07 (HyperD) | **6 : 2 : 2** | PEMS03/04/07/08; 5-run average |
| ST08 (GenCast) | Spatial **4 : 1 : 5**; Temporal **70% / 30%** | Unique: spatial split ensures unobserved regions in test; 4 random spatial partitions averaged |
| ST09 (TraffiDent) | Not a single fixed split | Dataset paper; downstream tasks define their own splits |
| ST10 (IGSTGNN) | **70 : 15 : 15** | Three XTraffic sub-datasets |
| ST11 (ConFormer) | Standard benchmarks | Follows established PEMS/Tokyo/CA splits |
| ST12 (CAMEF) | **6 : 2 : 2** | Financial event dataset |
| ST13 (STEVE) | **7 : 1 : 2** | NYC/BJ taxi & bike datasets |

### Most Common Splits
> - **6:2:2** is the most frequent (ST02, ST07, ST12).
> - **7:1:2** or **7:2:1** is also common (ST04, ST13).
> - **70:15:15** appears in ST10.
> - ST08 uses a unique **spatial** split (4:1:5) specifically designed for evaluating generalization to unobserved regions, plus a temporal 70/30 split.

---

## 3. Evaluation Metrics

### Core Metrics and Their Formulas

| Metric | Formula | Used By |
|--------|---------|---------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | ST01, ST02, ST04, ST05, ST06, ST07, ST08, ST09, ST10, ST11, ST12, ST13 |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | ST01, ST03, ST04, ST05, ST08, ST09, ST10, ST11 |
| **MAPE** | $\frac{1}{n}\sum_{i=1}^{n} \frac{\|y_i - \hat{y}_i\|}{\|y_i\|} \times 100\%$ | ST01, ST07, ST08, ST10, ST11 |
| **MSE** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | ST03, ST05, ST06, ST12 |
| **R²** | $1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$ | ST04, ST08 |
| **PCC (Pearson Correlation)** | $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$ | ST04 (as *r* for real-world validation) |
| **NRMSE** | RMSE normalized by data range or mean | ST03 (cancer treatment) |
| **CRPS** | Continuous Ranked Probability Score | ST03 (probabilistic evaluation) |
| **MMD** | Maximum Mean Discrepancy | ST03 (distribution matching) |
| **Log-density** | $\log p_\theta(\hat{x}_{\tau+1:T} \| \hat{H}_\tau)$ | ST03 (anomaly detection via CNF) |
| **HF-MSE** | High-Frequency MSE (frequency-domain filtered) | ST05 (high-frequency component error) |
| **FLOPs / Training Time / Inference Speed** | Computational efficiency metrics | ST01, ST04, ST07, ST11 |

### Summary of Metric Usage Patterns
- **MAE + RMSE + MAPE** is the "standard trio" for traffic prediction (used by the majority of papers: ST01, ST07, ST08, ST10, ST11).
- **MSE** is preferred over RMSE in general time series forecasting papers (ST05, ST06, ST12).
- **R²** is used when the focus is on *explained variance* rather than absolute error, particularly in physics-informed / scientific prediction contexts (ST04 UrbanGraph, ST08 GenCast).
- **PCC** appears only in ST04 (UrbanGraph), used as *r* for validating against real-world sensor data and cross-validation with HiGTS dataset. It is not a primary training metric but a validation/correlation metric.
- **Probabilistic metrics** (CRPS, MMD, log-density) are exclusive to ST03 (DoFlow), which uses generative flow models.
- **HF-MSE** is a novel metric introduced in ST05 (TimeAlign) to specifically evaluate high-frequency component fidelity.

---

## 4. Cross-City / Transfer Learning Evaluation

### Papers That Evaluate Cross-City or Cross-Domain Transfer

| Paper | Transfer Type | Setup Details |
|-------|--------------|---------------|
| **ST08 (GenCast)** | **Cross-region (unobserved areas within same city/network)** | Evaluates prediction at spatially contiguous regions with **zero traffic observations**. Uses spatial 4:1:5 split ensuring test regions are geographically separated from training regions. Also tests **cross-domain transfer** on NREL solar energy dataset (traffic model → solar generation). Tests with varying unobserved ratios (0.2–0.8). |
| **ST04 (UrbanGraph)** | **Cross-site validation** | Trained on ENVI-met simulated Singapore data; validated against **real-world NUS campus sensor network** (micro-scale, r > 0.73) and **HiGTS city-scale dataset** (r = 0.842). Not strictly cross-city transfer but cross-data-source validation. |
| **ST13 (STEVE)** | **Distribution shift robustness (temporal + spatial)** | Designs explicit **Temporal Distribution Shift (TDS)** experiments: trains with workday:holiday ≈ 5:2 ratio, then tests with shifted ratios (1:0 and 0:1). Also designs **Spatial Distribution Shift (SDS)** using k-means clustering of traffic entities into different flow-level groups. While not cross-city per se, it evaluates generalization across distributional regimes that mimic cross-scenario transfer. |
| **ST02 (STBP)** | **Continual learning across expanding graph topology** | Not cross-city, but evaluates adaptation to **new nodes** (up to +254% growth in CA-Stream). Tests few-shot prediction (10% training data) in later incremental periods, simulating cold-start for new locations. |

### Key Observation
> **No paper performs a clean "train on City A, test on City B" cross-city transfer experiment.** The closest is ST08's cross-domain test (traffic → solar) and its spatial unobserved-region evaluation. ST13's distribution shift evaluation (workday-only vs holiday-only train/test) is the most relevant proxy for evaluating transfer between normal and anomalous regimes. Most papers evaluate on multiple datasets independently rather than transferring between them.

---

## 5. Normal vs. Anomaly Evaluation

### How Specific Papers Split Evaluation Between Normal and Anomalous Periods

#### ST09 (TraffiDent) — Post-Incident vs. General Performance
- **Core contribution**: Explicitly measures the performance gap between **General (normal)** and **Post-incident (anomalous)** traffic prediction.
- **Method**: Selects test samples where an incident occurred within the 5-minute prediction window. Runs all SOTA models on both the full test set ("General") and the incident-only subset ("Post-incident").
- **Finding**: All models show **significant performance degradation** under post-incident conditions. This validates the need for specialized anomaly evaluation.
- **Evaluation metrics**: MAE, RMSE, MAPE reported separately for General and Post-incident subsets.

#### ST10 (IGSTGNN) — Incident-Aware Modeling & Implicit Normal/Anomaly Distinction
- Does **not** report separate normal/anomaly metrics in tables.
- Instead, the entire model architecture is designed around the premise that **normal and anomalous patterns require different modeling**: the ICSF module handles spatial event impact, and the TID module handles temporal decay.
- Ablation studies implicitly demonstrate the gap: removing incident-aware modules causes ~4% MAE increase, showing how much of the overall performance depends on correctly handling anomalous periods.
- The dataset construction itself (Alameda, Contra Costa, Orange) includes aligned incident records, enabling potential normal/anomaly split analysis.

#### ST11 (ConFormer) — Accident/Regulation Scenario-Specific Evaluation
- **Explicitly reports accident-scenario and regulation-scenario performance** alongside overall results.
- Ablation Table 4 shows separate rows for **"Accident" and "Regulation" scenario MAE**, demonstrating that:
  - Removing accident data: Tokyo accident-scenario MAE rises from 8.98 → 10.97 (+22%)
  - Removing regulation data: Tokyo regulation-scenario MAE rises from 9.18 → 10.64 (+16%)
  - Replacing GLN with standard LN: accident-scenario MAE rises from 8.98 → 9.58
- t-SNE visualization confirms GLN creates **separable latent representations** for normal vs. anomalous traffic patterns.
- This is the most explicit normal-vs-anomaly breakdown among all papers.

#### ST13 (STEVE) — Distribution Shift as Proxy for Normal/Anomaly
- **Temporal Distribution Shift (TDS)**: Trains with mixed workday/holiday data (≈5:2 ratio), then evaluates under two extreme shifts:
  - **Workday-only** (ratio → 1:0): tests model's ability to handle pure normal patterns
  - **Holiday-only** (ratio → 0:1): tests model's ability to handle pure anomalous patterns
- **Spatial Distribution Shift (SDS)**: Clusters traffic entities by flow level, evaluates transfer between clusters.
- STEVE outperforms all baselines specifically under **holiday distribution shift**, directly validating its confounder-learning approach for anomalous scenarios.
- This is the most rigorous experimental design for evaluating normal-vs-anomaly generalization.

### Summary of Normal/Anomaly Evaluation Approaches

| Approach | Papers | Description |
|----------|--------|-------------|
| **Separate metric reporting** | ST09, ST11 | Report MAE/RMSE/MAPE separately for normal and anomalous subsets |
| **Architecture-level separation** | ST10 | Design dedicated modules for anomalous patterns; ablation reveals contribution |
| **Distribution shift experiments** | ST13 | Manipulate train/test composition ratios to stress-test generalization |
| **Latent space separation** | ST11 | Visualize (t-SNE) whether model internally distinguishes normal vs. anomalous |
| **No explicit split** | ST01–ST08, ST12 | Evaluate on mixed data without separating normal/anomalous periods |

### Key Takeaway
> Only **four papers** (ST09, ST10, ST11, ST13) explicitly address the normal-vs-anomaly evaluation gap. Among these:
> - **ST09** provides the clearest evidence that existing models fail under anomalies.
> - **ST11** provides the most granular scenario-specific metrics (accident/regulation/overall).
> - **ST13** provides the most principled experimental design (controlled distribution shift).
> - **ST10** provides architectural solutions but less explicit evaluation separation.
>
> The remaining nine papers evaluate on mixed normal/anomalous data, which means their reported metrics may mask poor performance during anomalous periods — exactly the problem ST09 highlights.
