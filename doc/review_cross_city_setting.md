# STCross Cross-City Flow Prediction Setting Review

**Reviewer**: Data Mining PhD Candidate  
**Date**: 2026-06-25  
**Scope**: Cross-city and same-city traffic flow prediction evaluation pipeline

---

## 1. Review Scope

The following 9 files were systematically reviewed:

| # | File | Role |
|---|------|------|
| 1 | `eval/config.py` | EvalConfig dataclass -- all configuration parameters |
| 2 | `eval/evaluator.py` | Evaluator -- full evaluation pipeline orchestration |
| 3 | `eval/reporter.py` | Reporter -- console output and JSON result saving |
| 4 | `experiments/run_all.py` | Batch runner -- multi-model x multi-setting matrix |
| 5 | `experiments/run_model.py` | Single-model runner -- CLI entry point |
| 6 | `data_preprocess/dataset.py` | Data loading, split, sliding window, anomaly label |
| 7 | `data_preprocess/normalizer.py` | PerNodeStandardScaler |
| 8 | `data_preprocess/torch_dataset.py` | PyTorch Dataset and DataLoader wrapper |
| 9 | `model/registry.py` | Model registry -- build functions for all models |

Supporting files also read for completeness:
- `model/base.py`, `model/torch_base.py` -- model base classes
- `model/ha.py`, `model/stgcn.py` -- representative model implementations
- `train/trainer.py` -- Trainer class
- `data_preprocess/build_holiday_mask.py`, `data_preprocess/build_weather_mask.py` -- mask construction
- `metrics/metrics.py` -- metric functions

---

## 2. Per-File Review

### 2.1 `eval/config.py` -- EvalConfig

**Verdict: CORRECT**

`source_city` and `target_city` are independently configurable via the dataclass constructor. The `experiment_matrix` in `run_all.py` and the `--source`/`--target` CLI args in `run_model.py` both pass these directly, so all four combinations (NYC->NYC, CHI->CHI, NYC->CHI, CHI->NYC) are fully supported.

The `adj_path(city)` method correctly takes a `city` parameter, decoupling the adjacency matrix path from the `target_city` default. The `flow_path`, `holiday_mask_path`, and `weather_mask_path` methods also correctly accept a `city` parameter.

**Minor observation**: `adj_path` uses `config.adj_top_k` but the `city` parameter is positional (not pulling from `self.target_city`), which is correct and flexible. The registry calls `config.adj_path(config.target_city)`, which is the right semantics for graph models that operate on the target city's topology.

**No hardcoded cities or paths**. All paths are constructed from configurable parameters.

---

### 2.2 `eval/evaluator.py` -- Evaluator

**Verdict: CORRECT with one important note**

#### 2.2.1 Cross-city vs. same-city code path

The evaluator correctly distinguishes between same-city and cross-city settings:

```python
is_cross_city = cfg.source_city != cfg.target_city and not isinstance(model, HAModel)
```

**Cross-city path**:
1. Loads `source_flow` and `source_mask` from the source city.
2. Fits `source_scaler` on the **source city's train set** (indices `s_train_s:s_train_e`).
3. Normalizes the **entire source sequence** using `source_scaler`.
4. Creates `train_X, train_Y` and `val_X, val_Y` from the **source city's normalized data**.
5. Calls `model.fit(train_X, train_Y, val_X, val_Y)`.

**Same-city path**:
1. Fits `target_scaler` on the **target city's train set** (indices `train_s:train_e`).
2. Normalizes the **entire target sequence** using `target_scaler`.
3. Creates `train_X, train_Y` and `val_X, val_Y` from the **target city's normalized data**.
4. Calls `model.fit(train_X, train_Y, val_X, val_Y)`.

**Test phase (shared)**:
1. Constructs `raw_test_Y` from the **raw (unnormalized) target flow**.
2. For non-HA models: normalizes target flow with `target_scaler`, creates `test_X`, predicts, then **inverse-transforms** predictions back to raw scale.
3. For HA models: uses `predict_with_timestamps()` which directly outputs raw-scale values.

This means:
- **Train**: normalized data (source-normalized for cross-city, target-normalized for same-city)
- **Test**: raw-scale ground truth vs. raw-scale predictions
- The normalization-distribution-shift in cross-city is intentional and documented

#### 2.2.2 Scaler correctness for cross-city

**Critical finding**: The `target_scaler` is always fitted on the **target city's train set** (`target_flow[train_s:train_e]`), regardless of same-city or cross-city setting. This is correct because:

- At test time, we need to normalize the target city's test input and then inverse-transform predictions back to raw scale.
- The `target_scaler` must be fit on the target city's training data to enable proper inverse transformation.
- The `source_scaler` (cross-city only) is used to normalize the source city's training data for model fitting.

**Validation**: The design is correct. The source scaler normalizes the training data the model sees; the target scaler normalizes the test input and inverse-transforms predictions. The distribution shift between source-normalized training and target-normalized test input is the core research challenge.

#### 2.2.3 Ground truth construction

The ground truth for evaluation (`raw_test_Y`) is built from **raw, unnormalized target flow**. No normalization round-trip is applied to the ground truth. This avoids any information leakage from the scaler into the ground truth. The `preds_raw` are inverse-transformed from normalized predictions. Both y_true and y_pred are in the same raw scale. **Correct**.

#### 2.2.4 Anomaly mask alignment

The `test_labels` are constructed from `target_mask` (holiday OR weather anomaly for the target city) using the same sliding window logic. The anomaly indicators are used only for stratified evaluation (full/anomaly/normal subsets), not for training. **Correct**.

#### 2.2.5 HA model special handling

The HA model is always treated as same-city (bypasses cross-city logic). This is correct because HA simply computes per-(hour, weekday, node) historical averages from the training data, which must come from the target city. The `is_cross_city` flag explicitly excludes HA models.

#### 2.2.6 Potential issue: validation set from source city in cross-city mode

In cross-city mode, the validation set is built from the source city's normalized data. The early stopping criterion is based on source-city validation loss, which may not correlate with target-city performance. This is a known limitation of unsupervised domain adaptation (the model has no access to target-city labels during training). **Not a bug, but a research design choice** that should be acknowledged in the paper.

---

### 2.3 `eval/reporter.py` -- Reporter

**Verdict: CORRECT**

Outputs are clean and informative. The `save_results` function uses a timestamped filename format that includes model name, source, and target cities, making results easy to trace. The `print_results` function displays a formatted table with all subsets.

**No issues found**.

---

### 2.4 `experiments/run_all.py` -- Batch Runner

**Verdict: CORRECT**

The `EXPERIMENT_MATRIX` covers all four settings:
```python
{"source": "NYC", "target": "NYC", "tag": "same-city"},
{"source": "CHI", "target": "CHI", "tag": "same-city"},
{"source": "NYC", "target": "CHI", "tag": "cross-city"},
{"source": "CHI", "target": "NYC", "tag": "cross-city"},
```

The `--settings` CLI argument supports `same-city`, `cross-city`, and `all`, providing flexible filtering.

**Resume/skip logic**: The `_config_matches` function checks if existing results match the current config parameters. This is a good practice for long-running experiments. The check compares `history_len`, `predict_len`, `anomaly_min_ratio`, and `train_steps`. **Correct**.

**One minor note**: The resume logic only checks the most recent matching file (`existing[-1]`). If multiple results exist for the same setting, only the last one is checked. This is reasonable behavior.

---

### 2.5 `experiments/run_model.py` -- Single Model Runner

**Verdict: CORRECT**

Provides `--source` and `--target` as independent CLI arguments, defaulting to "NYC" for both. This allows any combination. The `EvalConfig` is constructed with these values directly. **Clean and correct**.

---

### 2.6 `data_preprocess/dataset.py` -- Data Loading and Splitting

**Verdict: CORRECT**

#### 2.6.1 Data split logic

```python
def split_data(T, train_steps, val_steps):
    return (0, train_steps), (train_steps, train_steps + val_steps), (train_steps + val_steps, T)
```

The split is:
- **Train**: indices `[0, train_steps)` = 2014-10-01 to 2016-06-30 (15336 hourly steps)
- **Val**: indices `[train_steps, train_steps + val_steps)` = 2016-07-01 to 2017-01-31 (5160 hourly steps)
- **Test**: indices `[train_steps + val_steps, T)` = 2017-02-01 to 2017-09-29 (5784 hourly steps)

The total T = 15336 + 5160 + 5784 = 26280 steps. From 2014-10-01 to 2017-09-30 is 1095 days = 26280 hours. **Correct**.

The split is month-aligned (train ends June 30, val ends January 31, test starts February 1), which is a good practice for time series to avoid splitting within a month.

#### 2.6.2 Sliding window construction

```python
num_samples = max(0, end - H - P + 1 - start)
```

For train: `num_samples = 15336 - 12 - 12 + 1 - 0 = 15313`.  
For val: `num_samples = 5160 - 12 - 12 + 1 = 5137`.  
For test: `num_samples = 5784 - 12 - 12 + 1 = 5761`.  

**Correct**.

#### 2.6.3 Anomaly label construction

```python
ratio = mask[i + H: i + H + P].mean()
sample_labels[idx] = ratio >= anomaly_min_ratio
```

The anomaly label is computed from the prediction window (future P steps), not the history window. This is correct because we want to know whether the prediction target period contains anomalies. The default `anomaly_min_ratio = 0.5` means a sample is labeled "anomaly" if at least 50% of the prediction window steps have anomaly flags.

#### 2.6.4 `load_flow` and `load_anomaly_mask`

Both functions accept `city` as a parameter and construct paths dynamically. **No hardcoded cities**. The anomaly mask is the logical OR of holiday and weather masks, which is the intended behavior.

---

### 2.7 `data_preprocess/normalizer.py` -- PerNodeStandardScaler

**Verdict: CORRECT**

Per-node standardization (mean and std computed independently per node). The `transform` and `inverse_transform` are proper inverses (assuming no numerical issues). Near-zero std nodes are handled by setting std to 1.0 with a warning. **Correct and numerically stable**.

The `fit` is called on the training portion only, which is standard practice. The evaluator correctly fits the source scaler on `source_flow[s_train_s:s_train_e]` and the target scaler on `target_flow[train_s:train_e]`.

---

### 2.8 `data_preprocess/torch_dataset.py` -- TrafficFlowDataset

**Verdict: CORRECT**

Simple PyTorch Dataset wrapping numpy arrays. The `labels` are stored but not returned in `__getitem__` (only X and Y are returned). This is intentional: labels are used by the evaluator for stratified evaluation, not by the model during training. **Correct**.

The `build_dataloader` function correctly sets `pin_memory` based on CUDA availability.

---

### 2.9 `model/registry.py` -- Model Registry

**Verdict: CORRECT with caveat**

All 16 models (HA, LSTM, DLinear, PatchTST, TimesNet, STGCN, DCRNN, GWNET, STGODE, TimeAlign, ProtoTS, ModernTCN, EVL, EALGAP, DAN, STSSDL) are registered with consistent build functions.

**Graph-based models** (STGCN, DCRNN, GWNET, STGODE): Load adjacency from `config.adj_path(config.target_city)`. This means in cross-city mode, the graph model uses the target city's adjacency matrix. This is the correct design: the model is trained on source data but the graph structure should reflect the target city's topology (since we evaluate on the target city). The adjacency is only used for the spatial convolution structure, not for training labels.

**Non-graph models** (LSTM, DLinear, PatchTST, TimesNet, TimeAlign, ProtoTS, ModernTCN, EVL, EALGAP, DAN, STSSDL): Do not use adjacency. They only receive `H`, `P`, and `train_config`. In cross-city mode, these models are purely temporal and thus can be trained on source data and tested on target data without any spatial structure mismatch.

**Potential concern for graph models**: When the source and target cities have different numbers of nodes (N), the graph model's input dimension is determined by the target city's adjacency (`N = adj.shape[0]`). During training, the model receives data from the source city, which may have a different N. Let me trace this more carefully...

In the evaluator's cross-city path:
```python
source_flow_norm = source_scaler.transform(source_flow.astype(np.float64))
train_X, train_Y, _ = create_sliding_windows(source_flow_norm, ...)
```
`train_X` has shape `(num_train, H, N_source)`.

In the model's `fit()` method (via `TorchModel.fit()`):
```python
train_loader = build_dataloader(train_X, train_Y, ...)
```
The DataLoader yields batches of shape `(B, H, N_source)`.

For `all_nodes` models like STGCN, `forward()` calls `_net_forward(X)` directly. But the model was initialized with `N = adj.shape[0]` where `adj` comes from the **target** city. If `N_source != N_target`, the model will receive input with `N_source` nodes but its internal graph convolution expects `N_target` nodes. This will cause a **dimension mismatch error** at runtime.

**This is a real issue for graph-based models in cross-city settings.** However, since the paper focuses on NYC and CHI, and both cities may have the same number of nodes (or the graph models may not be intended for cross-city evaluation), this needs to be verified. If NYC and CHI have different N, graph models will fail in cross-city mode.

**Severity: HIGH** if graph models are intended for cross-city evaluation with different N. If all cities have the same N by construction, this is not an issue.

---

## 3. Cross-Cutting Concerns

### 3.1 Source and Target City Independence

**Verdict: CORRECT**

Both source and target cities can be independently specified through:
- `EvalConfig(source_city=..., target_city=...)`
- `--source` and `--target` CLI arguments
- `EXPERIMENT_MATRIX` in `run_all.py`

All four combinations (NYC->NYC, CHI->CHI, NYC->CHI, CHI->NYC) are supported and should work correctly.

### 3.2 Scaler Usage in Cross-City Mode

**Verdict: CORRECT**

| Scenario | Train Scaler | Test Scaler |
|----------|-------------|-------------|
| Same-city | target_scaler (fit on target train) | target_scaler (fit on target train) |
| Cross-city | source_scaler (fit on source train) | target_scaler (fit on target train) |

The design is logically sound. The source scaler normalizes training data for stable model optimization. The target scaler normalizes test input and inverse-transforms predictions. The distribution shift is the intended research challenge.

### 3.3 Adjacency Matrix Loading

**Verdict: CORRECT (with caveat for graph models, see Section 2.9)**

The adjacency matrix is always loaded for the **target city** via `config.adj_path(config.target_city)`. This is the correct choice:
- In same-city mode: source = target, so adjacency matches the data.
- In cross-city mode: the graph structure should reflect the target city's spatial topology.

### 3.4 Same-City and Cross-City Code Path Sharing

**Verdict: CORRECT**

Both settings share the same `Evaluator.run()` method. The `is_cross_city` flag branches only the training data construction. The test phase and evaluation are identical. This is clean and maintainable.

### 3.5 Month-Aligned Data Split

**Verdict: CORRECT**

The split boundaries are at month boundaries:
- Train ends: 2016-06-30
- Val ends: 2017-01-31
- Test starts: 2017-02-01

This avoids splitting within a month, which is important for time series with seasonal patterns.

### 3.6 Anomaly Label Construction

**Verdict: CORRECT**

Holiday masks are built from US federal holidays + city-specific holidays (e.g., Lincoln's Birthday for both NYC and CHI). Weather masks are built from extreme weather conditions (heavy rain, heavy snow, storm, low visibility) based on `wx_phrase` field matching. The anomaly mask is the logical OR of holiday and weather masks.

The `anomaly_min_ratio` parameter controls the threshold for labeling a sample as anomaly. The default of 0.5 (majority of prediction window steps are anomalous) is reasonable.

### 3.7 Hardcoded Paths and City Names

**Verdict: CORRECT -- No hardcoded paths or city names in the evaluation pipeline**

All paths are constructed from configurable parameters. The only "hardcoded" values are the defaults in `EvalConfig` (NYC for both source and target), which are overridable. The city names NYC and CHI appear in `EXPERIMENT_MATRIX` in `run_all.py`, but this is a configuration list, not hardcoded logic.

The data preprocessing scripts (`build_holiday_mask.py`, `build_weather_mask.py`) have hardcoded city lists in the `get_city_extra_holidays` function, but this is expected (holiday definitions are city-specific). The evaluation pipeline itself does not depend on these hardcoded lists.

---

## 4. Issues Found

### Issue 1: Graph Model Dimension Mismatch in Cross-City Mode (Severity: HIGH)

**File**: `model/registry.py`, `model/stgcn.py`, `model/dcrnn.py`, `model/gwnet.py`, `model/stgode.py`

**Description**: Graph-based models (STGCN, DCRNN, GWNET, STGODE) are initialized with `N = adj.shape[0]` where `adj` comes from the **target** city. In cross-city mode, the training data comes from the **source** city, which may have a different number of nodes. If `N_source != N_target`, the model's internal graph convolution layers will receive input tensors with a mismatched spatial dimension.

**Impact**: Runtime error (dimension mismatch) when running graph models in cross-city mode with cities of different sizes.

**Recommendation**: 
- If NYC and CHI have the same N by design, document this explicitly.
- If they may differ, either: (a) pad/truncate nodes to match, (b) use a node-alignment mechanism, or (c) restrict graph models to same-city evaluation only.

### Issue 2: Cross-City Validation Set from Source City (Severity: LOW)

**File**: `eval/evaluator.py`

**Description**: In cross-city mode, the validation set is constructed from the source city's normalized data. The early stopping criterion is based on source-city validation loss, which may not correlate with target-city test performance.

**Impact**: Models may early-stop at suboptimal points for target-city performance. This is a known limitation of unsupervised domain adaptation.

**Recommendation**: Acknowledge this in the paper. Consider adding an option to use a small portion of target-city data for validation (if available), or to disable early stopping and use a fixed number of epochs for cross-city experiments.

### Issue 3: No Warm-Up or Adaptation Phase for Cross-City (Severity: MEDIUM)

**File**: `eval/evaluator.py`

**Description**: In cross-city mode, the model is trained entirely on source-city data and then directly evaluated on target-city test data. There is no domain adaptation or fine-tuning on the target city's training/validation data.

**Impact**: This is a "zero-shot" cross-city transfer setting. While this is a valid research design, it represents the most challenging scenario. The paper should clearly state this assumption.

**Recommendation**: Explicitly document that this is a zero-shot transfer setting. If the research question includes fine-tuning, consider adding a target-city fine-tuning phase.

### Issue 4: Trainer's `build_dataloader` Ignores Labels (Severity: LOW)

**File**: `model/torch_base.py`, lines 69-73

**Description**: In `TorchModel.fit()`, `build_dataloader` is called with `np.zeros(len(train_X), dtype=bool)` for labels, effectively discarding anomaly information during training.

**Impact**: Anomaly-aware training strategies (e.g., weighted loss, curriculum learning) cannot be implemented without modifying this interface.

**Recommendation**: This is fine for the current models (none use anomaly labels during training), but if future models need anomaly-aware training, the `fit()` interface should pass the actual labels.

---

## 5. Overall Assessment

### Strengths

1. **Clean separation of concerns**: Config, Evaluator, Reporter, and Runner are well-isolated.
2. **Flexible city configuration**: Source and target cities are independently configurable.
3. **Correct scaler design**: Source scaler for training, target scaler for test -- the distribution shift is the intended challenge.
4. **Month-aligned time split**: Avoids splitting within months, which is critical for seasonal time series.
5. **Robust anomaly labeling**: Holiday + weather masks with configurable threshold.
6. **No hardcoded paths**: Fully configurable through EvalConfig.
7. **Resume/skip logic**: Prevents re-running completed experiments.
8. **Both CLI and batch interfaces**: `run_model.py` for single runs, `run_all.py` for systematic experiments.

### Weaknesses

1. **Graph model dimension mismatch risk** in cross-city mode (Issue 1).
2. **No target-city validation** in cross-city mode, making early stopping suboptimal (Issue 2).
3. **Zero-shot transfer only** -- no fine-tuning phase (Issue 3).

### Recommendation

1. **Verify** that NYC and CHI have the same number of nodes, or add a dimension check/assertion in the evaluator for graph models in cross-city mode.
2. **Document** the zero-shot cross-city transfer setting clearly in the paper.
3. **Consider** adding an assertion in `evaluator.py` that checks `N_source == N_target` when graph models are used in cross-city mode, to fail fast with a clear error message rather than a cryptic dimension mismatch.

---

## 6. Summary

The cross-city traffic flow prediction setting is **correctly implemented** for the intended use case. The code paths for same-city and cross-city evaluation are well-designed and share the same evaluation logic. The key design decisions (separate source/target scalers, target-city adjacency for graph models, month-aligned splits) are sound. The main concern is the potential dimension mismatch for graph models when source and target cities have different numbers of nodes, which should be verified or explicitly handled.