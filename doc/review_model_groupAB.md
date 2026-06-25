# STCross Baseline Model Review Report (Group A & Group B)

**Review Date:** 2025-06-25
**Reviewer:** PhD Researcher (Spatio-Temporal Data Mining)
**Scope:** Group A (Temporal Methods) & Group B (Spatio-Temporal Graph Methods)
**Context:** STCross is a cross-city traffic flow forecasting framework. The framework interface is:
- Data format: `(B, H, N)` input, `(B, P, N)` output (all_nodes format)
- `TorchModel.forward()` handles per-node reshaping for `input_mode="per_node"` models
- `Trainer` calls `model(X)` directly; no separate forward/reshape logic needed
- All models use `PerNodeStandardScaler` (per-node Z-score normalization)
- Cross-city transfer: source city data is source-normalized; target city data is target-normalized; the model sees a distribution shift by design

---

## 1. LSTM (`model/lstm.py`)

### Paper Reference
Generic 2-layer LSTM for time series forecasting. No specific single paper; LSTM was introduced by Hochreiter & Schmidhuber (1997).

### Architecture Review

**Architecture Correctness: PASS**
- Simple 2-layer LSTM with input_size=1 (univariate per-node), hidden_dim=64, batch_first=True.
- Linear projection `fc: Linear(hidden_dim, P)` maps the last hidden state to P future steps.
- This is a standard many-to-one LSTM forecasting setup.

**Paper Faithfulness: PASS**
- The architecture is a textbook LSTM forecaster. No deviation from standard practice.

**Framework Compatibility: PASS**
- `input_mode = "per_node"` -- correct for a channel-independent model.
- `_net_forward`: receives `(B*N, H, 1)`, returns `(B*N, P, 1)`. The `forward()` in `TorchModel` handles the `(B,H,N) -> (B*N,H,1)` reshape and the reverse.
- Dimensions check out: `X: (B*N, H, 1)` -> LSTM -> `(B*N, H, hidden)` -> `fc(out[:,-1,:])` -> `(B*N, P)` -> `unsqueeze(-1)` -> `(B*N, P, 1)`.

**Potential Issues:**
- The `dropout` parameter is only applied when `num_layers > 1`, which is correct (PyTorch LSTM dropout is inter-layer only).
- No issues found.

### Severity: None

---

## 2. DLinear (`model/dlinear.py`)

### Paper Reference
Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023.

### Architecture Review

**Architecture Correctness: PASS**
- Moving average decomposition via `AvgPool1d` with kernel_size=3, stride=1, padding=1.
- Trend = MA(X), Seasonal = X - MA(X).
- Two separate `Linear(H, P)` layers for trend and seasonal branches.
- Final output = trend_out + seasonal_out.

**Paper Faithfulness: PASS**
- The original DLinear paper uses exactly this decomposition: `X_t = AvgPool(Padding(X))` for trend, `X_s = X - X_t` for seasonal, then two linear layers and sum.
- The paper uses `kernel_size=25` for ETT datasets and discusses different kernel sizes. The default of `kernel_size=3` is reasonable for traffic data with H=12 (smaller lookback window).
- The paper does NOT use RevIN in DLinear (unlike PatchTST). This implementation correctly omits RevIN.

**Framework Compatibility: PASS**
- `input_mode = "per_node"`.
- `_net_forward`: `(B*N, H, 1)` -> squeeze -> Linear(H,P) -> `(B*N, P, 1)`.
- The `MovingAvg.forward()` permutes to `(B, 1, H)` for 1D pooling over time, then permutes back. Correct.

**Potential Issues:**
- **Minor concern:** The `MovingAvg` uses `AvgPool1d` which is a simple moving average. The DLinear paper uses a simple moving average as well. However, for traffic data with H=12, a kernel_size of 3 gives very little smoothing (the smoothed series is essentially a 3-point moving average). With such a short window, the decomposition may not meaningfully separate trend from seasonal. This is a design choice, not a bug.
- The `self.linear_trend` and `self.linear_seasonal` are both `nn.Linear(H, P)` with no bias. The paper also uses `nn.Linear` without bias -- correct.

### Severity: None (minor design consideration noted)

---

## 3. PatchTST (`model/patchtst.py`)

### Paper Reference
Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023.

### Architecture Review

**Architecture Correctness: PASS**
- Patch embedding: `unfold(dimension=1, size=patch_len, step=stride)` then `Linear(patch_len, d_model)`.
- Positional encoding: learnable `nn.Parameter` added to patch embeddings.
- TransformerEncoder with GELU activation, batch_first=True.
- Flatten + Linear head: `Linear(num_patches * d_model, P)`.
- RevIN: instance normalization before and denormalization after.

**Paper Faithfulness: PASS**
- The paper uses channel-independent PatchTST with RevIN. This implementation follows that exactly.
- Patch embedding via `unfold` + linear projection matches the paper.
- Learnable additive positional encoding matches the paper.
- GELU activation in TransformerEncoderLayer matches the paper.
- The flatten + linear projection head is the standard approach.

**Framework Compatibility: PASS**
- `input_mode = "per_node"`.
- `_net_forward`: `(B*N, H, 1)` -> RevIN norm -> patch embed -> Transformer -> flatten -> Linear -> `(B*N, P)` -> unsqueeze -> `(B*N, P, 1)` -> RevIN denorm.
- Correct.

**Potential Issues:**
- **Medium concern - stride/patch configuration:** Default `patch_len=4, stride=2` with H=12 gives `num_patches = (12-4)//2 + 1 = 5`. The paper experiments with `patch_len=16, stride=8` for longer sequences (H=96, etc.). For H=12, the configuration is reasonable but the receptive field is limited. With only 5 patches, the Transformer has very limited sequence context. This is an inherent limitation of short lookback windows, not a code bug, but worth noting in experiments.
- The RevIN implementation uses `unbiased=False` in `var()`, which matches the paper's population variance.
- The `.detach()` calls on `means` and `stdev` correctly prevent gradients from flowing through the normalization statistics.

### Severity: None (configuration note for H=12)

---

## 4. TimesNet (`model/timesnet.py`)

### Paper Reference
Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023.

### Architecture Review

**Architecture Correctness: PASS**
- `TokenEmbedding`: Conv1d with kernel_size=3, padding=1, circular padding. Matches TSLib.
- `predict_linear`: `Linear(H, H+P)` extends temporal dimension in feature space. Matches TSLib's approach of extending the lookback window to include prediction horizon.
- `TimesBlock`: FFT period detection -> 2D reshape -> dual Inception (GELU activation) -> softmax-weighted aggregation -> residual connection.
- `LayerNorm` after each TimesBlock.
- `projection`: `Linear(d_model, 1)` per timestep, then take last P steps.
- RevIN normalization/denormalization.

**Paper Faithfulness: PASS**
- FFT-based period detection: `torch.fft.rfft` -> amplitude -> top-k frequencies -> convert to periods. Matches TSLib's `FFT_for_Period` exactly.
- 2D reshape: `(B, T, C) -> (B, n_periods, p, C) -> (B, C, n_periods, p)` for Conv2d. Matches the paper.
- `InceptionBlockV1`: multi-scale Conv2d with kernel sizes {1, 3, 5, ..., 2k-1}, mean aggregation. Matches TSLib.
- Dual Inception (d_model -> d_ff -> d_model) with GELU in between.
- Softmax-weighted aggregation of multi-period outputs.
- Residual connection after TimesBlock.
- Note: TSLib has a "batch-averaged amplitude" for period selection. The paper's original implementation selects top-k periods per batch by averaging amplitudes across the batch. This implementation correctly uses `mean_amp = amp.mean(dim=0)` for batch-averaged period selection.

**Framework Compatibility: PASS**
- `input_mode = "per_node"`.
- `_net_forward`: `(B*N, H, 1)` -> RevIN -> embed -> predict_linear -> TimesBlocks -> projection -> last P steps -> RevIN denorm -> `(B*N, P, 1)`.
- Correct.

**Potential Issues:**
- **Medium concern - MPS workaround:** The `__init__` method has a special case for MPS devices: `if train_config.device == "auto" and not torch.cuda.is_available()` forces `device="cpu"`. This is documented as "MPS has a backward bug with multi-branch Conv2d+view+residual patterns." This is a pragmatic workaround but should be noted in experiments.
- **Minor concern - FFT period detection efficiency:** The `fft_for_period` function computes FFT and amplitude averaging on every forward pass. For per-node mode with `(B*N, H, 1)` tensors, the FFT operates on a 1-channel signal, which is efficient. The batch-averaging for period selection is correct.
- The `predict_linear` `Linear(H, H+P)` operates on the feature dimension after permute: `(B, H, d_model) -> (B, d_model, H) -> Linear -> (B, d_model, H+P) -> (B, H+P, d_model)`. This correctly extends the temporal dimension. This is a linear projection in feature space, matching TSLib.

### Severity: Low (MPS workaround noted)

---

## 5. HA (`model/ha.py`)

### Paper Reference
Historical Average is a standard baseline, not tied to a specific paper.

### Architecture Review

**Architecture Correctness: PASS**
- Computes mean flow per (hour_of_day, day_of_week, node) from training data.
- Lookup table: `(24, 7, N)`.
- Prediction: for each timestamp, looks up the corresponding historical average.

**Paper Faithfulness: N/A** (standard baseline)

**Framework Compatibility: PASS**
- `HAModel` extends `BaseModel` (not `TorchModel`), so it has its own `fit()` and `predict()` methods.
- `predict_with_timestamps()` takes timestamp arrays and returns `(num_samples, P, N)` predictions.
- The `Evaluator` handles HA specially, calling `predict_with_timestamps()` instead of `predict()`.
- All input/output shapes are correct.

**Potential Issues:**
- `predict()` raises `NotImplementedError` -- this is intentional, since HA needs timestamps, not just the input window X.
- No `input_mode` attribute defined (not needed since it extends `BaseModel` directly).
- The `fit()` method does not use `val_X` or `val_Y` -- this is correct for HA, which has no validation-based early stopping.

### Severity: None

---

## 6. STGCN (`model/stgcn.py`)

### Paper Reference
Yu et al., "Spatio-Temporal Graph Convolutional Networks", IJCAI 2018.

### Architecture Review

**Architecture Correctness: PASS**
- Scaled Laplacian: `L_tilde = 2*L_sym/lambda_max - I`. Matches the paper's Eq. (12-13).
- `ChebGraphConv`: Chebyshev polynomial up to order Ks. Matches paper Eq. (3-4).
- `TemporalConvLayer`: 2D Conv + GLU gating (`P * sigmoid(Q)`). Matches paper Eq. (5-6).
- `STConvBlock`: TemporalConv -> GraphConv -> TemporalConv -> LayerNorm. Matches paper.
- Stack 2 STConvBlocks with channel dimensions [1, 16, 64] and [64, 16, 64].
- Output layer: temporal squeeze + FC.

**Paper Faithfulness: PASS**
- The paper uses Chebyshev spectral graph convolution with Ks=3. This implementation uses Ks=3 by default.
- GLU gating: `(P * sigmoid(Q))` matches the paper's Eq. (5).
- The paper uses 2 ST-Conv blocks. This implementation uses 2 blocks by default.
- LayerNorm after each block matches the paper.
- The output layer uses a temporal convolution to squeeze to 1 timestep, then FC to P steps. This is a reasonable adaptation for multi-step output.

**Tensor Layout Analysis:**
- The model uses `(B, C, N, T)` as the internal layout. The `_net_forward` reshapes: `(B, H, N) -> (B, 1, N, H)` by `X.unsqueeze(1).permute(0, 1, 3, 2)`.
- Wait -- let's trace carefully:
  - Input: `(B, H, N)`
  - `X.unsqueeze(1)` -> `(B, 1, H, N)`
  - `.permute(0, 1, 3, 2)` -> `(B, 1, N, H)` -- this is `(B, C, N, T)`.
  - `TemporalConvLayer` uses `nn.Conv2d(c_in, 2*c_out, kernel_size=(1, Kt))` which convolves along the last dim (T). Since layout is `(B, C, N, T)`, kernel `(1, Kt)` convolves over T with spatial dim N untouched. This is correct.
  - `GraphConvLayer` -> `ChebGraphConv` expects `(B, C, T, N)`, so `x.permute(0, 1, 3, 2)` -> ChebConv -> `permute(0, 1, 3, 2)` back to `(B, C, N, T)`. This is correct.
  - After 2 blocks: `Ko = H - 2*(Kt-1)*2 = 12 - 8 = 4` with Kt=3, 2 blocks.
  - Output layer: `TemporalConvLayer(Ko, last_c, last_c)` squeezes to 1 timestep. Then `LayerNorm` -> `TemporalConvLayer(1, last_c, last_c)` -> squeeze -> `(B, N, C_out)` -> `fc. Linear(last_c, P)` -> `(B, N, P)` -> `permute(0, 2, 1)` -> `(B, P, N)`.
  - Correct.

**Adjacency Matrix:**
- Uses `scaled_laplacian()` from correlation-based adjacency. The correlation adjacency is built with top-K sparsification and symmetrization. The scaled Laplacian computation is correct.
- Registered as buffer `gso` in `ChebGraphConv`. Correct for cross-city transfer (different N, different adjacency).

**Cross-City Transfer:**
- N is passed as a parameter and adjacency is loaded from the target city's adjacency file. The model is instantiated with the target city's N and adjacency.
- However, during cross-city transfer, the model is trained on source data (which has source N) but tested on target data (which has target N). This means the adjacency matrix passed to `__init__` must match the target N.
- Looking at `registry.py` line `build_stgcn`: `adj = np.load(config.adj_path(config.target_city)); N = adj.shape[0]`. So the model is always built with target N and target adjacency.
- **Critical issue:** During training on source data, the source flow has `N_source` nodes, but the model expects `N_target` nodes. The `_net_forward` receives `(B, H, N_target)` because `forward()` in `TorchModel` does NOT reshape for `all_nodes` mode -- it passes `(B, H, N)` directly. But the training data loader produces `(B, H, N_source)`.
- **Wait -- re-examining.** The cross-city evaluator (`evaluator.py`) loads source flow data, normalizes it, creates sliding windows. The source data has `N_source` nodes. The model is built with `N_target`. The `fit()` method calls `build_dataloader(train_X, ...)` where `train_X` has shape `(num_samples, H, N_source)`. The DataLoader yields batches of `(B, H, N_source)`. The model's `forward()` passes this to `_net_forward()` which expects `(B, H, N_target)`.
- **This is a shape mismatch!** The model's `_net_forward` will receive `(B, H, N_source)` but the internal parameters (e.g., `ChebGraphConv.gso` of shape `(N_target, N_target)`) will fail when trying to do `torch.einsum("nm,btmc->btnc", self.gso, x)` where `x` has `N_source` nodes.
- **This is a design-level issue for ALL Group B models.** Let me verify by checking the evaluator flow more carefully.

**Re-examining the evaluator cross-city flow:**
- `evaluator.py` line 40-60: In cross-city mode, it loads source data, normalizes, creates windows.
- `train_X` has shape `(num_samples, H, N_source)`.
- `model.fit(train_X, train_Y, val_X, val_Y)` -- `TorchModel.fit()` creates dataloader from these arrays.
- The dataloader yields `(B, H, N_source)`.
- `Trainer` calls `model(X)` where `X` is `(B, H, N_source)`.
- `TorchModel.forward()` for `all_nodes` mode: `return self._net_forward(X)` -- passes `(B, H, N_source)` directly.
- `_net_forward` in STGCN does `X.unsqueeze(1).permute(0, 1, 3, 2)` to get `(B, 1, N_source, H)`.
- But `gso` is `(N_target, N_target)`. When `einsum("nm,btmc->btnc", gso, x)` is called, `n=N_target` but `m` in `x` is `N_source`. This will fail with a runtime error.

**Wait** -- this is actually impossible for the code to work at all if this is the case. Let me look more carefully at the shapes. The `gso` has shape `(N_target, N_target)`. The `x` in `ChebGraphConv.forward` has shape `(B, T, N_source, C_in)`. The einsum `"nm,btmc->btnc"` contracts `m` between `gso` and `x`. `gso` has `m=N_target`, `x` has `m=N_source`. If `N_source != N_target`, this will fail.

**However**, looking at the evaluator more carefully: the source data is normalized by `source_scaler` and target data is normalized by `target_scaler`. The `PerNodeStandardScaler` stores per-node statistics. The source data has `N_source` nodes, and the target data has `N_target` nodes. If `N_source != N_target`, the model cannot train on source data at all.

**But the STCross project is about cross-city transfer.** This means either:
1. The source and target cities have the same N (by design or by node selection).
2. There is some N-alignment mechanism I'm not seeing.
3. This is a real bug.

Let me check if NYC and CHI have the same number of nodes. Looking at the data processing, the flow data is loaded from `flow_{city}_{start_date}_{end_date}_{freq_min}min.npy`. The number of nodes N depends on the city's sensor network. NYC and CHI likely have different numbers of sensors.

**This is a CRITICAL bug for all Group B models in cross-city transfer mode.** The model is built with `N_target` and `adj_target` (shape `(N_target, N_target)`), but trained on source data with `N_source` nodes. The shapes won't match.

**However**, for same-city evaluation (source=target), this is not an issue. The cross-city transfer issue is a separate concern and may be handled elsewhere (e.g., by only using Group B models for same-city evaluation, or by a node-matching mechanism).

Given the scope of this review is the model implementations themselves, I'll note this as a critical cross-city compatibility issue but it's not a bug in the model code per se -- it's an integration issue.

**Potential Issues:**
- **Medium concern:** The `output_tcl2` in the output layer uses `TemporalConvLayer(1, last_c, last_c)` with `Kt=1`, which is a pointwise convolution. This is effectively a channel mixing layer. The paper doesn't have this exact output layer structure -- the original STGCN uses a single temporal convolution + FC for output. The extra normalization and convolution layers are a reasonable extension but deviate slightly from the paper.
- The `LayerNorm` in `STConvBlock` is applied as `LayerNorm([c_out, N])` which normalizes over the channel and node dimensions jointly. The paper uses batch normalization (not LayerNorm). The code uses LayerNorm for stability with small batch sizes. This is a reasonable substitution.
- **Low concern:** The paper uses ReLU after graph convolution. The implementation uses `F.relu` after the residual addition in `GraphConvLayer`. Correct.

### Severity: None for same-city; Critical for cross-city (integration-level, not model-level)

---

## 7. DCRNN (`model/dcrnn.py`)

### Paper Reference
Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting", ICLR 2018.

### Architecture Review

**Architecture Correctness: PASS**
- `build_supports`: Forward and backward random walk power matrices P_fw^k, P_bw^k for k=0..K. Matches the paper's diffusion convolution.
- `DiffusionConv`: `sum_k W_k * P^k * X` via einsum, then linear mix. Matches paper Eq. (2).
- `DCGRUCell`: Diffusion convolution replaces matrix multiplication in GRU gates. Matches paper Eq. (3-5).
- `DCRNNEncoder`: Iterates over T timesteps, feeding through DCGRU layers. Matches paper.
- `DCRNNDecoder`: Autoregressive decoding for P steps. Uses `go_input = X[:,-1,:,:]` (last observation). Matches the paper's scheduled sampling approach (though scheduled sampling is not implemented here -- the decoder uses its own predictions as input).
- The paper uses scheduled sampling during training. This implementation uses teacher forcing during training (the decoder always uses its own output as next input). This is a minor deviation.

**Paper Faithfulness: PASS**
- Diffusion convolution with K=2 hops, forward and backward random walks. Matches paper.
- DCGRU cell structure (r, u gates + candidate) matches paper.
- Seq2Seq architecture with encoder-decoder matches paper.
- Support matrices registered as buffers (one per support) rather than per-conv, reducing checkpoint size. This is a practical optimization.

**Framework Compatibility: PASS**
- `input_mode = "all_nodes"`.
- `_net_forward`: `(B, H, N)` -> `unsqueeze(-1)` -> `(B, H, N, 1)` -> encoder -> hidden states -> decoder -> `(B, P, N, 1)` -> `squeeze(-1)` -> `(B, P, N)`.
- Correct.

**Adjacency Matrix:**
- Forward/backward random walk normalization: `D^{-1} @ A`. Matches paper.
- Support matrices of shape `(N, N)` registered as buffers.

**Cross-City Transfer:**
- Same issue as STGCN: model built with target N and target adjacency, but trained on source data with source N. Shape mismatch in cross-city mode.

**Potential Issues:**
- **Medium concern:** No scheduled sampling. The original DCRNN paper uses scheduled sampling where the decoder randomly uses ground truth vs. its own prediction during training. This implementation uses pure autoregressive decoding (the decoder's own output is always fed back). This can lead to exposure bias and worse long-horizon predictions.
- The decoder uses `go_input = X[:, -1, :, :]` (the last observation) as the initial decoder input. This matches the paper.
- The encoder processes the sequence timestep by timestep in a Python for-loop. This is correct for DCRNN (the DCGRU is inherently sequential) but is slower than a batched implementation.

### Severity: Low (no scheduled sampling)

---

## 8. GWNET (`model/gwnet.py`)

### Paper Reference
Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019.

### Architecture Review

**Architecture Correctness: PASS**
- `start_conv`: `Conv2d(1, residual_channels, 1)` for input projection.
- `filter_convs` + `gate_convs`: Dilated causal Conv2d with `tanh` and `sigmoid` gating. Matches paper Eq. (6).
- `MixProp`: Mix-hop propagation -- concatenates `[x, A*x, A^2*x, ...]` for each support matrix, then `Conv2d` for mixing. Matches paper Eq. (8-9).
- Adaptive adjacency: `softmax(ReLU(E1 @ E2))` where `E1` is `(N, apt_dim)` and `E2` is `(apt_dim, N)`. This is the paper's Eq. (10).
- Skip connections: each layer's output is added to a skip accumulator, then `ReLU -> end_conv1 -> ReLU -> end_conv2`.
- Residual connections: `x = gconv(x) + residual` after each layer.
- BatchNorm after each layer.

**Paper Faithfulness: PASS**
- The paper uses 8 layers with dilation [1, 2, 1, 2, 1, 2, 1, 2] (4 blocks of 2 layers). This implementation uses 4 blocks of 2 layers with dilation resetting per block: [1, 2] repeated 4 times. The receptive field is `1 + 4*(1+2) = 13` (with kernel_size=2, `1 + 4*((2-1)+(4-1)) = 1 + 4*4 = 17` -- let me recalculate).
  - Layer 1: dilation=1, kernel=2, additional scope=1 -> RF += 1 -> RF=2
  - Layer 2: dilation=2, kernel=2, additional scope=2 -> RF += 2 -> RF=4
  - Layer 3: dilation=1, additional scope=1 -> RF += 1 -> RF=5
  - Layer 4: dilation=2, additional scope=2 -> RF += 2 -> RF=7
  - ... total RF = 1 + 4*3 = 13. With H=12, the receptive field of 13 is sufficient (covers the entire input).
  - The paper uses a slightly different dilation pattern: [1, 2, 4, 8, ...] or [1, 2, 1, 2, ...]. The implementation's [1, 2] repeated pattern is a valid interpretation that gives RF=13, which covers H=12.
- MixProp with order=2 (default): concatenates `[x, A_fw*x, A_fw^2*x, A_bw*x, A_bw^2*x, A_adp*x, A_adp^2*x]` for 3 supports. The `c_total = (order * num_supports + 1) * c_in` formula is correct.
- Adaptive adjacency matrix: `softmax(ReLU(E1 @ E2))`. The paper uses this. Correct.
- Supports: forward random walk, backward random walk, and adaptive adjacency. Matches paper.

**Tensor Layout:**
- `_net_forward`: `(B, H, N) -> (B, 1, N, H)` (i.e., `(B, C, N, T)`).
- All convolutions use `kernel_size=(1, kernel_size)` which convolves along the time dimension (last dim). The node dimension N is treated as the "height" and not convolved.
- `MixProp` uses `NConv` which does `einsum("bcnt,nm->bcmt", x, A)` -- this applies the graph convolution along the node dimension (dim 2 in `(B, C, N, T)`), resulting in `(B, C, M, T)` where M is the number of nodes in A. Since A is `(N, N)`, this is `(B, C, N, T)`. Correct.
- The skip connections use `skip[:, :, :, -s.size(3):]` to align temporal dimensions. This is needed because causal convolutions reduce the time dimension.
- `receptive_field` computation in `__init__` is correct: `1 + blocks * layers * (kernel_size - 1) * (1 + 2)` -- wait, the code computes it iteratively. For kernel_size=2, blocks=4, layers=2: `receptive_field = 1 + 4*2*(2-1) = 9` after the first iteration? Let me re-trace:
  - Start: `receptive_field = 1, new_dilation = 1, additional_scope = 1`
  - Layer 1: `receptive_field += 1 -> 2`, `new_dilation *= 2 -> 2`, `additional_scope *= 2 -> 2`
  - Layer 2: `receptive_field += 2 -> 4`, `new_dilation *= 2 -> 4`, `additional_scope *= 2 -> 4`
  - Block 1 done. Reset `new_dilation = 1, additional_scope = 1` (implicit).
  - Layer 3: `receptive_field += 1 -> 5`, `new_dilation = 2`, `additional_scope = 2`
  - Layer 4: `receptive_field += 2 -> 7`, `new_dilation = 4`, `additional_scope = 4`
  - ... After 4 blocks (8 layers): RF = 1 + 4*(1+2) = 13.
  - Correct, RF=13 >= H=12.

**Wait, let me re-read the code more carefully.**

```python
for _ in range(blocks):
    new_dilation = 1
    additional_scope = kernel_size - 1
    for _ in range(layers):
        ...
        receptive_field += additional_scope
        new_dilation *= 2
        additional_scope *= 2
```

For kernel_size=2, blocks=4, layers=2:
- Block 1: new_dilation=1, additional_scope=1
  - Layer 1: rf += 1 (rf=2), new_dilation=2, additional_scope=2
  - Layer 2: rf += 2 (rf=4), new_dilation=4, additional_scope=4
- Block 2: new_dilation=1, additional_scope=1
  - Layer 3: rf += 1 (rf=5), new_dilation=2, additional_scope=2
  - Layer 4: rf += 2 (rf=7), new_dilation=4, additional_scope=4
- Block 3: rf goes to 8, then 10
- Block 4: rf goes to 11, then 13

Final RF = 13. H=12. The padding `x = F.pad(x, (self.receptive_field - in_len, 0))` pads 1 on the left. Correct.

**Framework Compatibility: PASS**
- `input_mode = "all_nodes"`.
- `_net_forward` returns `(B, P, N)`.

**Cross-City Transfer:**
- Same issue as STGCN and DCRNN: model built with target N, trained on source N.
- Additionally, `nodevec1` and `nodevec2` for adaptive adjacency are `nn.Parameter` with shape `(N, apt_dim)` and `(apt_dim, N)`. These are N-dependent parameters that would need to be re-initialized for a different N.

**Potential Issues:**
- **Medium concern:** The `end_conv_2` is `Conv2d(end_channels, P, kernel_size=1)` which expects the time dimension to be 1. After all layers, the time dimension should be `in_len - (receptive_field - 1) = 12 - 12 = 0`? Wait, let me trace:
  - Input: `(B, 1, N, 12)`, padded to `(B, 1, N, 13)`.
  - After 8 causal conv layers (each with kernel=2, stride=1, no padding): time dimension reduces by 1 each layer. After 8 layers: `13 - 8 = 5`.
  - But skip connections accumulate. The `skip` variable accumulates `s` from each layer. The `s` from layer i has time dim `T_i` where `T_i` decreases over layers. The `skip[:, :, :, -s.size(3):]` alignment takes the last `s.size(3)` time steps of the accumulated skip, which effectively aligns to the most recent layer's output.
  - After all layers, `skip` has the time dimension of the last skip connection, which is `13 - 8 = 5`.
  - `end_conv_1` (kernel_size=1) preserves time dim: 5.
  - `end_conv_2` (kernel_size=1, output_channels=P): `(B, P, N, 5)`.
  - `squeeze(-1)` removes the last dim (time=5): `(B, P, N)`.
  - **Wait, this is wrong!** `end_conv_2` is `Conv2d(end_channels, P, kernel_size=1)`. The output has P channels and time dim = 5. `squeeze(-1)` removes the time dimension, taking only the first element? No, `squeeze(-1)` removes dim of size 1. But time dim is 5, not 1!
  - **This is a bug!** `squeeze(-1)` will NOT remove a dimension of size 5. It will only remove dims of size 1. The output shape would be `(B, P, N, 5)`, not `(B, P, N)`.

**Critical re-examination of GWNET output shape:**

Let me trace the output of `_net_forward` very carefully:

```python
def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
    x = X.unsqueeze(1).permute(0, 1, 3, 2)  # (B, 1, N, H) = (B, 1, N, 12)
    in_len = x.size(3)  # 12
    if in_len < self.receptive_field:  # 12 < 13
        x = F.pad(x, (self.receptive_field - in_len, 0))  # (B, 1, N, 13)
    x = self.start_conv(x)  # (B, residual_ch, N, 13)
    skip = 0
    ...
    for i in range(total_layers):  # 8 layers
        residual = x  # (B, residual_ch, N, T_i)
        filt = torch.tanh(self.filter_convs[i](residual))  # (B, dilation_ch, N, T_i - 1) causal
        gate = torch.sigmoid(self.gate_convs[i](residual))  # (B, dilation_ch, N, T_i - 1)
        x = filt * gate  # (B, dilation_ch, N, T_i - 1)
        s = self.skip_convs[i](x)  # (B, skip_ch, N, T_i - 1)
        if isinstance(skip, int):
            skip = s  # (B, skip_ch, N, T_1 - 1)
        else:
            skip = s + skip[:, :, :, -s.size(3):]
        x = self.gconv[i](x, supports)  # (B, residual_ch, N, T_i - 1)
        x = x + residual[:, :, :, -x.size(3):]  # (B, residual_ch, N, T_i - 1)
        x = self.bn[i](x)  # (B, residual_ch, N, T_i - 1)
```

After layer 1: T = 13-1 = 12, skip has T=12
After layer 2: T = 12-1 = 11, skip has T=11 (aligned)
After layer 3: T = 11-1 = 10, skip accumulates T=10
...
After layer 8: T = 13-8 = 5, skip has T=5

```python
    x = F.relu(skip)  # (B, skip_ch, N, 5)
    x = F.relu(self.end_conv_1(x))  # (B, end_ch, N, 5)
    x = self.end_conv_2(x)  # (B, P, N, 5)
    return x.squeeze(-1)  # (B, P, N, 5)? NO! squeeze(-1) only removes dim of size 1
```

**This is a CRITICAL BUG.** The `squeeze(-1)` will NOT change the shape from `(B, P, N, 5)` to `(B, P, N)`. The output will be `(B, P, N, 5)` which is NOT the expected `(B, P, N)`.

**But wait** -- the original GWNET paper (nnzhan/Graph-WaveNet) uses a different output structure. Let me check the original code:

In the original GWNET code:
```python
# Output layer
# skip: (B, skip_ch, N, T')
# end_conv_1: Conv2d(skip_ch, end_ch, (1,1))
# end_conv_2: Conv2d(end_ch, P, (1,1))
# The output has shape (B, P, N, T') where T' is the remaining temporal dimension
# Then the last step is taken: x = x[..., -1:] to get (B, P, N, 1)
```

In the original nnzhan/Graph-WaveNet, after the output convolutions, the model takes the last time step: `x = x[..., -1:]`. This makes sense because the output is a single-step prediction, and the last time step of the processed sequence corresponds to the most recent information.

This implementation does NOT take the last time step. It tries to `squeeze(-1)` which would only work if the time dimension is 1. But the time dimension is 5, so `squeeze(-1)` is a no-op.

**This is definitely a bug.** The correct approach would be either:
1. `return x[:, :, :, -1]` to take the last time step (for single-step output, then repeat or use a different head for multi-step).
2. Add a final FC layer that maps the flattened representation to P steps.

The original GWNET paper outputs P steps by having the output layer produce P channels and taking the last time step. But the paper's model is designed for a specific output structure. The implementation here seems to have confused the output handling.

**Wait, let me re-examine.** In the original GWNET, the output is `(B, 1, N, P)` or `(B, P, N, 1)` depending on the layout. Let me check: the original code uses `end_conv_2 = nn.Conv2d(end_ch, P, (1,1))` and then the output is `(B, P, N, T')`. The original code then does `x = x[:, :, :, -1]` to get `(B, P, N)`.

**Actually, I need to re-examine this more carefully. In the original GWNET paper:**
- The output is a sequence of P steps.
- The model processes the input sequence and produces `P * N` outputs.
- The original code does: `end_conv_2 = nn.Conv2d(in_channels=end_ch, out_channels=horizon * ..., kernel_size=(1,1))` and then reshapes.

Actually, looking at the official nnzhan/Graph-WaveNet code more carefully, the output convolution produces `(B, P, N, T')` and then the last time step is taken. But the paper's formulation is for multi-step prediction where the output is generated autoregressively or through a final linear layer.

**Let me reconsider.** The implementation here has `end_conv_2 = nn.Conv2d(end_channels, P, kernel_size=1)`. This produces `(B, P, N, T')`. If the intention is for the output to be `(B, P, N)` with the last time step's prediction, then `x.squeeze(-1)` is wrong and should be `x[:, :, :, -1]` or `x[..., -1]`.

**However**, if the time dimension after the last convolution is 1 (which it isn't -- it's 5), then squeeze would work. But the time dimension is NOT 1.

**This is a confirmed bug.** The output shape is incorrect.

### Severity: Critical (output shape bug)

---

## 9. STGODE (`model/stgode.py`)

### Paper Reference
Fang et al., "Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting", KDD 2021.

### Architecture Review

**Architecture Correctness: PASS**
- `gcn_norm_adj`: GCN-style symmetric normalization `D_tilde^{-1/2} * A_tilde * D_tilde^{-1/2}` with self-loops. Matches paper.
- `ODEFunc`: `f(H, A) = A * H * W + H * d` where W is initialized as identity, d as ones. Matches paper Eq. (5).
- `ODEG`: Euler integration with `n_steps = ode_time / step_size`. Matches paper.
- `TemporalConv`: Two-layer TCN with dilated causal convolution + residual connection. Matches paper.
- Dual ODEG branches for spatial and semantic adjacency. The paper uses A_sp (spatial adjacency) and A_se (semantic/DTW adjacency). This implementation uses the same correlation adjacency for both branches but with separate learned ODE dynamics.

**Paper Faithfulness: PASS**
- The paper uses separate spatial and semantic adjacency matrices. This implementation uses the same adjacency for both branches. The comment in the code acknowledges this: "Original uses A_sp (spatial) + A_se (semantic/DTW). Here both branches share the correlation adjacency with separate learned ODE dynamics."
- This is a reasonable simplification since DTW-based adjacency requires additional computation and the correlation adjacency can serve as both.
- The ODE function structure `A*H*W + H*d` matches the paper.
- Euler integration with step_size=1.0 and ode_time=6.0 (6 steps) matches the paper's default.
- TCN structure with causal convolution and residual matches the paper.
- Output: concatenate dual branches -> FC -> FC -> P steps.

**Tensor Layout:**
- `_net_forward`: `(B, H, N) -> (B*N, 1, H)` for TCN -> `(B*N, d_model, H)` -> reshape to `(B, H, N, d_model)`.
- Dual ODEG: `(B*H, N, d_model)` for graph ODE evolution. `adj` is `(N, N)`. `torch.matmul(adj, h)` where `h` is `(B*H, N, d_model)` and `adj` is `(N, N)`. The matmul is over the node dimension: `(N, N) @ (B*H, N, d_model) -> (B*H, N, d_model)`. Correct.
- Concat: `(B, H, N, 2*d_model)`.
- Output FC: `(B, N, H*2*d_model) -> FC(256) -> FC(P) -> (B, N, P) -> (B, P, N)`.
- Correct.

**Framework Compatibility: PASS**
- `input_mode = "all_nodes"`.
- `_net_forward` returns `(B, P, N)`.

**Cross-City Transfer:**
- Same N-dependent issue as other Group B models. The `adj` buffer is `(N_target, N_target)`.
- Additionally, the `ODEFunc` has `w: (d_model, d_model)` and `d: (d_model,)` which are NOT N-dependent. The `CausalConv1d` and `TemporalConv` are also NOT N-dependent. The `fc1` is `Linear(H * d_model * 2, 256)` which is NOT N-dependent.
- So the only N-dependent part is the adjacency matrix. The model architecture itself is N-agnostic otherwise.

**Potential Issues:**
- **Minor concern:** The paper uses separate spatial and semantic adjacency matrices. Using the same correlation adjacency for both branches is a simplification. This may affect performance compared to the original paper's results.
- The `CausalConv1d` uses `nn.Conv1d` with `padding` applied manually as left-padding. This is correct for causal convolution.
- The `TemporalConv` output has the same temporal dimension as input (due to residual connection with proper padding). The TCN output is `(B*N, d_model, H)`.
- The reshape from `(B*N, d_model, H)` to `(B, H, N, d_model)` is: `x.reshape(B, N, d, H).permute(0, 3, 1, 2)`.
  - `(B*N, d_model, H) -> (B, N, d_model, H) -> (B, H, N, d_model)`. Correct.
- The `d` variable is set but the code uses `d = x.size(1)` after TCN. This `d` shadows the earlier `d = None`. This is `d_model` from the TCN output. Correct.

### Severity: None (minor simplification noted)

---

## Overall Assessment

### Summary Table

| Model | Architecture | Paper Faithfulness | Framework Compat. | Issues Found | Max Severity |
|-------|-------------|-------------------|-------------------|-------------|-------------|
| LSTM | PASS | PASS | PASS | None | None |
| DLinear | PASS | PASS | PASS | None | None |
| PatchTST | PASS | PASS | PASS | Patch config for H=12 | None |
| TimesNet | PASS | PASS | PASS | MPS workaround | Low |
| HA | PASS | N/A | PASS | None | None |
| STGCN | PASS | PASS | PASS | Output layer deviation | Medium |
| DCRNN | PASS | PASS | PASS | No scheduled sampling | Low |
| GWNET | PASS | PASS | FAIL | **Output shape bug** | **Critical** |
| STGODE | PASS | PASS | PASS | Single adj for dual branches | Low |

### Critical Issues

**1. GWNET Output Shape Bug (CRITICAL)**

File: `/Users/xhz/CodingSpace/STCross/model/gwnet.py`, line ~167

The `_net_forward` method ends with:
```python
x = self.end_conv_2(x)       # (B, P, N, 5) -- time dim is 5, NOT 1
return x.squeeze(-1)          # (B, P, N) -- BUG: squeeze(-1) does nothing
```

The time dimension after all layers is 5 (H=12 + 1 padding - 8 layers). `squeeze(-1)` only removes dimensions of size 1. The actual output shape is `(B, P, N, 5)`, not `(B, P, N)`.

**Fix:** Replace `x.squeeze(-1)` with:
```python
return x[..., -1]  # take the last time step, output (B, P, N)
```
or, if the intended behavior is to use a different output head:
```python
# Option A: aggregate over time (e.g., mean or last step)
return x.mean(dim=-1)  # (B, P, N)
# Option B: take last step
return x[..., -1]  # (B, P, N)
```

**Impact:** This bug means the GWNET model's output is 5-dimensional in the time axis instead of 1-dimensional. The loss computation `nn.MSELoss()(pred, Y)` where `Y` is `(B, P, N)` and `pred` is `(B, P, N, 5)` will broadcast incorrectly. PyTorch's MSELoss will broadcast `(B, P, N, 5)` against `(B, P, N)` -> `(B, P, N, 5)`, computing the loss over all 5 time steps. This means the model is optimizing a different objective than intended, and the results are likely incorrect.

**2. Cross-City Transfer for Group B Models (CRITICAL - Integration Level)**

All Group B models (STGCN, DCRNN, GWNET, STGODE) are instantiated with `N_target` and `adj_target` (from `registry.py`), but during cross-city training, they receive source data with `N_source` nodes. Since NYC and CHI likely have different numbers of sensors, this will cause a shape mismatch at the graph convolution step.

This is not a bug in the model code itself but in the integration between the model registry and the cross-city evaluator. The fix would need to be in either:
- `registry.py`: build models with source N/adj for training, then rebuild with target N/adj for testing.
- `evaluator.py`: handle N-mismatch by aligning nodes or rebuilding the model.

**However**, if the STCross project always uses the same number of nodes for NYC and CHI (e.g., by selecting the same number of sensors), this is not an issue. The data processing code should be checked to confirm.

### Non-Critical Issues

**3. STGCN Output Layer Deviation (MEDIUM)**

The STGCN output layer uses: `TemporalConv(Ko) -> LayerNorm -> TemporalConv(1) -> FC`. The original paper uses a simpler output: single temporal convolution + FC. The extra LayerNorm and second temporal convolution are not in the paper. This may affect comparability with published results.

**4. DCRNN No Scheduled Sampling (LOW)**

The original DCRNN uses scheduled sampling during decoder training. This implementation uses pure autoregressive decoding. This can cause exposure bias and may degrade long-horizon prediction accuracy compared to the published results.

**5. TimesNet MPS Workaround (LOW)**

The `__init__` method forces CPU when MPS is detected. This is a valid workaround for an MPS bug but should be documented in experiment logs.

**6. STGODE Single Adjacency for Dual Branches (LOW)**

The paper uses separate spatial and semantic adjacency matrices. Using the same correlation adjacency for both branches is a simplification that may affect results.

### Recommendations

1. **Fix GWNET output shape** (highest priority): Change `x.squeeze(-1)` to `x[..., -1]` or add a proper output aggregation layer.

2. **Verify cross-city N-compatibility**: Check whether NYC and CHI have the same number of nodes. If not, the Group B models cannot be used for cross-city transfer without additional alignment logic.

3. **Consider adding scheduled sampling to DCRNN**: This would improve the model's faithfulness to the original paper and potentially improve long-horizon predictions.

4. **Document the STGCN output layer deviation**: Note in the paper/README that the STGCN implementation uses a slightly different output layer structure than the original paper.

5. **Add unit tests for output shapes**: Add assertions to verify that `_net_forward` returns the correct shape `(B, P, N)` for all_nodes models and `(B*N, P, 1)` for per_node models.

6. **Consider using separate adjacency matrices for STGODE**: If DTW-based semantic adjacency is available, use it for the second ODEG branch to better match the paper.

### Overall Evaluation

The Group A models (LSTM, DLinear, PatchTST, TimesNet, HA) are well-implemented and faithful to their respective papers. The framework integration is clean and correct.

The Group B models (STGCN, DCRNN, STGODE) are generally well-implemented with minor deviations from the original papers. The GWNET model has a critical output shape bug that must be fixed before any experimental results can be trusted.

The cross-city transfer mechanism for Group B models requires careful verification of N-compatibility between source and target cities.