# ProtoDRIFT — Reproduction Package

This folder is a self-contained reproduction package for **ProtoDRIFT**
(Prototype-Disentangled Deviation-Referenced Invariant Forecasting for
transferable cross-city anomalous urban flow forecasting).

> **Note.** `HA` is also registered in `model/registry.py`, but only because
> the evaluation pipeline uses historical averages to build normal
> references; it is not part of the method and not a compared baseline.

Once the six data files listed in Section 3 are placed under
`data/processed_data/`, a single command reproduces the full
train-and-evaluate pipeline for both transfer directions (Section 4).

---

## 1. Directory layout

```
ProtoDRIFT/
├── README.md                  # this file
├── requirements.txt
├── data/processed_data/       # <- put the data files here (empty on purpose)
│   ├── traffic_flow/
│   ├── holiday/
│   └── weather/
├── data_preprocess/           # dataset loading, normalization, sliding windows
│   ├── dataset.py             #   flow/mask loading, splits, window construction
│   ├── normalizer.py          #   per-node z-score scaler (train-fitted, clipped)
│   ├── torch_dataset.py       #   PyTorch dataloader construction
│   └── build_*.py             #   optional: rebuild the .npy files from raw data
├── model/
│   ├── protodrift.py          # ProtoDRIFT (all components, ablation switches)
│   ├── torch_base.py          # shared PyTorch model wrapper (fit/predict/ckpt)
│   ├── base.py                # abstract model interface
│   ├── ha.py                  # historical-average lookup (pipeline dependency)
│   └── registry.py            # model factory (ProtoDRIFT + HA only)
├── train/trainer.py           # training loop, seeding, LR schedule, early stop
├── eval/
│   ├── config.py              # all data paths and split constants
│   ├── evaluator.py           # train-once / evaluate-per-target protocol
│   └── reporter.py            # metric printing and JSON export
├── metrics/metrics.py         # MAE / RMSE / MAPE / PCC / R2
└── experiments/run_all.py     # single entry point for all experiments
```

Three directories are created automatically at runtime and are therefore
not shipped: `checkpoints/` (best validation checkpoint per source city),
`results/` (JSON metrics and summaries), and `logs/` (training logs).

## 2. Environment

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `tqdm`, `torch` (>= 2.3).
The device is selected automatically in the order
CUDA > Apple MPS > CPU; no flags are required.

## 3. Data preparation

### 3.1 Required files (recommended path)

Place the following **six** preprocessed NumPy files (provided with our data
release) at exactly these paths. All arrays cover 2014-10-01 to 2017-09-29
at a 60-minute granularity (T = 26280 hourly steps).

| Path | Shape | Content |
|---|---|---|
| `data/processed_data/traffic_flow/flow_NYC_2014-10-01_2017-09-30_60min.npy` | (26280, 820) int32 | NYC hourly bike flow per node |
| `data/processed_data/traffic_flow/flow_CHI_2014-10-01_2017-09-30_60min.npy` | (26280, 585) int32 | CHI hourly bike flow per node |
| `data/processed_data/holiday/mask_NYC_holiday_2014-10-01_2017-09-30_60min.npy` | (26280,) int8 | NYC holiday step mask |
| `data/processed_data/holiday/mask_CHI_holiday_2014-10-01_2017-09-30_60min.npy` | (26280,) int8 | CHI holiday step mask |
| `data/processed_data/weather/mask_NYC_all_2014-10-01_2017-09-30_60min.npy` | (26280,) int8 | NYC extreme-weather step mask |
| `data/processed_data/weather/mask_CHI_all_2014-10-01_2017-09-30_60min.npy` | (26280,) int8 | CHI extreme-weather step mask |

The anomaly mask of a step is the logical OR of the holiday and weather
masks; a sliding-window sample belongs to the *anomaly* subset when at least
half of its 12-step prediction window is anomalous (`anomaly_min_ratio=0.5`
in `eval/config.py`).

### 3.2 Rebuilding from raw data (optional)

All raw data originate from the publicly available STContext benchmark
(Chen et al., "STContext: A Multifaceted Dataset for Developing
Context-Aware Spatio-Temporal Crowd Mobility Prediction Models",
arXiv:2501.03583, 2025; also cited in the paper), which provides the
bike-sharing records and the historical weather files consumed by the
scripts below. If you start from that raw release instead of the
preprocessed `.npy` files, the scripts under `data_preprocess/` regenerate
the same files:

```bash
# raw bike records ->  flow_*.npy
#   expects data/raw_data/traffic_flow/Bike_NYC.pkl and Bike_Chicago.pkl
python data_preprocess/build_flow_data.py

# holiday calendar ->  mask_*_holiday_*.npy
#   uses a self-contained U.S. federal + city holiday calendar (no extra data needed)
python data_preprocess/build_holiday_mask.py

# raw weather records ->  mask_*_all_*.npy
#   expects data/raw_data/weather_his/*HistoricalWeather.csv
python data_preprocess/build_weather_mask.py
```

### 3.3 Fixed experimental protocol

These constants live in `eval/config.py` and match the paper; you do not
need to change them:

- Chronological split: train 15336 steps (2014-10 to 2016-06), validation
  5160 steps (2016-07 to 2017-01), test 5784 steps (2017-02 to 2017-09).
- Windows: H = 12 past steps predict P = 12 future steps.
- Normalization: per-node z-score fitted on each city's training split,
  clipped to [-10, 10].

## 4. Reproducing the results

All commands are run from this folder's root. The entry point
`experiments/run_all.py` trains once per source city and evaluates on both
cities, covering the four settings NYC→NYC, NYC→CHI, CHI→CHI, CHI→NYC.

### 4.1 Main results (paper Table 1, ProtoDRIFT row)

All CLI defaults already match the paper configuration
(`alpha=0.3, K=8, gamma=0.1, beta=0.01`, seed 42, early-stopping patience 10),
so the plain command reproduces the reported numbers:

```bash
python experiments/run_all.py --models ProtoDRIFT --settings all --output_dir results/main
```

The equivalent fully explicit form is:

```bash
python experiments/run_all.py \
    --models ProtoDRIFT --settings all \
    --seed 42 --early_stop_patience 10 \
    --protodrift_alpha 0.3 --protodrift_k 8 --protodrift_gamma 0.1 --protodrift_beta 0.01 \
    --output_dir results/main
```

Training uses Adam with cosine annealing and early stopping; with patience
10 it terminates long before the `--epochs 5000` cap. Cached results are
reused when a matching JSON already exists in `--output_dir`; add `--force`
to retrain from scratch.

### 4.2 Ablation study (paper Fig. "Ablation")

Each variant disables one component; use a separate output directory per
variant so cached results do not collide:

```bash
# w/o DevSpace: predict in the absolute-value space (alpha is meaningless there)
python experiments/run_all.py --models ProtoDRIFT --settings all --seed 42 \
    --early_stop_patience 15 --force --no_devspace \
    --protodrift_k 8 --protodrift_alpha 0 --protodrift_gamma 0.1 --protodrift_beta 0.01 \
    --output_dir results/A1_no_devspace

# w/o ProtoBank: replace prototype retrieval with a plain FC head
python experiments/run_all.py --models ProtoDRIFT --settings all --seed 42 \
    --early_stop_patience 10 --force --no_proto \
    --protodrift_k 8 --protodrift_alpha 0.3 --protodrift_gamma 0.1 --protodrift_beta 0.01 \
    --output_dir results/A2_no_proto

# w/o CityBranch: remove the city-conditioned encoder/decoder and MI constraint
python experiments/run_all.py --models ProtoDRIFT --settings all --seed 42 \
    --early_stop_patience 10 --force --no_city_branch \
    --protodrift_k 8 --protodrift_alpha 0.3 --protodrift_gamma 0.1 --protodrift_beta 0.01 \
    --output_dir results/A3_no_citybranch
```

### 4.3 Sensitivity sweeps (paper Fig. "Sensitivity")

One-at-a-time sweeps around the selected configuration, e.g. for alpha:

```bash
for A in 0.02 0.05 0.1 0.3; do
python experiments/run_all.py --models ProtoDRIFT --settings cross-city --seed 42 \
    --early_stop_patience 10 --force \
    --protodrift_alpha $A --protodrift_k 8 --protodrift_gamma 0.1 --protodrift_beta 0.01 \
    --output_dir results/sens_alpha_$A
done
```

Sweep `--protodrift_k` over {4, 8, 16, 32} and `--protodrift_gamma` over
{0.05, 0.1, 0.2, 0.4} analogously.

## 5. Outputs

All output directories are created on first use.

- `results/<dir>/ProtoDRIFT_<SRC>_to_<TGT>_<timestamp>.json` — metrics
  (MAE / RMSE / MAPE / PCC / R2) for the `full`, `anomaly`, and `normal`
  test subsets, plus the exact configuration used.
- `results/<dir>/summary.md` — an aggregated table over the runs in that
  directory (written by `run_all.py` at the end).
- `checkpoints/ProtoDRIFT_<SRC>_best.pt` — best validation checkpoint per
  source city; reuse with `--eval_only` to skip retraining.
- `logs/` — per-run training logs.

## 6. Expected results

Paper configuration (`alpha=0.3, K=8, gamma=0.1, beta=0.01`, seed 42,
patience 10), zero-shot cross-city transfer:

| Setting | Subset | MAE | RMSE | R2 |
|---|---|---|---|---|
| NYC→CHI | full | 0.6988 | 1.8742 | 0.5867 |
| NYC→CHI | anomaly | 0.6292 | 1.7539 | 0.4679 |
| CHI→NYC | full | 1.7061 | 3.6897 | 0.5127 |
| CHI→NYC | anomaly | 1.4906 | 3.1140 | 0.4309 |

Exact values vary slightly across hardware and runs even with a fixed seed
(GPU non-determinism); in our environment, repeated runs of the identical
configuration differed by roughly ±0.015 MAE / ±0.017 RMSE / ±0.010 R2 on
the NYC→CHI anomaly subset. Deviations within this range are expected.

## 7. Method-to-code map

| Paper | Code (`model/protodrift.py`) | CLI flag |
|---|---|---|
| Normal reference B, deviation space | NBL (normal baseline lookup), `use_deviation_space` | `--no_devspace` disables |
| Anomaly encoder + capacity term (alpha) | variational LSTM `AEnc`, `L_cap` | `--protodrift_alpha` |
| Prototype bank (K), regularizer (gamma) | `AnomalyPrototypeBank`, `L_proto` | `--protodrift_k`, `--protodrift_gamma` |
| City-conditioned branch | `CEnc` + `CAD` | `--no_city_branch` disables |
| MI constraint (beta) | `MIEstimator`, `L_mic` | `--protodrift_beta` |
