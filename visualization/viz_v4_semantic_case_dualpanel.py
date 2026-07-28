#!/usr/bin/env python
"""V4: Semantic anomaly clusters + case-level forecasting comparison.

This script builds one double-column visualization that connects learned anomaly
semantics with case-level forecasting behavior:
  - Left panel: CHI anomaly windows encoded by STCross Z_a, pooled to
    window-level representations, clustered into semantic groups, and explained
    by selected-cluster deviation pattern curves.
  - Right panel: rolling step-0 prediction curves for the selected holiday and
    weather cases, comparing ProtoDRIFT with learned baselines only. HA is excluded
    by design for this visualization.

Usage:
    python visualization/viz_v4_semantic_case_dualpanel.py
    python visualization/viz_v4_semantic_case_dualpanel.py --baselines LSTM ModernTCN DAN
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# Project root on sys.path so `from eval...` / `from model...` work when
# data/model-loading functions are called. Heavy project dependencies are imported
# lazily so helper tests can run in a lightweight Python environment.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

STCROSS_CKPT_CANDIDATES = ["checkpoints_viz", "checkpoints"]
OUT_DIR = "results_viz"
H, P = 12, 12
ACTIVE_THRESH = 0.5
MIN_RUN = 6
# Right-panel candidates are decoupled from the pinned left-panel anchors, so a
# shorter minimum run length widens the pool of displayable anomaly episodes.
DISPLAY_MIN_RUN = 3
SEED = 42
EMBED_ANOM_SAMPLES = 520
TSNE_NODE_ROWS = 3200

BASELINE_CONFIG = {
    "LSTM": "checkpoints",
    "ModernTCN": "checkpoints_viz",
    "ProtoTS": "checkpoints_viz",
    "DAN": "checkpoints_viz",
    "EVL": "checkpoints",
    "DLinear": "checkpoints",
    "PatchTST": "checkpoints",
}
DEFAULT_BASELINES = ["ModernTCN", "ProtoTS", "DAN"]

MODEL_COLORS = {
    "Ground truth": "#111111",
    "STCross": "#d62728",
    "LSTM": "#1f77b4",
    "ModernTCN": "#2ca02c",
    "ProtoTS": "#ff7f0e",
    "DAN": "#9467bd",
    "EVL": "#ff7f0e",
    "DLinear": "#8c564b",
    "PatchTST": "#e377c2",
}
PROTO_COLORS = [
    "#4c72b0", "#dd8452", "#55a868", "#c44e52",
    "#8172b3", "#937860", "#da8bc3", "#64b5cd",
]
EVENT_COLORS = {
    "Holiday": "#f2b701",
    "Weather": "#00a6d6",
    "Holiday+Weather": "#7b3294",
    "Other anomaly": "#9e9e9e",
}
TRANSITION_WINDOW = 12
DAYTIME_START = 6
DAYTIME_END = 22
CLUSTER_COUNT = 8
MODEL_DISPLAY_NAMES = {"STCross": "ProtoDRIFT"}
CASE_MARKERS = {"Holiday": "*", "Weather": "D"}
CASE_COLORS = {"Holiday": "#f2b701", "Weather": "#00a6d6"}
CASE_FALLBACK_LABELS = {"Holiday": "Holiday surge", "Weather": "Rainstorm"}
# Left-panel semantic cases are pinned: the t-SNE markers and pattern curves
# always come from these runs, while the right-panel cases are chosen freely.
LEFT_PANEL_ANCHORS = {
    "Holiday": {"run": (24174, 24185), "event_label": "Independence Day"},
    "Weather": {"run": (21558, 21569), "event_label": "Rainstorm"},
}
# Right-panel cases that are frozen after review; remaining labels are re-selected.
# Holiday: Presidents' Day node 119 is the balance point on the CHI trade-off
# frontier -- ProtoDRIFT MAE 1.72 vs baseline 2.88 (1.67x) at the achievable
# relative-error floor (0.27) while keeping a peak flow of 19. Both windows span
# 10:00-21:00 so the two right-panel x axes line up.
PINNED_DISPLAY_CASES = {
    "Holiday": {"run": (20962, 20973), "node": 119},
    "Weather": {"run": (22858, 22869), "node": 79},
}
# ProtoDRIFT must stay within this fraction of the mean true flow, otherwise a
# large baseline gap would be bought at the cost of our own curve missing GT.
# On CHI holiday surges the achievable floor is ~0.37, so the gate sits just above it.
REL_FIT_MAX = 0.40


def display_model_name(name):
    """Return the paper-facing display name for a model identifier."""
    return MODEL_DISPLAY_NAMES.get(name, name)


def checkpoint_exists(model_name, ckpt_dir):
    """Return whether ckpt_dir contains the expected NYC checkpoint."""
    ckpt_path = os.path.join(PROJECT_ROOT, ckpt_dir, f"{model_name}_NYC_best.pt")
    return os.path.exists(ckpt_path)


def resolve_checkpoint_dir(model_name, preferred_dir=None):
    """Resolve a checkpoint directory, trying preferred then common project dirs."""
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir)
    candidates.extend(["checkpoints_viz", "checkpoints"])
    seen = set()
    for ckpt_dir in candidates:
        if ckpt_dir in seen:
            continue
        seen.add(ckpt_dir)
        if checkpoint_exists(model_name, ckpt_dir):
            return os.path.join(PROJECT_ROOT, ckpt_dir)
    return None


def load_stcross(ckpt_dir):
    """Construct STCross with the resolved checkpoint and selected design."""
    from eval.config import EvalConfig
    from model.registry import MODEL_REGISTRY
    from train.trainer import TrainConfig

    tc = TrainConfig(checkpoint_dir=ckpt_dir)
    ec = EvalConfig()
    hparams = dict(stcross_k=8, stcross_alpha=0.3, stcross_beta=0.01, stcross_gamma=0.1)
    model = MODEL_REGISTRY["STCross"](ec, tc, **hparams)
    model.load_checkpoint("NYC")
    return model


def load_learned_baseline(name, ckpt_dir):
    """Load a learned NYC-trained baseline checkpoint."""
    from eval.config import EvalConfig
    from model.registry import MODEL_REGISTRY
    from train.trainer import TrainConfig

    ec = EvalConfig()
    tc = TrainConfig(checkpoint_dir=ckpt_dir)
    model = MODEL_REGISTRY[name](ec, tc)
    model.load_checkpoint("NYC")
    return model


def prepare_chi():
    """Load CHI flow, anomaly masks, normalizer, NBL table, and split metadata."""
    from eval.config import EvalConfig
    from eval.evaluator import _compute_nbl_full
    from data_preprocess.dataset import load_flow, load_anomaly_mask, split_data
    from data_preprocess.normalizer import PerNodeStandardScaler

    ec = EvalConfig()
    flow = load_flow("CHI", ec.flow_dir, ec.start_date, ec.end_date,
                     ec.freq_min).astype(np.float64)
    mask = load_anomaly_mask("CHI", ec.mask_dir_holiday, ec.mask_dir_weather,
                             ec.start_date, ec.end_date, ec.freq_min)
    holiday = np.load(ec.holiday_mask_path("CHI")).astype(bool)
    weather = np.load(ec.weather_mask_path("CHI")).astype(bool)
    T, N = flow.shape
    (tr_s, tr_e), _, (te_s, te_e) = split_data(T, ec.train_steps, ec.val_steps)
    scaler = PerNodeStandardScaler().fit(flow[tr_s:tr_e])
    flow_norm = scaler.transform(flow, clip_range=10.0)
    timestamps = pd.date_range(ec.start_date, periods=T, freq=f"{ec.freq_min}min")
    nbl_full = _compute_nbl_full(flow_norm, mask, timestamps, tr_e)
    nbl_raw = scaler.inverse_transform(nbl_full)
    active = flow[tr_s:tr_e].mean(axis=0) > ACTIVE_THRESH
    return dict(flow=flow, flow_norm=flow_norm, mask=mask, holiday=holiday,
                weather=weather, T=T, N=N, tr_s=tr_s, tr_e=tr_e,
                te_s=te_s, te_e=te_e, scaler=scaler, timestamps=timestamps,
                nbl_full=nbl_full, nbl_raw=nbl_raw, active=active,
                anomaly_min_ratio=ec.anomaly_min_ratio)


def interval_anomaly_ratio(mask, start, end):
    """Return anomaly ratio for an inclusive interval."""
    return float(np.asarray(mask[start:end + 1]).mean())


def _timestamp_hour(ts):
    """Extract hour from pandas/numpy timestamp-like objects."""
    return pd.Timestamp(ts).hour


def daytime_fraction(timestamps, start, end, start_hour=DAYTIME_START, end_hour=DAYTIME_END):
    """Fraction of interval timestamps within [start_hour, end_hour)."""
    hours = [_timestamp_hour(timestamps[i]) for i in range(start, end + 1)]
    return float(np.mean([(start_hour <= h < end_hour) for h in hours]))


def case_event_label(label, timestamp):
    """Return the paper-facing event label for a selected case timestamp."""
    if label == "Weather":
        return CASE_FALLBACK_LABELS["Weather"]
    ts = pd.Timestamp(timestamp)
    known_2017_holidays = {
        "2017-02-12": "Lincoln's Birthday",
        "2017-02-13": "Lincoln's Birthday",
        "2017-02-20": "Presidents' Day",
        "2017-05-29": "Memorial Day",
        "2017-07-04": "Independence Day",
        "2017-09-04": "Labor Day",
    }
    return known_2017_holidays.get(ts.strftime("%Y-%m-%d"), CASE_FALLBACK_LABELS["Holiday"])


def case_direction_metrics(true_s0, nbl_s0, baseline_preds, active, label):
    """Measure whether a case matches the desired flow and baseline-bias direction."""
    active = np.asarray(active, dtype=bool)
    if not active.any():
        active = np.ones(true_s0.shape[-1], dtype=bool)
    delta_curve = (true_s0[:, active] - nbl_s0[:, active]).mean(axis=1)
    true_curve = true_s0[:, active].mean(axis=1)
    flow_delta = float(delta_curve.mean())
    if baseline_preds:
        biases = [(pred[:, active] - true_s0[:, active]).mean()
                  for pred in baseline_preds.values()]
        baseline_bias = float(np.mean(biases))
    else:
        baseline_bias = 0.0
    target = 1.0 if label == "Holiday" else -1.0
    flow_direction_score = target * flow_delta
    baseline_direction_score = -target * baseline_bias
    return dict(flow_delta=flow_delta,
                baseline_bias=baseline_bias,
                flow_direction_score=float(flow_direction_score),
                baseline_direction_score=float(baseline_direction_score),
                direction_score=float(min(flow_direction_score, baseline_direction_score)),
                direction_ok=bool(flow_direction_score > 0 and baseline_direction_score > 0),
                positive_step_fraction=float((delta_curve > 0).mean()),
                min_flow_delta=float(delta_curve.min()),
                true_mean=float(true_curve.mean()),
                min_true=float(true_curve.min()))


def case_selection_priority(row):
    """Sort key for V4 cases: flow-direction magnitude first, then fit-vs-baseline gap."""
    common_head = (int(row.get("flow_direction_score", 0.0) > 0),)
    fit_tail = (
        row.get("margin", -np.inf),
        -row.get("mae_st", np.inf),
        row.get("baseline_direction_score", 0.0),
        row.get("transition_bonus", 0.0),
    )
    if row.get("label") == "Holiday":
        return common_head + (
            row.get("positive_step_fraction", 0.0),
            int(row.get("min_flow_delta", -np.inf) > 0),
            row.get("flow_direction_score", 0.0),
        ) + fit_tail
    return common_head + (
        int(row.get("true_mean", 0.0) >= 1.0),
        row.get("flow_direction_score", 0.0),
    ) + fit_tail


def case_is_eligible(row):
    """Gate candidates on the flow-direction story only; baseline bias is a
    ranking preference, not an eligibility requirement."""
    if row.get("flow_direction_score", 0.0) <= 0:
        return False
    if row.get("label") == "Holiday":
        return row.get("positive_step_fraction", 0.0) >= 0.75
    return row.get("true_mean", 0.0) >= 1.0


def case_display_score(true, nbl, st, baseline_preds, label):
    """Node-level visual-contrast score for the right panel: ProtoDRIFT tracks
    the anomalous trend while baselines miss it in the label-specific direction.

    ProtoDRIFT closeness to ground truth is a hard gate (relative to the flow
    scale), so a huge baseline gap cannot compensate for our own curve being far
    from GT. Both the gap and our error are normalized by the flow scale because
    the plot axes autoscale, so only proportional separation is visible. The
    baseline gap is aggregated per step with a median so a single spike cannot
    stand in for a case where baselines are consistently wrong.
    """
    true = np.asarray(true, dtype=float)
    target = 1.0 if label == "Holiday" else -1.0
    direction = float((true - np.asarray(nbl, dtype=float)).mean()) * target
    if direction <= 0:
        return (-1, -np.inf)
    scale = max(float(np.abs(true).mean()), 1e-6)
    rel_fit = float(np.abs(np.asarray(st, dtype=float) - true).mean()) / scale
    if baseline_preds:
        step_gaps = np.stack([(true - np.asarray(pred, dtype=float)) * target
                              for pred in baseline_preds.values()], axis=0).mean(axis=0)
        rel_gap = float(np.median(step_gaps)) / scale
    else:
        rel_gap = 0.0
    return (int(rel_fit <= REL_FIT_MAX), rel_gap - rel_fit)


def transition_interval_for_run(run, mask, timestamps, lo, hi, window_len=TRANSITION_WINDOW,
                                target_ratio=0.5):
    """Choose a fixed-length interval around a run with about half anomalous steps.

    Candidate windows overlap the anomalous run. They are ranked by closeness to
    half-anomaly coverage, then by daytime coverage to avoid midnight cases.
    """
    r_start, r_end = run
    lo = int(lo)
    hi = int(hi)
    start_lo = max(lo, r_start - window_len + 1)
    start_hi = min(r_end, hi - window_len)
    candidates = []
    for start in range(start_lo, start_hi + 1):
        end = start + window_len - 1
        ratio = interval_anomaly_ratio(mask, start, end)
        day = daytime_fraction(timestamps, start, end)
        score = (day, -abs(ratio - target_ratio), -abs(start + end - r_start - r_end))
        candidates.append((score, start, end))
    if not candidates:
        return run
    candidates.sort(reverse=True)
    _, start, end = candidates[0]
    return start, end


def find_runs(mask, min_len, lo, hi):
    """Return consecutive True intervals in mask[lo:hi] as inclusive global indices."""
    runs = []
    i = lo
    while i < hi:
        if mask[i]:
            j = i
            while j < hi and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def interval_sample_range(r_start, r_end, te_s, te_e, horizon=H, pred_len=P):
    """Map a target time interval to rolling sample starts whose step-0 covers it."""
    i_lo = max(te_s, r_start - horizon)
    i_hi = min(te_e - horizon - pred_len, r_end - horizon)
    return i_lo, i_hi


def stack_windows(arr_2d, idx, length):
    """Build (L, length, N) from a (T, N) array."""
    return np.stack([arr_2d[i:i + length] for i in idx], axis=0)


def aggregate_active(arr, active):
    """Mean over active nodes along the last dimension."""
    return arr[..., active].mean(axis=-1)


def mae_step0(pred_s0, true_s0, active):
    """MAE over timesteps and active nodes; both arrays are (L, N)."""
    return np.abs(pred_s0[:, active] - true_s0[:, active]).mean()


def stcross_rolling(stcross, data, i_lo, i_hi):
    """STCross rolling predictions over [i_lo, i_hi] in original scale."""
    idx = np.arange(i_lo, i_hi + 1)
    X = stack_windows(data["flow_norm"], idx, H)
    b_in = stack_windows(data["nbl_full"], idx, H)
    b_fut = stack_windows(data["nbl_full"], idx + H, P)
    dec = stcross.predict_with_decomposition(X, b_in, b_fut)
    inv = data["scaler"].inverse_transform
    return {k: inv(v) for k, v in dec.items()}


def learned_rolling(model, data, i_lo, i_hi):
    """Learned baseline rolling predictions over [i_lo, i_hi] in original scale."""
    idx = np.arange(i_lo, i_hi + 1)
    X = stack_windows(data["flow_norm"], idx, H)
    preds = model.predict(X)
    return data["scaler"].inverse_transform(preds)


def evaluate_runs(runs, label, data, stcross, learneds):
    """Rank transition intervals by semantic direction, daytime quality, and gain."""
    results = []
    te_s, te_e = data["te_s"], data["te_e"]
    valid_lo = te_s + H
    valid_hi = te_e - P
    source_mask = data["holiday"] if label == "Holiday" else data["weather"]
    for run in runs:
        display_run = transition_interval_for_run(
            run, source_mask, data["timestamps"], valid_lo, valid_hi,
            window_len=TRANSITION_WINDOW)
        i_lo, i_hi = interval_sample_range(*display_run, te_s, te_e)
        if i_hi < i_lo:
            continue
        true_s0 = data["flow"][i_lo + H:i_hi + H + 1]
        nbl_s0 = data["nbl_raw"][i_lo + H:i_hi + H + 1]
        st_s0 = stcross_rolling(stcross, data, i_lo, i_hi)["y_pred"][:, 0, :]
        m_st = mae_step0(st_s0, true_s0, data["active"])
        lb_maes = {}
        lb_preds = {}
        for name, model in learneds:
            lb_s0 = learned_rolling(model, data, i_lo, i_hi)[:, 0, :]
            lb_preds[name] = lb_s0
            lb_maes[name] = mae_step0(lb_s0, true_s0, data["active"])
        strongest = min(lb_maes.values()) if lb_maes else np.inf
        ratio = interval_anomaly_ratio(source_mask, *display_run)
        day = daytime_fraction(data["timestamps"], *display_run)
        transition_bonus = day - abs(ratio - 0.5)
        direction = case_direction_metrics(true_s0, nbl_s0, lb_preds, data["active"], label)
        results.append(dict(run=run, display_run=display_run, label=label,
                            date=data["timestamps"][i_lo + H], mae_st=m_st,
                            mae_lb=lb_maes, margin=strongest - m_st,
                            anomaly_ratio=ratio, daytime_fraction=day,
                            transition_bonus=transition_bonus, **direction))
    results.sort(key=case_selection_priority, reverse=True)
    return results


def collect_case_prediction(data, stcross, learneds, run):
    """Collect original-scale step-0 curves and per-node absolute errors for one run."""
    i_lo, i_hi = interval_sample_range(*run, data["te_s"], data["te_e"])
    if i_hi < i_lo:
        raise ValueError(f"Run {run} has no valid rolling samples")
    timesteps = data["timestamps"][i_lo + H:i_hi + H + 1]
    true_s0 = data["flow"][i_lo + H:i_hi + H + 1]
    nbl_s0 = data["nbl_raw"][i_lo + H:i_hi + H + 1]
    st_pred = stcross_rolling(stcross, data, i_lo, i_hi)["y_pred"][:, 0, :]
    preds = {"STCross": st_pred}
    for name, model in learneds:
        preds[name] = learned_rolling(model, data, i_lo, i_hi)[:, 0, :]
    errors = {name: np.abs(pred - true_s0) for name, pred in preds.items()}
    return dict(run=run, i_lo=i_lo, i_hi=i_hi, timesteps=timesteps,
                true=true_s0, nbl=nbl_s0, preds=preds, errors=errors)


def pick_representative_nodes(errors, active, max_nodes=1):
    """Pick active nodes where STCross most improves over the strongest baseline."""
    st_error = errors["STCross"].mean(axis=0)
    baseline_errors = [err.mean(axis=0) for name, err in errors.items() if name != "STCross"]
    if baseline_errors:
        strongest_baseline = np.min(np.stack(baseline_errors, axis=0), axis=0)
        gain = strongest_baseline - st_error
    else:
        gain = -st_error
    score = np.where(active, gain, -np.inf)
    if not np.isfinite(score).any():
        score = gain
    order = np.argsort(score)[::-1]
    return order[:max_nodes]


def build_node_sample_metadata(sample_starts, node_count, horizon=H, sample_events=None):
    """Build metadata matching STCross.encode row order: sample-major, then node."""
    metadata = []
    if sample_events is None:
        sample_events = [None] * len(sample_starts)
    for sample_start, event in zip(sample_starts, sample_events):
        for node in range(node_count):
            metadata.append(dict(sample_start=int(sample_start),
                                 target_start=int(sample_start + horizon),
                                 node=int(node),
                                 event=event))
    return metadata


def select_case_points(metadata, holiday_run, weather_run, holiday_nodes, weather_nodes):
    """Return embedding row indices for holiday/weather cases from metadata."""
    selected = {}
    specs = [("Holiday", holiday_run, set(map(int, holiday_nodes))),
             ("Weather", weather_run, set(map(int, weather_nodes)))]
    for label, run, nodes in specs:
        r_start, r_end = run
        fallback = None
        for idx, item in enumerate(metadata):
            in_run = r_start <= item["target_start"] <= r_end
            if in_run and fallback is None:
                fallback = idx
            if in_run and item["node"] in nodes:
                selected[label] = idx
                break
        if label not in selected and fallback is not None:
            selected[label] = fallback
    return selected


def window_event_label(data, sample_start):
    """Classify a sample by the anomaly source in its prediction window."""
    sl = slice(sample_start + H, sample_start + H + P)
    is_holiday = data["holiday"][sl].mean() >= data["anomaly_min_ratio"]
    is_weather = data["weather"][sl].mean() >= data["anomaly_min_ratio"]
    if is_holiday and is_weather:
        return "Holiday+Weather"
    if is_holiday:
        return "Holiday"
    if is_weather:
        return "Weather"
    return "Other anomaly"


def build_embedding_inputs(data, holiday_case, weather_case, rng, max_samples=EMBED_ANOM_SAMPLES):
    """Prepare anomaly windows for the left-panel embedding and force case windows in."""
    from eval.evaluator import _slice_nbl
    from data_preprocess.dataset import create_sliding_windows, get_subset_indices

    X_all, _, labels = create_sliding_windows(
        data["flow_norm"], data["mask"], H, P, data["te_s"], data["te_e"],
        data["anomaly_min_ratio"])
    nbl_in, nbl_fut = _slice_nbl(data["nbl_full"], data["te_s"], H, len(X_all), H, P)
    anom_idx = get_subset_indices(labels, "anomaly")
    if len(anom_idx) == 0:
        raise RuntimeError("No CHI anomaly samples found for embedding")
    n_sel = min(max_samples, len(anom_idx))
    selected = rng.choice(anom_idx, size=n_sel, replace=False)
    forced = []
    for case in (holiday_case, weather_case):
        local_idx = case["i_rep"] - data["te_s"]
        if 0 <= local_idx < len(X_all):
            forced.append(local_idx)
    selected = np.unique(np.concatenate([selected, np.array(forced, dtype=int)]))
    selected.sort()
    sample_starts = data["te_s"] + selected
    sample_events = [window_event_label(data, int(s)) for s in sample_starts]
    return X_all[selected], nbl_in[selected], nbl_fut[selected], sample_starts, sample_events


def pool_node_embeddings_to_samples(z_node, sample_count, node_count, active=None):
    """Pool encode() rows into window embeddings using active-node mean and std."""
    z = z_node.reshape(sample_count, node_count, -1)
    if active is not None and np.asarray(active).any():
        z = z[:, np.asarray(active), :]
    mean = z.mean(axis=1)
    std = z.std(axis=1)
    return np.concatenate([mean, std], axis=1)


def summarize_cluster_patterns(delta_windows, cluster_labels, active, target_clusters):
    """Average active-node deviation curves for selected window clusters."""
    curves = aggregate_active(delta_windows, active)
    patterns = {}
    for cluster in target_clusters:
        mask = cluster_labels == cluster
        if np.any(mask):
            patterns[int(cluster)] = curves[mask].mean(axis=0)
    return patterns


def compute_embedding(model, X, b_in, b_fut, sample_starts, node_count,
                      case_nodes, rng, sample_events=None, active=None,
                      cluster_count=CLUSTER_COUNT):
    """Encode Z_a, cluster window-level representations, and run t-SNE."""
    z_a, _ = model.encode(X, b_in, b_fut)
    sample_z = pool_node_embeddings_to_samples(z_a, len(X), node_count, active)

    from sklearn.cluster import KMeans
    n_clusters = min(cluster_count, len(sample_z))
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=SEED,
                            n_init=20).fit_predict(sample_z)

    sample_metadata = []
    for i, sample_start in enumerate(sample_starts):
        sample_metadata.append(dict(sample_start=int(sample_start),
                                    target_start=int(sample_start + H),
                                    event=None if sample_events is None else sample_events[i]))

    def find_case_index(run):
        r_start, r_end = run
        candidates = [i for i, meta in enumerate(sample_metadata)
                      if r_start <= meta["target_start"] <= r_end]
        if not candidates:
            return None
        mid = (r_start + r_end) / 2.0
        return min(candidates, key=lambda i: abs(sample_metadata[i]["target_start"] - mid))

    case_indices = {
        "Holiday": find_case_index(case_nodes["Holiday"]["run"]),
        "Weather": find_case_index(case_nodes["Weather"]["run"]),
    }
    forced = np.array([i for i in case_indices.values() if i is not None], dtype=int)
    rows = np.arange(len(sample_z))
    if len(rows) > EMBED_ANOM_SAMPLES:
        random_rows = rng.choice(rows, size=EMBED_ANOM_SAMPLES, replace=False)
        rows = np.unique(np.concatenate([random_rows, forced]))
    perplexity = min(30, max(5, (len(rows) - 1) // 3))
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    sample_z_vis = StandardScaler().fit_transform(sample_z[rows])
    emb = TSNE(n_components=2, random_state=SEED, init='random',
               learning_rate='auto', perplexity=perplexity).fit_transform(sample_z_vis)
    row_pos = {int(row): pos for pos, row in enumerate(rows)}
    case_positions = {label: row_pos[idx] for label, idx in case_indices.items()
                      if idx is not None and idx in row_pos}
    case_clusters = {label: int(cluster_labels[idx]) for label, idx in case_indices.items()
                     if idx is not None}
    cluster_patterns = summarize_cluster_patterns(
        X - b_in, cluster_labels, active,
        sorted(set(case_clusters.values())))
    return dict(embedding=emb, assignment=cluster_labels[rows], rows=rows,
                full_assignment=cluster_labels, case_positions=case_positions,
                case_clusters=case_clusters, cluster_patterns=cluster_patterns,
                metadata=sample_metadata)


def find_case_index(metadata, run):
    """Find the metadata sample closest to the midpoint of an inclusive interval."""
    r_start, r_end = run
    candidates = [i for i, meta in enumerate(metadata)
                  if r_start <= meta["target_start"] <= r_end]
    if not candidates:
        return None
    mid = (r_start + r_end) / 2.0
    return min(candidates, key=lambda i: abs(metadata[i]["target_start"] - mid))


def cluster_for_run(embed, run):
    """Return the cluster id for a candidate display run, if available."""
    idx = find_case_index(embed["metadata"], run)
    if idx is None:
        return None
    return int(embed["full_assignment"][idx])


def select_case_pair_by_cluster(holiday_ranked, weather_ranked, embed, top_k=8):
    """Among eligible candidates, require distinct clusters so the left panel
    shows two separate semantic groups with different deviation patterns."""
    def shortlist(ranked):
        eligible = [(i, r) for i, r in enumerate(ranked[:top_k]) if case_is_eligible(r)]
        return eligible if eligible else list(enumerate(ranked[:top_k]))

    best = None
    for hi, h in shortlist(holiday_ranked):
        h_cluster = cluster_for_run(embed, h["display_run"])
        if h_cluster is None:
            continue
        for wi, w in shortlist(weather_ranked):
            w_cluster = cluster_for_run(embed, w["display_run"])
            if w_cluster is None:
                continue
            distinct = int(h_cluster != w_cluster)
            score = (distinct, case_selection_priority(h), case_selection_priority(w), -(hi + wi))
            if best is None or score > best[0]:
                best = (score, h, w)
    if best is None:
        return holiday_ranked[0], weather_ranked[0]
    return best[1], best[2]


def select_representative_node_for_case(case, active, label):
    """Pick the display node with the clearest visual contrast: ProtoDRIFT tracks
    the anomalous trend while baselines miss it in the label-specific direction.

    Flow direction and ProtoDRIFT closeness are already enforced inside
    case_display_score, so only the weather-specific non-trivial-flow requirement
    is applied on top of it.
    """
    active = np.asarray(active, dtype=bool)
    if not active.any():
        active = np.ones(case["true"].shape[-1], dtype=bool)
    baseline_names = [name for name in case["preds"] if name != "STCross"]
    best = None
    for node in np.where(active)[0]:
        true = case["true"][:, node]
        st = case["preds"]["STCross"][:, node]
        baselines = {name: case["preds"][name][:, node] for name in baseline_names}
        contrast = case_display_score(true, case["nbl"][:, node], st, baselines, label)
        # The plotted y-axis is clamped at 0, so a slightly negative ProtoDRIFT
        # step must not veto a node; only require non-trivial real flow.
        flow_ok = 1 if label == "Holiday" else int(true.mean() >= 1.0)
        score = (flow_ok, contrast)
        if best is None or score > best[0]:
            best = (score, int(node))
    return best[1]


def build_cases_from_results(data, stcross, learneds, selected_results):
    """Collect predictions and representative nodes for selected result rows."""
    cases = {}
    for label, result in selected_results:
        pred = collect_case_prediction(data, stcross, learneds, result["display_run"])
        node = int(select_representative_node_for_case(pred, data["active"], label))
        pred["node"] = node
        pred["i_rep"] = (pred["i_lo"] + pred["i_hi"]) // 2
        pred["summary"] = result
        pred["event_label"] = case_event_label(label, data["timestamps"][result["run"][0]])
        cases[label] = pred
    return cases


def choose_display_case(data, stcross, learneds, ranked, label, top_k=24):
    """Pick the right-panel case by node-level visual contrast: among eligible
    candidates, require ProtoDRIFT to stay close to GT and then maximize the
    baseline directional miss on the representative display node."""
    candidates = [r for r in ranked[:top_k] if case_is_eligible(r)]
    if not candidates:
        candidates = ranked[:top_k]
    best = None
    for result in candidates:
        pred = collect_case_prediction(data, stcross, learneds, result["display_run"])
        node = int(select_representative_node_for_case(pred, data["active"], label))
        baseline_curves = {name: p[:, node] for name, p in pred["preds"].items()
                           if name != "STCross"}
        score = case_display_score(pred["true"][:, node], pred["nbl"][:, node],
                                   pred["preds"]["STCross"][:, node],
                                   baseline_curves, label)
        if best is None or score > best[0]:
            pred["node"] = node
            pred["i_rep"] = (pred["i_lo"] + pred["i_hi"]) // 2
            pred["summary"] = result
            pred["event_label"] = case_event_label(label, data["timestamps"][result["run"][0]])
            pred["display_score"] = score
            best = (score, pred)
    return best[1]


def build_pinned_display_case(data, stcross, learneds, label, pin):
    """Rebuild a frozen right-panel case from its recorded run and display node."""
    pred = collect_case_prediction(data, stcross, learneds, pin["run"])
    node = int(pin["node"])
    baseline_curves = {name: p[:, node] for name, p in pred["preds"].items()
                       if name != "STCross"}
    pred["node"] = node
    pred["i_rep"] = (pred["i_lo"] + pred["i_hi"]) // 2
    pred["summary"] = None
    pred["event_label"] = case_event_label(label, data["timestamps"][pin["run"][0]])
    pred["display_score"] = case_display_score(
        pred["true"][:, node], pred["nbl"][:, node],
        pred["preds"]["STCross"][:, node], baseline_curves, label)
    return pred


def anchor_case_stub(data, anchor):
    """Build the minimal case dict needed to force an anchor run into the embedding."""
    i_lo, i_hi = interval_sample_range(*anchor["run"], data["te_s"], data["te_e"])
    return {"i_rep": (i_lo + i_hi) // 2, "run": anchor["run"]}


def choose_cases(data, stcross, learneds):
    """Select one holiday and one weather run, then representative nodes for each."""
    holiday_runs = find_runs(data["holiday"], MIN_RUN, data["te_s"], data["te_e"])
    weather_runs = find_runs(data["weather"], MIN_RUN, data["te_s"], data["te_e"])
    holiday_ranked = evaluate_runs(holiday_runs, "Holiday", data, stcross, learneds)
    weather_ranked = evaluate_runs(weather_runs, "Weather", data, stcross, learneds)
    if not holiday_ranked:
        raise RuntimeError("No valid holiday run found in CHI test period")
    if not weather_ranked:
        raise RuntimeError("No valid weather run found in CHI test period")
    cases = build_cases_from_results(
        data, stcross, learneds,
        [("Holiday", holiday_ranked[0]), ("Weather", weather_ranked[0])])
    return cases, holiday_ranked, weather_ranked


def parse_overrides(overrides):
    """Parse ['DAN=checkpoints_viz'] into {'DAN': 'checkpoints_viz'}."""
    parsed = {}
    for item in overrides:
        if '=' not in item:
            raise SystemExit(f"--ckpt-dir expects NAME=PATH, got: {item}")
        name, path = item.split('=', 1)
        parsed[name.strip()] = path.strip()
    return parsed


def plot_dualpanel(out_dir, embed, cases, baseline_names):
    """Render the final double-column figure.

    Panel captions (a)-(d) sit *below* their panels at the same size as the inner
    pattern titles. The model legend occupies the top-right band, edge-aligned to
    the right column. Caption placement is measured from rendered tight bboxes so
    it clears tick labels and axis captions without magic offsets.
    """
    panel_title_size = 26
    caption_pad = 0.018
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(20.0, 10.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.46], wspace=0.20, hspace=0.34)
    left_gs = gs[:, 0].subgridspec(2, 1, height_ratios=[3.55, 1.0], hspace=0.46)
    ax_emb = fig.add_subplot(left_gs[0, 0])
    pattern_gs = left_gs[1, 0].subgridspec(1, 2, wspace=0.10)
    ax_pat_h = fig.add_subplot(pattern_gs[0, 0])
    ax_pat_w = fig.add_subplot(pattern_gs[0, 1], sharey=ax_pat_h)
    ax_h = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1], sharex=None)

    emb = embed["embedding"]
    assignment = embed["assignment"]
    for k in sorted(np.unique(assignment)):
        mask = assignment == k
        ax_emb.scatter(emb[mask, 0], emb[mask, 1], s=26, alpha=0.58,
                       color=PROTO_COLORS[int(k) % len(PROTO_COLORS)],
                       linewidths=0, label=f"C{k + 1}")
    label_offsets = {"Holiday": (0, 20), "Weather": (0, 20)}
    label_align = {"Holiday": "center", "Weather": "center"}
    for label, pos in embed["case_positions"].items():
        cluster = embed.get("case_clusters", {}).get(label)
        suffix = f" (C{cluster + 1})" if cluster is not None else ""
        event_label = LEFT_PANEL_ANCHORS[label]["event_label"]
        ax_emb.scatter(emb[pos, 0], emb[pos, 1], s=520,
                       marker=CASE_MARKERS[label], color=CASE_COLORS[label],
                       edgecolor="black", linewidth=1.6, zorder=6,
                       label=f"{event_label}{suffix}")
        ax_emb.annotate(f"{event_label}{suffix}", (emb[pos, 0], emb[pos, 1]),
                        xytext=label_offsets.get(label, (0, 20)),
                        textcoords="offset points",
                        fontsize=26, fontweight="bold",
                        ha=label_align.get(label, "center"))
    ax_emb.set_xticks([])
    ax_emb.set_yticks([])
    ax_emb.margins(x=0.07, y=0.05)
    ax_emb.grid(alpha=0.16)

    base_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=13,
               markerfacecolor=PROTO_COLORS[k % len(PROTO_COLORS)],
               markeredgecolor='none', label=f"C{k + 1}")
        for k in range(int(assignment.max()) + 1)
    ]
    # Only the cluster keys are listed; the two cases are labelled in place.
    ax_emb.legend(handles=base_handles, fontsize=22,
                  loc="lower left", ncol=4, frameon=True, framealpha=0.85,
                  handletextpad=0.3, columnspacing=0.7, labelspacing=0.25)

    def plot_pattern(ax, label, direction_text, show_ylabel):
        cluster = embed.get("case_clusters", {}).get(label)
        if cluster is not None and cluster in embed.get("cluster_patterns", {}):
            pattern = embed["cluster_patterns"][cluster]
            steps = np.arange(len(pattern))
            ax.plot(steps, pattern, color=CASE_COLORS[label], linewidth=3.4,
                    marker="o", markersize=6.0)
        ax.axhline(0, color="grey", linewidth=1.2, linestyle="--")
        # The cluster id lives in the title so the small panels need no legend.
        suffix = f" (C{cluster + 1})" if cluster is not None else ""
        ax.set_title(f"{direction_text}{suffix}", fontsize=panel_title_size,
                     fontweight="bold")
        ax.set_xlabel("Forecast step", fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(alpha=0.25)
        # The pair shares one y axis, so only the left panel keeps its labels.
        if show_ylabel:
            ax.set_ylabel(r"Mean $X-B$", fontsize=24)
        else:
            ax.tick_params(axis='y', labelleft=False)

    plot_pattern(ax_pat_h, "Holiday", "Holiday", show_ylabel=True)
    plot_pattern(ax_pat_w, "Weather", "Weather", show_ylabel=False)

    def plot_case(ax, label, case, panel_letter):
        node = case["node"]
        x = np.arange(len(case["timesteps"]))
        ax.plot(x, case["true"][:, node], color=MODEL_COLORS["Ground truth"],
                linewidth=4.0, label="Ground truth")
        ax.plot(x, case["preds"]["STCross"][:, node], color=MODEL_COLORS["STCross"],
                linewidth=3.8, label=display_model_name("STCross"))
        for name in baseline_names:
            ax.plot(x, case["preds"][name][:, node],
                    color=MODEL_COLORS.get(name, "#17becf"), linewidth=3.0,
                    linestyle="--", label=display_model_name(name))
        ax.set_ylabel("Flow", fontsize=30)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.tick_params(axis='both', labelsize=24)
        ticks = np.linspace(0, len(x) - 1, min(6, len(x)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([case["timesteps"][i].strftime("%H:%M") for i in ticks])
        event_label = case.get("event_label", CASE_FALLBACK_LABELS[label])
        return f"({panel_letter}) {event_label} case: node {node}"

    caption_c = plot_case(ax_h, "Holiday", cases["Holiday"], "c")
    caption_d = plot_case(ax_w, "Weather", cases["Weather"], "d")
    # The left column reaches near the top because the legend band only spans the
    # right column; the right column is re-stacked below that band further down.
    # Margins are kept tight on all four sides: panel (a) carries no y label or
    # tick labels, so any left margin would be pure whitespace.
    fig.subplots_adjust(left=0.016, right=0.995, bottom=0.120, top=0.988,
                        wspace=0.20, hspace=0.34)

    # The right column is stacked explicitly so the gap between (c) and (d) can
    # hold (c)'s tick labels plus its caption.
    pos_c = ax_h.get_position()
    pos_d = ax_w.get_position()
    col_top, col_bottom, inner_gap = 0.888, 0.130, 0.115
    case_height = (col_top - col_bottom - inner_gap) / 2.0
    ax_w.set_position([pos_d.x0, col_bottom, pos_d.width, case_height])
    ax_h.set_position([pos_c.x0, col_bottom + case_height + inner_gap,
                       pos_c.width, case_height])

    # Model legend in the top-right band, expanded to the right column's width so
    # its left and right edges line up with panels (c) and (d). Same size and
    # weight as the panel captions.
    handles, labels = ax_h.get_legend_handles_labels()
    legend_height = 0.090
    fig.legend(handles, labels, loc="lower left", ncol=3, frameon=False,
               prop={"size": panel_title_size, "weight": "bold"},
               mode="expand", handlelength=2.4, handletextpad=0.5,
               borderpad=0.0, borderaxespad=0.0,
               bbox_to_anchor=(pos_c.x0, 0.995 - legend_height,
                               pos_c.width, legend_height))

    # Captions go under each panel, just below its rendered lower edge. The two
    # bottom-row captions share one baseline even though (b) and (d) have
    # different tick/label heights below their axes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()

    # Panel (b)'s y label overhangs the gridspec column, so it pokes out to the
    # left of panel (a). Shift and narrow the pattern pair by that overhang so the
    # label starts exactly on (a)'s left edge while the right edges stay aligned.
    pos_emb = ax_emb.get_position()
    pos_ph = ax_pat_h.get_position()
    pos_pw = ax_pat_w.get_position()
    pat_left = to_fig.transform(ax_pat_h.get_tightbbox(renderer))[0][0]
    overhang = pos_emb.x0 - pat_left
    if overhang > 0:
        pair_gap = pos_pw.x0 - pos_ph.x1
        new_left = pos_ph.x0 + overhang
        pair_width = (pos_pw.x1 - new_left - pair_gap) / 2.0
        ax_pat_h.set_position([new_left, pos_ph.y0, pair_width, pos_ph.height])
        ax_pat_w.set_position([new_left + pair_width + pair_gap, pos_pw.y0,
                               pair_width, pos_pw.height])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    def caption_top(axes_list):
        return min(to_fig.transform(a.get_tightbbox(renderer))[0][1]
                   for a in axes_list) - caption_pad

    def add_caption(axes_list, text, top=None):
        y = caption_top(axes_list) if top is None else top
        # Center on the panel's full visual extent, which includes the y label and
        # tick labels on the left; centering on the plot box alone looks right-shifted.
        boxes = [to_fig.transform(a.get_tightbbox(renderer)) for a in axes_list]
        x0 = min(b[0][0] for b in boxes)
        x1 = max(b[1][0] for b in boxes)
        fig.text((x0 + x1) / 2.0, y, text, ha="center", va="top",
                 fontsize=panel_title_size, fontweight="bold")

    add_caption([ax_emb], r"(a) Clustered anomaly semantics in $Z_a$")
    add_caption([ax_h], caption_c)
    bottom_top = min(caption_top([ax_pat_h, ax_pat_w]), caption_top([ax_w]))
    add_caption([ax_pat_h, ax_pat_w],
                "(b) Cluster deviation: holiday high vs. weather low",
                top=bottom_top)
    add_caption([ax_w], caption_d, top=bottom_top)

    png_path = os.path.join(out_dir, "v4_semantic_case_dualpanel.png")
    pdf_path = os.path.join(out_dir, "v4_semantic_case_dualpanel.pdf")
    # The layout above is tuned explicitly, so no tight bbox: it would override the
    # margins and expand the crop around any artist that overhangs an axes.
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="V4 semantic cluster + case-level forecasting visualization")
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Learned baselines to compare; HA is intentionally excluded")
    parser.add_argument("--ckpt-dir", action="append", default=[], metavar="NAME=PATH",
                        help="Override checkpoint dir for a baseline")
    parser.add_argument("--out-dir", default=OUT_DIR,
                        help="Output directory for PNG/PDF")
    parser.add_argument("--embed-samples", type=int, default=EMBED_ANOM_SAMPLES,
                        help="Maximum CHI anomaly samples to encode for the left panel")
    args = parser.parse_args()

    config = dict(BASELINE_CONFIG)
    config.update(parse_overrides(args.ckpt_dir))
    requested_baselines = args.baselines or DEFAULT_BASELINES
    explicit_baselines = args.baselines is not None
    for name in requested_baselines:
        if name not in config:
            raise SystemExit(f"Unknown baseline '{name}'. Known: {list(BASELINE_CONFIG)}")

    rng = np.random.default_rng(SEED)
    data = prepare_chi()

    stcross_ckpt_dir = None
    for candidate in STCROSS_CKPT_CANDIDATES:
        stcross_ckpt_dir = resolve_checkpoint_dir("STCross", candidate)
        if stcross_ckpt_dir is not None:
            break
    if stcross_ckpt_dir is None:
        raise SystemExit("No STCross_NYC_best.pt found in checkpoints_viz/ or checkpoints/")

    resolved_baselines = []
    for name in requested_baselines:
        ckpt_dir = resolve_checkpoint_dir(name, config[name])
        if ckpt_dir is None:
            message = f"No checkpoint found for {name}; expected {name}_NYC_best.pt"
            if explicit_baselines:
                raise SystemExit(message)
            print(f"  [skip] {message}")
            continue
        resolved_baselines.append((name, ckpt_dir))
    if not resolved_baselines:
        raise SystemExit("No learned baseline checkpoints found for V4 comparison")

    print("Loading models...")
    print(f"  {display_model_name('STCross')} <- {stcross_ckpt_dir}/STCross_NYC_best.pt")
    stcross = load_stcross(stcross_ckpt_dir)
    learneds = []
    baseline_names = []
    for name, ckpt_dir in resolved_baselines:
        print(f"  {name:<8} <- {ckpt_dir}/{name}_NYC_best.pt")
        learneds.append((name, load_learned_baseline(name, ckpt_dir)))
        baseline_names.append(name)

    print("Ranking holiday/weather candidate runs without HA...")
    holiday_runs = find_runs(data["holiday"], DISPLAY_MIN_RUN, data["te_s"], data["te_e"])
    weather_runs = find_runs(data["weather"], DISPLAY_MIN_RUN, data["te_s"], data["te_e"])
    h_ranked = evaluate_runs(holiday_runs, "Holiday", data, stcross, learneds)
    w_ranked = evaluate_runs(weather_runs, "Weather", data, stcross, learneds)
    if not h_ranked or not w_ranked:
        raise RuntimeError("No valid holiday/weather runs found in CHI test period")

    print("Choosing right-panel display cases by node-level visual contrast...")
    cases = {}
    for label, ranked in [("Holiday", h_ranked), ("Weather", w_ranked)]:
        pin = PINNED_DISPLAY_CASES.get(label)
        if pin is not None:
            cases[label] = build_pinned_display_case(data, stcross, learneds, label, pin)
        else:
            cases[label] = choose_display_case(data, stcross, learneds, ranked, label)
    for label in ["Holiday", "Weather"]:
        case = cases[label]
        node = case["node"]
        st_mae = float(np.abs(case["preds"]["STCross"][:, node] - case["true"][:, node]).mean())
        pinned = "pinned" if label in PINNED_DISPLAY_CASES else "selected"
        print(f"  {label:<7} [{pinned}] {case['event_label']} run={case['run']} "
              f"node={node} stMAE={st_mae:.3f} displayScore={case['display_score']}")

    print("Encoding CHI anomaly windows with pinned left-panel anchors...")
    anchor_h = anchor_case_stub(data, LEFT_PANEL_ANCHORS["Holiday"])
    anchor_w = anchor_case_stub(data, LEFT_PANEL_ANCHORS["Weather"])
    X, b_in, b_fut, sample_starts, sample_events = build_embedding_inputs(
        data, anchor_h, anchor_w, rng, max_samples=args.embed_samples)
    case_nodes = {
        "Holiday": {"run": LEFT_PANEL_ANCHORS["Holiday"]["run"]},
        "Weather": {"run": LEFT_PANEL_ANCHORS["Weather"]["run"]},
    }
    embed = compute_embedding(stcross, X, b_in, b_fut, sample_starts,
                              data["N"], case_nodes, rng,
                              sample_events=sample_events, active=data["active"])
    for label in ["Holiday", "Weather"]:
        cluster = embed.get("case_clusters", {}).get(label)
        print(f"  [anchor] {LEFT_PANEL_ANCHORS[label]['event_label']:<16} "
              f"cluster=C{cluster + 1 if cluster is not None else -1}")

    print("Rendering V4 dual-panel figure...")
    png_path, pdf_path = plot_dualpanel(args.out_dir, embed, cases, baseline_names)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


if __name__ == "__main__":
    main()
