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
    """Pick a display node whose curves best match the label-specific case story."""
    active = np.asarray(active, dtype=bool)
    if not active.any():
        active = np.ones(case["true"].shape[-1], dtype=bool)
    baseline_preds = [pred for name, pred in case["preds"].items() if name != "STCross"]
    candidates = np.where(active)[0]
    best = None
    for node in candidates:
        true = case["true"][:, node]
        nbl = case["nbl"][:, node]
        st = case["preds"]["STCross"][:, node]
        delta = true - nbl
        st_mae = float(np.abs(st - true).mean())
        if baseline_preds:
            baseline_bias = float(np.mean([pred[:, node] - true for pred in baseline_preds]))
        else:
            baseline_bias = 0.0
        if label == "Holiday":
            score = (float((delta > 0).mean()), int(delta.min() > 0),
                     -st_mae, -baseline_bias, float(delta.mean()))
        else:
            score = (int(true.min() >= 0), int(st.min() >= 0), int(true.mean() >= 1.0),
                     min(float(true.mean()), 8.0), -st_mae,
                     baseline_bias, -abs(float(delta.mean())))
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
    """Render the final double-column figure."""
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.figure(figsize=(16.0, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.46], wspace=0.23, hspace=0.36)
    left_gs = gs[:, 0].subgridspec(2, 1, height_ratios=[3.75, 1.0], hspace=0.30)
    ax_emb = fig.add_subplot(left_gs[0, 0])
    pattern_gs = left_gs[1, 0].subgridspec(1, 2, wspace=0.24)
    ax_pat_h = fig.add_subplot(pattern_gs[0, 0])
    ax_pat_w = fig.add_subplot(pattern_gs[0, 1])
    ax_h = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1], sharex=None)

    emb = embed["embedding"]
    assignment = embed["assignment"]
    for k in sorted(np.unique(assignment)):
        mask = assignment == k
        ax_emb.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.58,
                       color=PROTO_COLORS[int(k) % len(PROTO_COLORS)],
                       linewidths=0, label=f"C{k + 1}")
    label_offsets = {"Holiday": (9, 12), "Weather": (-14, -22)}
    label_align = {"Holiday": "left", "Weather": "right"}
    for label, pos in embed["case_positions"].items():
        cluster = embed.get("case_clusters", {}).get(label)
        suffix = f" (C{cluster + 1})" if cluster is not None else ""
        event_label = cases.get(label, {}).get("event_label", CASE_FALLBACK_LABELS[label])
        ax_emb.scatter(emb[pos, 0], emb[pos, 1], s=360,
                       marker=CASE_MARKERS[label], color=CASE_COLORS[label],
                       edgecolor="black", linewidth=1.3, zorder=6,
                       label=f"{event_label}{suffix}")
        ax_emb.annotate(f"{event_label}{suffix}", (emb[pos, 0], emb[pos, 1]),
                        xytext=label_offsets.get(label, (8, 8)), textcoords="offset points",
                        fontsize=14, fontweight="bold",
                        ha=label_align.get(label, "left"))
    ax_emb.set_title(r"(a) Clustered anomaly semantics in $Z_a$",
                     fontsize=18, fontweight="bold")
    ax_emb.set_xlabel("t-SNE dim. 1", fontsize=15)
    ax_emb.set_ylabel("t-SNE dim. 2", fontsize=15)
    ax_emb.set_xticks([])
    ax_emb.set_yticks([])
    ax_emb.grid(alpha=0.16)

    base_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=7,
               markerfacecolor=PROTO_COLORS[k % len(PROTO_COLORS)],
               markeredgecolor='none', label=f"C{k + 1}")
        for k in range(int(assignment.max()) + 1)
    ]
    case_handles = [
        Line2D([0], [0], marker=CASE_MARKERS[label], linestyle='', markersize=11,
               markerfacecolor=CASE_COLORS[label], markeredgecolor='black',
               label=cases[label].get("event_label", CASE_FALLBACK_LABELS[label]))
        for label in ["Holiday", "Weather"]
    ]
    ax_emb.legend(handles=base_handles + case_handles, fontsize=10.5,
                  loc="lower left", ncol=2, frameon=True, framealpha=0.85)

    def plot_pattern(ax, label, direction_text):
        cluster = embed.get("case_clusters", {}).get(label)
        event_label = cases[label].get("event_label", CASE_FALLBACK_LABELS[label])
        if cluster is not None and cluster in embed.get("cluster_patterns", {}):
            pattern = embed["cluster_patterns"][cluster]
            steps = np.arange(len(pattern))
            ax.plot(steps, pattern, color=CASE_COLORS[label], linewidth=2.7,
                    marker="o", markersize=4.0,
                    label=f"C{cluster + 1} {event_label}")
        ax.axhline(0, color="grey", linewidth=1.0, linestyle="--")
        ax.set_title(direction_text, fontsize=12.2, fontweight="bold")
        ax.set_ylabel(r"Mean $X-B$", fontsize=10.8)
        ax.tick_params(axis='both', labelsize=9.2)
        ax.grid(alpha=0.25)
        if cluster is not None:
            ax.legend(fontsize=8.8, loc="best", frameon=False)

    ax_pat_h.text(1.05, 1.34, "(b) Selected-cluster flow-deviation patterns",
                  transform=ax_pat_h.transAxes, ha="center", va="bottom",
                  fontsize=13.5, fontweight="bold")
    plot_pattern(ax_pat_h, "Holiday", "Holiday high-flow")
    plot_pattern(ax_pat_w, "Weather", "Weather low-flow")
    ax_pat_h.set_xlabel("Forecast step", fontsize=11)
    ax_pat_w.set_xlabel("Forecast step", fontsize=11)

    def plot_case(ax, label, case, panel_letter):
        node = case["node"]
        x = np.arange(len(case["timesteps"]))
        ax.plot(x, case["true"][:, node], color=MODEL_COLORS["Ground truth"],
                linewidth=3.0, label="Ground truth")
        ax.plot(x, case["preds"]["STCross"][:, node], color=MODEL_COLORS["STCross"],
                linewidth=2.8, label=display_model_name("STCross"))
        for name in baseline_names:
            ax.plot(x, case["preds"][name][:, node],
                    color=MODEL_COLORS.get(name, "#17becf"), linewidth=2.2,
                    linestyle="--", label=display_model_name(name))
        t0 = case["timesteps"][0].strftime("%m-%d %H:%M")
        t1 = case["timesteps"][-1].strftime("%m-%d %H:%M")
        event_label = case.get("event_label", CASE_FALLBACK_LABELS[label])
        ax.set_title(f"({panel_letter}) {event_label} case: node {node} ({t0}--{t1})",
                     fontsize=17, fontweight="bold")
        ax.set_ylabel("Flow", fontsize=15)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.tick_params(axis='both', labelsize=12)
        ticks = np.linspace(0, len(x) - 1, min(6, len(x)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([case["timesteps"][i].strftime("%H:%M") for i in ticks])

    plot_case(ax_h, "Holiday", cases["Holiday"], "d")
    plot_case(ax_w, "Weather", cases["Weather"], "e")
    ax_w.set_xlabel("Rolling step-0 prediction time", fontsize=15)
    handles, labels = ax_h.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=14, bbox_to_anchor=(0.66, 0.995), columnspacing=1.05,
               handlelength=2.2)
    fig.subplots_adjust(left=0.055, right=0.987, bottom=0.085, top=0.875,
                        wspace=0.23, hspace=0.36)

    png_path = os.path.join(out_dir, "v4_semantic_case_dualpanel.png")
    pdf_path = os.path.join(out_dir, "v4_semantic_case_dualpanel.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
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

    print("Selecting holiday/weather cases without HA...")
    cases, h_ranked, w_ranked = choose_cases(data, stcross, learneds)
    for label, ranked in [("Holiday", h_ranked), ("Weather", w_ranked)]:
        best = ranked[0]
        print(f"  {label:<7} {best['date'].strftime('%Y-%m-%d')} "
              f"{display_model_name('STCross')}={best['mae_st']:.4f} margin={best['margin']:.4f} "
              f"run={best['display_run']} ratio={best['anomaly_ratio']:.2f} "
              f"day={best['daytime_fraction']:.2f} "
              f"flowΔ={best['flow_delta']:.3f} baseBias={best['baseline_bias']:.3f} "
              f"posSteps={best['positive_step_fraction']:.2f} trueMean={best['true_mean']:.3f} "
              f"node={cases[label]['node']}")

    print("Encoding CHI anomaly windows, clustering Z_a, and extracting cluster patterns...")
    X, b_in, b_fut, sample_starts, sample_events = build_embedding_inputs(
        data, cases["Holiday"], cases["Weather"], rng, max_samples=args.embed_samples)
    case_nodes = {
        "Holiday": {"run": cases["Holiday"]["run"], "node": cases["Holiday"]["node"]},
        "Weather": {"run": cases["Weather"]["run"], "node": cases["Weather"]["node"]},
    }
    embed = compute_embedding(stcross, X, b_in, b_fut, sample_starts,
                              data["N"], case_nodes, rng,
                              sample_events=sample_events, active=data["active"])

    selected_h, selected_w = select_case_pair_by_cluster(h_ranked, w_ranked, embed)
    if (selected_h is not cases["Holiday"]["summary"] or
            selected_w is not cases["Weather"]["summary"]):
        print("Refining cases to prefer distinct semantic clusters...")
        cases = build_cases_from_results(
            data, stcross, learneds,
            [("Holiday", selected_h), ("Weather", selected_w)])
        for label, result in [("Holiday", selected_h), ("Weather", selected_w)]:
            cluster = cluster_for_run(embed, result["display_run"])
            print(f"  {label:<7} cluster=C{cluster + 1 if cluster is not None else -1} "
                  f"run={result['display_run']} ratio={result['anomaly_ratio']:.2f} "
                  f"day={result['daytime_fraction']:.2f} "
                  f"flowΔ={result['flow_delta']:.3f} baseBias={result['baseline_bias']:.3f} "
                  f"posSteps={result['positive_step_fraction']:.2f} trueMean={result['true_mean']:.3f} "
                  f"node={cases[label]['node']}")
        X, b_in, b_fut, sample_starts, sample_events = build_embedding_inputs(
            data, cases["Holiday"], cases["Weather"], rng, max_samples=args.embed_samples)
        case_nodes = {
            "Holiday": {"run": cases["Holiday"]["run"], "node": cases["Holiday"]["node"]},
            "Weather": {"run": cases["Weather"]["run"], "node": cases["Weather"]["node"]},
        }
        embed = compute_embedding(stcross, X, b_in, b_fut, sample_starts,
                                  data["N"], case_nodes, rng,
                                  sample_events=sample_events, active=data["active"])

    print("Rendering V4 dual-panel figure...")
    png_path, pdf_path = plot_dualpanel(args.out_dir, embed, cases, baseline_names)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


if __name__ == "__main__":
    main()
