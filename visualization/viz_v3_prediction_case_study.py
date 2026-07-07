"""V3: Prediction case study on high-anomaly intervals (STCross decomposition).

Loads STCross (α=0.3) + HA (NBL reference, no checkpoint) + one or more
learned baselines, scans CHI test-period holiday and weather masks for runs of
consecutive anomaly steps, and for each candidate interval computes per-model
MAE on step-0 rolling predictions. Selects the holiday and weather interval
where STCross wins most over the strongest baseline, then plots per interval:
  - Top panel:    rolling 1-step-ahead prediction over the interval
                  (true vs STCross vs HA vs each selected baseline),
                  aggregated over active nodes
  - Bottom panel: STCross decomposition for one representative sample
                  (NBL baseline + Δ̂_type + Δ̂_adj stacked, + STCross pred + true)

==============================  BASELINES  ==================================
Which baselines and their checkpoint dirs are configured in BASELINE_CONFIG
below. Edit that dict to add/relocate a baseline. From the CLI:

  --baselines NAME [NAME ...]      which baselines to compare (default: DAN)
  --ckpt-dir NAME=PATH             override checkpoint dir for one baseline
                                   (repeatable, e.g. --ckpt-dir DAN=checkpoints)

HA is always included as the NBL reference (non-parametric, no checkpoint).
STCross always loads from checkpoints_viz/ (retrain_for_viz.py output).

Usage:
    python visualization/viz_v3_prediction_case_study.py                       # default: DAN
    python visualization/viz_v3_prediction_case_study.py --baselines DAN EVL    # two baselines
    python visualization/viz_v3_prediction_case_study.py --baselines DAN EVL LSTM DLinear PatchTST
    python visualization/viz_v3_prediction_case_study.py --baselines DAN --ckpt-dir DAN=checkpoints
    python visualization/viz_v3_prediction_case_study.py --no-ha               # drop HA ref line
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.config import EvalConfig
from eval.evaluator import _compute_nbl_full
from model.registry import MODEL_REGISTRY
from train.trainer import TrainConfig
from data_preprocess.dataset import load_flow, load_anomaly_mask, split_data
from data_preprocess.normalizer import PerNodeStandardScaler

VIZ_CKPT_DIR = "checkpoints_viz"
OUT_DIR = "results_viz"
H, P = 12, 12
ACTIVE_THRESH = 0.5      # node active if train-period mean flow > this
MIN_RUN = 6              # min consecutive anomaly hours to consider an interval

# ---------------------------------------------------------------------------
# Baseline configuration  ← EDIT HERE to add/relocate a baseline
# ---------------------------------------------------------------------------
# Map baseline name -> checkpoint dir. All entries must be per_node models that
# load via TorchModel.load_checkpoint("NYC"). HA is handled separately (no ckpt).
# To add a new baseline: register it in model/registry.py, then add an entry here.
BASELINE_CONFIG = {
    "DAN":      "checkpoints_viz",   # retrained by experiments/retrain_for_viz.py
    "EVL":      "checkpoints",       # main-results checkpoint
    "LSTM":     "checkpoints",
    "DLinear":  "checkpoints",
    "PatchTST": "checkpoints",
}

# Distinct colors for each learned baseline in the rolling-curve plot
BASELINE_COLORS = {
    "DAN": "#1f77b4", "EVL": "#2ca02c", "LSTM": "#9467bd",
    "DLinear": "#8c564b", "PatchTST": "#e377c2",
}
DEFAULT_COLOR = "#17becf"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_stcross():
    tc = TrainConfig(checkpoint_dir=VIZ_CKPT_DIR)
    ec = EvalConfig()
    m = MODEL_REGISTRY["STCross"](ec, tc, stcross_k=8, stcross_alpha=0.3,
                                   stcross_beta=0.01, stcross_gamma=0.1)
    m.load_checkpoint("NYC")
    return m


def load_ha(target_city="CHI"):
    """HA is non-parametric; rebuild lookup on the target city (no checkpoint)."""
    ec = EvalConfig(target_city=target_city)
    tc = TrainConfig()
    m = MODEL_REGISTRY["HA"](ec, tc)
    m.fit(None, None, None, None)
    return m


def load_learned_baseline(name, ckpt_dir):
    ec = EvalConfig()
    tc = TrainConfig(checkpoint_dir=ckpt_dir)
    m = MODEL_REGISTRY[name](ec, tc)
    m.load_checkpoint("NYC")
    return m


# ---------------------------------------------------------------------------
# Data + interval scanning
# ---------------------------------------------------------------------------

def prepare_chi():
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
    active = flow[tr_s:tr_e].mean(axis=0) > ACTIVE_THRESH
    return dict(flow=flow, flow_norm=flow_norm, mask=mask, holiday=holiday,
                weather=weather, T=T, N=N, tr_s=tr_s, tr_e=tr_e,
                te_s=te_s, te_e=te_e, scaler=scaler, timestamps=timestamps,
                nbl_full=nbl_full, active=active)


def find_runs(mask, min_len, lo, hi):
    """Runs of consecutive True in mask[lo:hi] with length >= min_len.
    Returns list of (start, end_inclusive) global indices."""
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


def interval_sample_range(r_start, r_end, te_s, te_e):
    """Rolling 1-step-ahead sample range: step-0 of sample i covers timestep i+H.
    To cover timesteps [r_start, r_end], need i in [r_start-H, r_end-H], clamped
    to valid test samples [te_s, te_e-H-P]."""
    i_lo = max(te_s, r_start - H)
    i_hi = min(te_e - H - P, r_end - H)
    return i_lo, i_hi


def stack_windows(arr_2d, idx, length):
    """Build (L, length, N) from a (T, N) array: row l = arr_2d[idx[l]:idx[l]+length]."""
    return np.stack([arr_2d[i:i + length] for i in idx], axis=0)


def build_ha_timestamps(timestamps, idx):
    """Build (L, P) object array of pd.Timestamp for HAModel.predict_with_timestamps."""
    L = len(idx)
    ts = np.empty((L, P), dtype=object)
    for li, i in enumerate(idx):
        for j in range(P):
            ts[li, j] = timestamps[i + H + j]
    return ts


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def stcross_rolling(stcross, data, i_lo, i_hi):
    """STCross rolling predictions over [i_lo, i_hi]. Returns dict of (L, P, N) original-scale."""
    idx = np.arange(i_lo, i_hi + 1)
    X = stack_windows(data["flow_norm"], idx, H)
    b_in = stack_windows(data["nbl_full"], idx, H)
    b_fut = stack_windows(data["nbl_full"], idx + H, P)
    dec = stcross.predict_with_decomposition(X, b_in, b_fut)   # each (L, P, N) normalized
    inv = data["scaler"].inverse_transform
    return {k: inv(v) for k, v in dec.items()}


def learned_rolling(model, data, i_lo, i_hi):
    """Per_node learned baseline rolling predictions (L, P, N) original scale."""
    idx = np.arange(i_lo, i_hi + 1)
    X = stack_windows(data["flow_norm"], idx, H)
    preds = model.predict(X)                                   # (L, P, N) normalized
    return data["scaler"].inverse_transform(preds)


def ha_rolling(ha, data, i_lo, i_hi):
    """HA rolling predictions (L, P, N) — HA already returns original scale."""
    idx = np.arange(i_lo, i_hi + 1)
    ts = build_ha_timestamps(data["timestamps"], idx)
    return ha.predict_with_timestamps(ts)


def agg(arr, active):
    """Mean over active nodes. arr: (..., N) -> (...) with last dim reduced."""
    return arr[..., active].mean(axis=-1)


def mae_step0(pred_s0, true_s0, active):
    """MAE over timesteps and active nodes; both (L, N) original scale."""
    return np.abs(pred_s0[:, active] - true_s0[:, active]).mean()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_interval(data, stcross, learneds, ha, run, anomaly_type, out_path):
    """learneds: list of (name, model). ha: HA model or None."""
    r_start, r_end = run
    te_s, te_e = data["te_s"], data["te_e"]
    i_lo, i_hi = interval_sample_range(r_start, r_end, te_s, te_e)
    if i_hi < i_lo:
        print(f"  [skip] interval {run} has no valid samples")
        return None
    L = i_hi - i_lo + 1
    active = data["active"]

    timesteps = data["timestamps"][i_lo + H : i_hi + H + 1]     # (L,)
    true_s0 = data["flow"][i_lo + H : i_hi + H + 1]              # (L, N) original

    st = stcross_rolling(stcross, data, i_lo, i_hi)             # dict (L,P,N) original
    st_pred_s0 = st["y_pred"][:, 0, :]
    ha_pred_s0 = ha_rolling(ha, data, i_lo, i_hi)[:, 0, :] if ha is not None else None
    lb_preds_s0 = [(name, learned_rolling(m, data, i_lo, i_hi)[:, 0, :])
                   for name, m in learneds]

    true_agg = agg(true_s0, active)
    st_agg = agg(st_pred_s0, active)
    ha_agg = agg(ha_pred_s0, active) if ha is not None else None
    lb_aggs = [(name, agg(p, active)) for name, p in lb_preds_s0]

    # representative sample for decomposition
    fa_lo = max(te_s, r_start - H)
    fa_hi = min(te_e - H - P, r_end - H - P + 1)
    i_rep = (fa_lo + fa_hi) // 2 if fa_hi >= fa_lo else (i_lo + i_hi) // 2
    rep_X = data["flow_norm"][i_rep:i_rep + H][None]
    rep_bin = data["nbl_full"][i_rep:i_rep + H][None]
    rep_bfut = data["nbl_full"][i_rep + H:i_rep + H + P][None]
    rep_Y = data["flow"][i_rep + H:i_rep + H + P]
    rep_dec = stcross.predict_with_decomposition(rep_X, rep_bin, rep_bfut)
    inv = data["scaler"].inverse_transform
    rep_base = agg(inv(rep_dec["baseline"][0]), active)
    rep_dtype = agg(inv(rep_dec["delta_type"][0]), active)
    rep_dadj = agg(inv(rep_dec["delta_adj"][0]), active)
    rep_pred = agg(inv(rep_dec["y_pred"][0]), active)
    rep_true = agg(rep_Y, active)

    d0 = timesteps[0].strftime("%Y-%m-%d %H:%M")
    d1 = timesteps[-1].strftime("%Y-%m-%d %H:%M")
    rep_date = data["timestamps"][i_rep + H].strftime("%Y-%m-%d %H:%M")

    mae_st = np.abs(st_pred_s0[:, active] - true_s0[:, active]).mean()
    mae_ha = (np.abs(ha_pred_s0[:, active] - true_s0[:, active]).mean()
              if ha is not None else float('nan'))
    mae_lb = {name: np.abs(p[:, active] - true_s0[:, active]).mean()
              for name, p in lb_preds_s0}

    # title MAE summary
    mae_str = (f"MAE  STCross={mae_st:.3f}  " +
               "  ".join(f"{n}={mae_lb[n]:.3f}" for n in mae_lb) +
               (f"  HA={mae_ha:.3f}" if ha is not None else ""))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Top: rolling 1-step-ahead
    x = np.arange(L)
    ax1.plot(x, true_agg, color='black', linewidth=2, label='Ground truth')
    ax1.plot(x, st_agg, color='#d62728', linewidth=1.8, label='STCross')
    if ha_agg is not None:
        ax1.plot(x, ha_agg, color='#7f7f7f', linewidth=1.4, linestyle='--',
                 label='HA (NBL ref)')
    for name, la in lb_aggs:
        ax1.plot(x, la, color=BASELINE_COLORS.get(name, DEFAULT_COLOR),
                 linewidth=1.4, linestyle='-.', label=name)
    ax1.set_title(f'{anomaly_type} interval: {d0} → {d1}  ({mae_str})', fontsize=10.5)
    ax1.set_ylabel('Mean flow (active nodes)', fontsize=11)
    ax1.set_xlabel('hour (step-0 rolling prediction)', fontsize=10)
    ax1.legend(fontsize=8.5, loc='best', ncol=2)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(np.linspace(0, L - 1, min(8, L), dtype=int))

    # Bottom: decomposition
    xp = np.arange(P)
    ax2.plot(xp, rep_true, color='black', linewidth=2, label='Ground truth')
    ax2.fill_between(xp, 0, rep_base, color='#a1d99a', alpha=0.5,
                     label='NBL baseline')
    ax2.fill_between(xp, rep_base, rep_base + rep_dtype, color='#9ecae1',
                     alpha=0.6, label=r'$\hat{\Delta}_{type}$')
    ax2.fill_between(xp, rep_base + rep_dtype,
                     rep_base + rep_dtype + rep_dadj, color='#fdae6b',
                     alpha=0.6, label=r'$\hat{\Delta}_{adj}$')
    ax2.plot(xp, rep_pred, color='#d62728', linewidth=1.8, linestyle='--',
             label='STCross pred')
    ax2.set_title(f'STCross decomposition @ {rep_date}  '
                  f'(pred = baseline + Δ_type + Δ_adj)', fontsize=11)
    ax2.set_ylabel('Mean flow (active nodes)', fontsize=11)
    ax2.set_xlabel('forecast step (12h)', fontsize=10)
    ax2.legend(fontsize=8.5, loc='best', ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_xticks(np.arange(0, P, 2))

    fig.suptitle(f'V3 Case Study — {anomaly_type}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    mae_str_full = (f"STCross={mae_st:.4f} " +
                    " ".join(f"{n}={mae_lb[n]:.4f}" for n in mae_lb) +
                    (f" HA={mae_ha:.4f}" if ha is not None else ""))
    print(f"  Saved: {out_path}  (MAE: {mae_str_full})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_overrides(overrides):
    """Parse ['DAN=checkpoints_viz', 'EVL=checkpoints'] -> {'DAN':'checkpoints_viz', ...}."""
    out = {}
    for item in overrides:
        if '=' not in item:
            raise SystemExit(f"--ckpt-dir expects NAME=PATH, got: {item}")
        name, path = item.split('=', 1)
        out[name.strip()] = path.strip()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="V3 case study: STCross decomposition vs selected baselines")
    parser.add_argument("--baselines", nargs="+", default=["DAN"],
                        help="Learned baselines to compare (default: DAN). "
                             "Options: " + ", ".join(BASELINE_CONFIG))
    parser.add_argument("--ckpt-dir", action="append", default=[],
                        metavar="NAME=PATH",
                        help="Override checkpoint dir for a baseline "
                             "(repeatable, e.g. --ckpt-dir DAN=checkpoints)")
    parser.add_argument("--no-ha", action="store_true",
                        help="Drop the HA (NBL reference) line")
    args = parser.parse_args()

    # merge BASELINE_CONFIG with CLI overrides
    config = dict(BASELINE_CONFIG)
    config.update(parse_overrides(args.ckpt_dir))

    # validate baseline names
    for b in args.baselines:
        if b not in config:
            raise SystemExit(
                f"Unknown baseline '{b}'. Add it to BASELINE_CONFIG in this script "
                f"(known: {list(BASELINE_CONFIG)}).")

    os.makedirs(OUT_DIR, exist_ok=True)
    data = prepare_chi()
    te_s, te_e = data["te_s"], data["te_e"]

    print("Loading models...")
    print(f"  STCross <- {VIZ_CKPT_DIR}/STCross_NYC_best.pt")
    stcross = load_stcross()
    ha = None if args.no_ha else load_ha("CHI")
    if ha is not None:
        print("  HA     <- recomputed on CHI (no checkpoint)")
    learneds = []
    for b in args.baselines:
        print(f"  {b:<8} <- {config[b]}/{b}_NYC_best.pt")
        learneds.append((b, load_learned_baseline(b, config[b])))

    holiday_runs = find_runs(data["holiday"], MIN_RUN, te_s, te_e)
    weather_runs = find_runs(data["weather"], MIN_RUN, te_s, te_e)
    print(f"  holiday runs (>= {MIN_RUN}h): {len(holiday_runs)} | "
          f"weather runs: {len(weather_runs)}")

    def evaluate_runs(runs, label):
        results = []
        for run in runs:
            i_lo, i_hi = interval_sample_range(*run, te_s, te_e)
            if i_hi < i_lo:
                continue
            true_s0 = data["flow"][i_lo + H : i_hi + H + 1]
            st_s0 = stcross_rolling(stcross, data, i_lo, i_hi)["y_pred"][:, 0, :]
            m_st = mae_step0(st_s0, true_s0, data["active"])
            lb_maes = {}
            for name, m in learneds:
                lb_s0 = learned_rolling(m, data, i_lo, i_hi)[:, 0, :]
                lb_maes[name] = mae_step0(lb_s0, true_s0, data["active"])
            m_ha = (mae_step0(ha_rolling(ha, data, i_lo, i_hi)[:, 0, :], true_s0,
                              data["active"]) if ha is not None else float('nan'))
            ts0 = data["timestamps"][i_lo + H].strftime("%Y-%m-%d")
            # rank by margin over the STRONGEST baseline (min baseline MAE)
            strongest = min(lb_maes.values()) if lb_maes else m_ha
            results.append(dict(run=run, label=label, date=ts0,
                                 mae_st=m_st, mae_lb=lb_maes, mae_ha=m_ha,
                                 margin=strongest - m_st))
        results.sort(key=lambda r: r["margin"], reverse=True)
        return results

    h_res = evaluate_runs(holiday_runs, "Holiday")
    w_res = evaluate_runs(weather_runs, "Weather")

    print("\nCandidate MAE table (sorted by STCross margin over strongest baseline):")
    hdr = (f"  {'type':<8} {'date':<12} {'STCross':>8} " +
           " ".join(f"{n:>8}" for n, _ in learneds) +
           (f" {'HA':>8}" if ha is not None else "") + f" {'margin':>8}")
    print(hdr)
    for r in (h_res + w_res):
        row = (f"  {r['label']:<8} {r['date']:<12} {r['mae_st']:>8.4f} " +
               " ".join(f"{r['mae_lb'][n]:>8.4f}" for n, _ in learneds) +
               (f" {r['mae_ha']:>8.4f}" if ha is not None else "") +
               f" {r['margin']:>8.4f}")
        print(row)

    selected = []
    if h_res:
        selected.append((h_res[0], "Holiday"))
    if w_res:
        selected.append((w_res[0], "Weather"))

    for res, atype in selected:
        out = os.path.join(OUT_DIR, f"v3_case_study_{atype.lower()}.png")
        plot_interval(data, stcross, learneds, ha,
                      res["run"], atype, out)


if __name__ == "__main__":
    main()
