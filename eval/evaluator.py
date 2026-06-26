import numpy as np
import pandas as pd

from data_preprocess.dataset import (
    load_flow, load_anomaly_mask, split_data, create_sliding_windows, get_subset_indices,
)
from data_preprocess.normalizer import PerNodeStandardScaler
from metrics.metrics import calc_mae, calc_rmse, calc_mape, calc_pcc, calc_r2
from model.base import BaseModel
from model.ha import HAModel
from .config import EvalConfig


class Evaluator:
    """Train-once-per-source, evaluate-on-multiple-targets evaluator.

    For per_node models (Group A/C/D): train() fits on source city data once,
    evaluate() runs inference on any target city's test set.

    For HAModel: train() is a no-op (HA builds its lookup from target city flow
    during evaluate(), so it cannot share training across targets).
    """

    def __init__(self, config: EvalConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, model: BaseModel, source_city: str) -> None:
        """Train model on source_city's train/val split.

        For HAModel, this is a no-op: HA computes its lookup table from the
        target city's flow during evaluate() (HA is target-specific by design).
        """
        if isinstance(model, HAModel):
            return

        cfg = self.config
        H, P = cfg.history_len, cfg.predict_len

        source_flow = load_flow(source_city, cfg.flow_dir,
                                cfg.start_date, cfg.end_date, cfg.freq_min)
        source_mask = load_anomaly_mask(source_city, cfg.mask_dir_holiday,
                                        cfg.mask_dir_weather, cfg.start_date,
                                        cfg.end_date, cfg.freq_min)
        s_T = source_flow.shape[0]
        (s_train_s, s_train_e), (s_val_s, s_val_e), _ = split_data(
            s_T, cfg.train_steps, cfg.val_steps)

        source_scaler = PerNodeStandardScaler().fit(
            source_flow[s_train_s:s_train_e].astype(np.float64))
        source_flow_norm = source_scaler.transform(source_flow.astype(np.float64))

        train_X, train_Y, _ = create_sliding_windows(
            source_flow_norm, source_mask, H, P,
            s_train_s, s_train_e, cfg.anomaly_min_ratio)
        val_X, val_Y, _ = create_sliding_windows(
            source_flow_norm, source_mask, H, P,
            s_val_s, s_val_e, cfg.anomaly_min_ratio)

        model.fit(train_X, train_Y, val_X, val_Y)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, model: BaseModel, source_city: str, target_city: str) -> dict:
        """Evaluate trained model on target_city's test split.

        For HAModel: builds lookup from target train flow, then predicts
        using timestamps (source_city is irrelevant for HA).
        For other models: normalizes test input by target scaler, runs
        model.predict(), inverse-transforms with target scaler.
        """
        cfg = self.config
        H, P = cfg.history_len, cfg.predict_len

        target_flow = load_flow(target_city, cfg.flow_dir,
                                cfg.start_date, cfg.end_date, cfg.freq_min)
        target_mask = load_anomaly_mask(target_city, cfg.mask_dir_holiday,
                                        cfg.mask_dir_weather, cfg.start_date,
                                        cfg.end_date, cfg.freq_min)
        T, N = target_flow.shape
        (train_s, train_e), (val_s, val_e), (test_s, test_e) = split_data(
            T, cfg.train_steps, cfg.val_steps)

        target_flow_f64 = target_flow.astype(np.float64)

        _, raw_test_Y, test_labels = create_sliding_windows(
            target_flow_f64, target_mask, H, P,
            test_s, test_e, cfg.anomaly_min_ratio)

        if isinstance(model, HAModel):
            model.flow_path = cfg.flow_path(target_city)
            model.target_city = target_city
            model.train_steps = cfg.train_steps
            model.fit(None, None, None, None)
            start_ts = pd.Timestamp(cfg.start_date)
            all_timestamps = pd.date_range(start_ts, periods=T, freq=f"{cfg.freq_min}min")
            test_indices = list(range(test_s, test_e - H - P + 1))
            num_test = len(test_indices)
            pred_timestamps = np.empty((num_test, P), dtype=object)
            for idx, i in enumerate(test_indices):
                for j in range(P):
                    pred_timestamps[idx, j] = all_timestamps[i + H + j]
            preds_raw = model.predict_with_timestamps(pred_timestamps)
        else:
            target_scaler = PerNodeStandardScaler().fit(
                target_flow[train_s:train_e].astype(np.float64))
            target_flow_norm = target_scaler.transform(target_flow_f64)
            test_X, _, _ = create_sliding_windows(
                target_flow_norm, target_mask, H, P,
                test_s, test_e, cfg.anomaly_min_ratio)
            preds_norm = model.predict(test_X)
            preds_raw = target_scaler.inverse_transform(preds_norm)

        results = {
            "model": model.name,
            "source_city": source_city,
            "target_city": target_city,
            "history_len": H,
            "predict_len": P,
            "train_steps": cfg.train_steps,
            "val_steps": cfg.val_steps,
            "test_steps": test_e - test_s,
            "anomaly_min_ratio": cfg.anomaly_min_ratio,
            "num_test_samples": len(test_labels),
            "subsets": {},
        }

        for subset in cfg.eval_subsets:
            idx = get_subset_indices(test_labels, subset)
            if len(idx) == 0:
                results["subsets"][subset] = {
                    "count": 0, "MAE": None, "RMSE": None,
                    "MAPE": None, "PCC": None, "R2": None,
                }
                continue

            y_true = raw_test_Y[idx].flatten()
            y_pred = preds_raw[idx].flatten()

            results["subsets"][subset] = {
                "count": int(len(idx)),
                "MAE": round(calc_mae(y_true, y_pred), 4),
                "RMSE": round(calc_rmse(y_true, y_pred), 4),
                "MAPE": round(calc_mape(y_true, y_pred), 2),
                "PCC": round(calc_pcc(y_true, y_pred), 4),
                "R2": round(calc_r2(y_true, y_pred), 4),
            }

        return results

    # ------------------------------------------------------------------
    # Legacy single-setting entry point (backward compat)
    # ------------------------------------------------------------------

    def run(self, model: BaseModel) -> dict:
        """Train on source_city and evaluate on target_city (single setting).

        Convenience wrapper: equivalent to train(model, source) then
        evaluate(model, source, target).
        """
        cfg = self.config
        self.train(model, cfg.source_city)
        return self.evaluate(model, cfg.source_city, cfg.target_city)
