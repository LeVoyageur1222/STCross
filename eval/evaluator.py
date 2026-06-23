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
    def __init__(self, config: EvalConfig):
        self.config = config

    def run(self, model: BaseModel) -> dict:
        cfg = self.config

        target_flow = load_flow(cfg.target_city, cfg.flow_dir,
                                cfg.start_date, cfg.end_date, cfg.freq_min)
        target_mask = load_anomaly_mask(cfg.target_city, cfg.mask_dir_holiday,
                                        cfg.mask_dir_weather, cfg.start_date,
                                        cfg.end_date, cfg.freq_min)
        T, N = target_flow.shape
        H, P = cfg.history_len, cfg.predict_len

        (train_s, train_e), (val_s, val_e), (test_s, test_e) = split_data(T, cfg.split_ratio)

        scaler = PerNodeStandardScaler().fit(target_flow[train_s:train_e].astype(np.float64))

        if cfg.source_city != cfg.target_city and not isinstance(model, HAModel):
            source_flow = load_flow(cfg.source_city, cfg.flow_dir,
                                    cfg.start_date, cfg.end_date, cfg.freq_min)
            source_mask = load_anomaly_mask(cfg.source_city, cfg.mask_dir_holiday,
                                            cfg.mask_dir_weather, cfg.start_date,
                                            cfg.end_date, cfg.freq_min)
            s_T = source_flow.shape[0]
            (s_train_s, s_train_e), (s_val_s, s_val_e), _ = split_data(s_T, cfg.split_ratio)

            source_scaler = PerNodeStandardScaler().fit(
                source_flow[s_train_s:s_train_e].astype(np.float64))
            source_flow_norm = source_scaler.transform(source_flow.astype(np.float64))

            train_X, train_Y, _ = create_sliding_windows(
                source_flow_norm, source_mask, H, P, s_train_s, s_train_e, cfg.anomaly_min_ratio)
            val_X, val_Y, _ = create_sliding_windows(
                source_flow_norm, source_mask, H, P, s_val_s, s_val_e, cfg.anomaly_min_ratio)
        else:
            target_flow_norm = scaler.transform(target_flow.astype(np.float64))
            train_X, train_Y, _ = create_sliding_windows(
                target_flow_norm, target_mask, H, P, train_s, train_e, cfg.anomaly_min_ratio)
            val_X, val_Y, _ = create_sliding_windows(
                target_flow_norm, target_mask, H, P, val_s, val_e, cfg.anomaly_min_ratio)

        model.fit(train_X, train_Y, val_X, val_Y)

        if isinstance(model, HAModel):
            start_ts = pd.Timestamp(cfg.start_date)
            all_timestamps = pd.date_range(start_ts, periods=T, freq=f"{cfg.freq_min}min")

            test_indices = list(range(test_s, test_e - H - P + 1))
            num_test = len(test_indices)
            pred_timestamps = np.empty((num_test, P), dtype=object)
            for idx, i in enumerate(test_indices):
                for j in range(P):
                    pred_timestamps[idx, j] = all_timestamps[i + H + j]

            preds_raw = model.predict_with_timestamps(pred_timestamps)

            _, test_Y_norm, test_labels = create_sliding_windows(
                scaler.transform(target_flow.astype(np.float64)),
                target_mask, H, P, test_s, test_e, cfg.anomaly_min_ratio,
            )
            test_Y_raw = scaler.inverse_transform(test_Y_norm)
        else:
            target_flow_norm = scaler.transform(target_flow.astype(np.float64))
            test_X, test_Y_norm, test_labels = create_sliding_windows(
                target_flow_norm, target_mask, H, P, test_s, test_e, cfg.anomaly_min_ratio)
            preds_norm = model.predict(test_X)
            preds_raw = scaler.inverse_transform(preds_norm)
            test_Y_raw = scaler.inverse_transform(test_Y_norm)

        results = {
            "model": model.name,
            "source_city": cfg.source_city,
            "target_city": cfg.target_city,
            "history_len": H,
            "predict_len": P,
            "split_ratio": list(cfg.split_ratio),
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

            y_true = test_Y_raw[idx].flatten()
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
