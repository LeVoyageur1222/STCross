import numpy as np
from scipy.stats import pearsonr


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """MAPE computed only on ground truth values above threshold (percentage)."""
    if len(y_true) == 0:
        return float("nan")
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
