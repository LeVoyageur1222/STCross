import numpy as np
from scipy.stats import pearsonr


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """MAPE computed only on ground truth values above threshold (percentage).

    Default threshold=0.5 filters out zero-count entries robustly against
    floating-point noise from inverse normalization.
    """
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calc_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return float(r)


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
