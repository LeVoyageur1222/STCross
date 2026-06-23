import numpy as np
from typing import Tuple


def load_flow(city: str, flow_dir: str, start_date: str, end_date: str, freq_min: int) -> np.ndarray:
    path = f"{flow_dir}/flow_{city}_{start_date}_{end_date}_{freq_min}min.npy"
    return np.load(path)


def load_anomaly_mask(city: str, mask_dir_holiday: str, mask_dir_weather: str,
                      start_date: str, end_date: str, freq_min: int) -> np.ndarray:
    holiday_path = f"{mask_dir_holiday}/mask_{city}_holiday_{start_date}_{end_date}_{freq_min}min.npy"
    weather_path = f"{mask_dir_weather}/mask_{city}_all_{start_date}_{end_date}_{freq_min}min.npy"
    holiday = np.load(holiday_path).astype(bool)
    weather = np.load(weather_path).astype(bool)
    return holiday | weather


def split_data(T: int, split_ratio: Tuple[float, float, float]) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    t_train = int(T * split_ratio[0])
    t_val = int(T * (split_ratio[0] + split_ratio[1]))
    return (0, t_train), (t_train, t_val), (t_val, T)


def create_sliding_windows(
    flow: np.ndarray,
    mask: np.ndarray,
    H: int,
    P: int,
    start: int,
    end: int,
    anomaly_min_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding window samples from a specific time range.

    Args:
        anomaly_min_ratio: minimum fraction of anomalous steps in prediction window
            to label sample as anomaly. 0.0=any step, 0.5=majority, 1.0=all steps.

    Returns (X, Y, sample_labels) where:
        X: (num_samples, H, N)
        Y: (num_samples, P, N)
        sample_labels: (num_samples,) bool
    """
    num_samples = max(0, end - H - P + 1 - start)
    N = flow.shape[1]
    X = np.empty((num_samples, H, N), dtype=flow.dtype)
    Y = np.empty((num_samples, P, N), dtype=flow.dtype)
    sample_labels = np.empty(num_samples, dtype=bool)

    for idx, i in enumerate(range(start, end - H - P + 1)):
        X[idx] = flow[i: i + H]
        Y[idx] = flow[i + H: i + H + P]
        ratio = mask[i + H: i + H + P].mean()
        sample_labels[idx] = ratio > anomaly_min_ratio

    return X, Y, sample_labels


def get_subset_indices(sample_labels: np.ndarray, subset: str) -> np.ndarray:
    if subset == "full":
        return np.arange(len(sample_labels))
    elif subset == "anomaly":
        return np.where(sample_labels)[0]
    elif subset == "normal":
        return np.where(~sample_labels)[0]
    else:
        raise ValueError(f"Unknown subset: {subset}")
