import numpy as np
import pandas as pd

from .base import BaseModel


class HAModel(BaseModel):
    """Historical Average: per-(hour_of_day, day_of_week, node) mean from training data."""

    def __init__(self, target_city: str, flow_path: str, start_date: str,
                 freq_min: int, train_steps: int):
        self.target_city = target_city
        self.flow_path = flow_path
        self.start_date = start_date
        self.freq_min = freq_min
        self.train_steps = train_steps
        self.lookup = None  # (24, 7, N)

    @property
    def name(self) -> str:
        return "HA"

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        flow = np.load(self.flow_path)
        T = flow.shape[0]
        train_end = self.train_steps
        train_flow = flow[:train_end]

        start = pd.Timestamp(self.start_date)
        timestamps = pd.date_range(start, periods=T, freq=f"{self.freq_min}min")
        train_ts = timestamps[:train_end]

        hours = train_ts.hour
        weekdays = train_ts.dayofweek

        N = train_flow.shape[1]
        self.lookup = np.zeros((24, 7, N), dtype=np.float64)
        counts = np.zeros((24, 7), dtype=np.int64)

        for t in range(len(train_flow)):
            h, w = hours[t], weekdays[t]
            self.lookup[h, w] += train_flow[t]
            counts[h, w] += 1

        for h in range(24):
            for w in range(7):
                if counts[h, w] > 0:
                    self.lookup[h, w] /= counts[h, w]

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use predict_with_timestamps instead")

    def predict_with_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """Predict using timestamps of prediction window steps.

        Args:
            timestamps: (num_samples, P) array of pd.Timestamp
        Returns:
            predictions: (num_samples, P, N)
        """
        num_samples, P = timestamps.shape
        N = self.lookup.shape[2]
        preds = np.empty((num_samples, P, N), dtype=np.float64)

        for i in range(num_samples):
            for j in range(P):
                ts = timestamps[i, j]
                preds[i, j] = self.lookup[ts.hour, ts.dayofweek]

        return preds
