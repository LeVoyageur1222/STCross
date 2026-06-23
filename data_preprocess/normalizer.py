import numpy as np


class PerNodeStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_flow: np.ndarray) -> "PerNodeStandardScaler":
        self.mean = train_flow.mean(axis=0)
        self.std = train_flow.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, flow: np.ndarray) -> np.ndarray:
        return (flow - self.mean) / self.std

    def inverse_transform(self, flow: np.ndarray) -> np.ndarray:
        return flow * self.std + self.mean
