from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
