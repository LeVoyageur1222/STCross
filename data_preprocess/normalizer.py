import warnings

import numpy as np


class PerNodeStandardScaler:
    """Per-node standard scaler (z-score normalization) with optional clipping.

    Clipping rationale: some nodes have near-zero std in training but occasional
    spikes in val/test. After normalization, these spikes become extreme values
    (up to 383 std in our NYC data) that dominate MSE and destabilize training.
    Clipping to [-clip_range, clip_range] removes ~0.1% extreme outliers while
    preserving 99.9% of the data distribution.

    Important: clip applies to normalized data used for model INPUT and training
    TARGET. Raw test Y (for metric computation) is never clipped — evaluation
    remains on true values, ensuring fair comparison.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_flow: np.ndarray) -> "PerNodeStandardScaler":
        self.mean = train_flow.mean(axis=0)
        self.std = train_flow.std(axis=0)
        n_dead = int((self.std < 1e-8).sum())
        if n_dead > 0:
            warnings.warn(f"PerNodeStandardScaler: {n_dead}/{len(self.std)} nodes "
                          f"have near-zero std (set to 1.0)")
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, flow: np.ndarray, clip_range: float = None) -> np.ndarray:
        """Normalize: (x - mean) / std. Optionally clip to [-clip_range, clip_range].

        Args:
            flow: raw flow data of shape (T, N).
            clip_range: if not None, clip normalized values to this range.
                        Recommended: 10.0 (preserves 99.9% of data).
        """
        normalized = (flow - self.mean) / self.std
        if clip_range is not None:
            normalized = np.clip(normalized, -clip_range, clip_range)
        return normalized

    def inverse_transform(self, flow: np.ndarray) -> np.ndarray:
        return flow * self.std + self.mean
