import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrafficFlowDataset(Dataset):
    """PyTorch Dataset wrapping sliding-window numpy arrays.

    Always stores data in all_nodes format: (num_samples, H, N) / (num_samples, P, N).
    Per-node reshaping is handled inside the model's forward pass for efficiency.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, labels: np.ndarray):
        """
        Args:
            X: (num_samples, H, N) float64
            Y: (num_samples, P, N) float64
            labels: (num_samples,) bool
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        self.labels = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def build_dataloader(X: np.ndarray, Y: np.ndarray, labels: np.ndarray,
                     batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    dataset = TrafficFlowDataset(X, Y, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())
