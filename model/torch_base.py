from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn

from model.base import BaseModel
from train.trainer import Trainer, TrainConfig
from data_preprocess.torch_dataset import build_dataloader


class TorchModel(BaseModel, nn.Module):
    """Base class for PyTorch-based models.

    Bridges the numpy-based BaseModel interface with PyTorch nn.Module.
    Subclasses implement _net_forward() and name; fit/predict/forward are handled here.

    DataLoader always uses all_nodes format: batch shape (B, H, N) / (B, P, N).
    For per_node models, forward() reshapes (B,H,N)→(B*N,H,1) before calling
    _net_forward(), then reshapes back to (B,P,N). Trainer just calls model(X).

    Attributes:
        input_mode: "per_node" for channel-independent models (Group A),
                    "all_nodes" for graph-based models (Group B).
        train_config: TrainConfig instance controlling training hyperparameters.
    """

    input_mode: str = "per_node"

    def __init__(self, train_config: TrainConfig = None):
        nn.Module.__init__(self)
        if self.input_mode not in ("per_node", "all_nodes"):
            raise ValueError(f"Invalid input_mode: {self.input_mode!r}")
        self.train_config = train_config or TrainConfig()
        self._device = self.train_config.resolve_device()

    @abstractmethod
    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        """Subclass network forward.
        - per_node models: X is (B', H, 1), return (B', P, 1)
        - all_nodes models: X is (B, H, N), return (B, P, N)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Called by Trainer. Handles per_node reshape around _net_forward()."""
        if self.input_mode == "per_node":
            B, H, N = X.shape
            X_flat = X.permute(0, 2, 1).reshape(B * N, H, 1)
            out = self._net_forward(X_flat)
            assert out.dim() == 3 and out.shape[0] == B * N and out.shape[2] == 1, \
                f"per_node _net_forward() must return (B*N, P, 1), got {tuple(out.shape)}"
            P = out.shape[1]
            return out.reshape(B, N, P).permute(0, 2, 1)  # (B, P, N)
        return self._net_forward(X)

    def fit(self, train_X: np.ndarray, train_Y: np.ndarray,
            val_X: np.ndarray, val_Y: np.ndarray) -> None:
        cfg = self.train_config
        train_loader = build_dataloader(
            train_X, train_Y, np.zeros(len(train_X), dtype=bool),
            batch_size=cfg.batch_size, shuffle=True)
        val_loader = build_dataloader(
            val_X, val_Y, np.zeros(len(val_X), dtype=bool),
            batch_size=cfg.batch_size, shuffle=False)

        print(f"[{self.name}] Training on {self._device} | "
              f"train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

        trainer = Trainer(cfg)
        trainer.train(self, train_loader, val_loader, model_name=self.name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        self.to(self._device)
        X_tensor = torch.from_numpy(X).float()
        results = []
        batch_size = self.train_config.batch_size * 2
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size].to(self._device)
                out = self.forward(batch).cpu().numpy()
                results.append(out)
        return np.concatenate(results, axis=0)
