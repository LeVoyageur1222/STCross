from abc import abstractmethod
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

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
        self._source_city = ""  # set by Evaluator.train() for per-source checkpoints

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

    def _get_checkpoint_name(self) -> str:
        """Checkpoint name with optional source-city suffix.

        If Evaluator.train() sets ``self._source_city`` before calling fit(),
        the checkpoint is saved as ``{name}_{source}_best.pt`` so that
        different cities' best checkpoints don't overwrite each other.
        Falls back to ``self.name`` when unset (legacy callers).
        """
        source = getattr(self, "_source_city", "")
        return f"{self.name}_{source}" if source else self.name

    def load_checkpoint(self, source_city: str) -> str:
        """Load a previously-trained checkpoint for the given source city.

        Used by --eval_only mode: skip training, load existing best weights,
        then evaluate. The checkpoint must have been saved by a prior
        ``fit()`` call with the same source city.

        Returns:
            Path to the loaded checkpoint.

        Raises:
            FileNotFoundError: if no checkpoint exists for this source city.
        """
        self._source_city = source_city
        ckpt_name = self._get_checkpoint_name()
        ckpt_path = os.path.abspath(os.path.join(
            self.train_config.checkpoint_dir, f"{ckpt_name}_best.pt"))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found for {self.name} on {source_city}: "
                f"{ckpt_path}")
        self.load_state_dict(torch.load(
            ckpt_path, map_location=self._device, weights_only=True))
        self.to(self._device)
        self.eval()
        print(f"  [EVAL ONLY] Loaded checkpoint: {ckpt_path}")
        return ckpt_path

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
        trainer.train(self, train_loader, val_loader,
                       model_name=self._get_checkpoint_name())

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        self.to(self._device)
        X_tensor = torch.from_numpy(X).float()
        results = []
        # For per_node models, forward() reshapes (B, H, N) → (B*N, H, 1).
        # The effective batch seen by _net_forward is B*N, which can exceed
        # CUDA kernel grid limits (e.g., nn.TransformerEncoder's fused kernel
        # ~65536) when B is large. Cap the effective batch to a safe limit.
        max_effective_batch = 32768
        if self.input_mode == "per_node":
            N = X_tensor.shape[2]
            batch_size = min(
                self.train_config.batch_size * 2,
                max(1, max_effective_batch // max(N, 1))
            )
        else:
            batch_size = self.train_config.batch_size * 2
        n_batches = (len(X_tensor) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(X_tensor), batch_size),
                          total=n_batches, desc="Predict", unit="batch", leave=False):
                batch = X_tensor[i:i + batch_size].to(self._device)
                out = self.forward(batch).cpu().numpy()
                results.append(out)
        return np.concatenate(results, axis=0)
