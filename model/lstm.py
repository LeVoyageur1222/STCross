import torch
import torch.nn as nn

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class LSTMModel(TorchModel):
    """2-layer LSTM baseline for per-node traffic flow prediction."""

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.1,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.P = P
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, P)

    @property
    def name(self) -> str:
        return "LSTM"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, 1)
        out, _ = self.lstm(X)        # (B, H, hidden)
        pred = self.fc(out[:, -1, :])  # (B, P)
        return pred.unsqueeze(-1)    # (B, P, 1)
