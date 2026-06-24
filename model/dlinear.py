import torch
import torch.nn as nn

from model.torch_base import TorchModel
from train.trainer import TrainConfig


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, 1) → pool over H dim
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class DLinearModel(TorchModel):
    """DLinear: trend-seasonal decomposition + dual linear layers.

    Reference: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
    """

    input_mode = "per_node"

    def __init__(self, H: int = 12, P: int = 12, kernel_size: int = 3,
                 train_config: TrainConfig = None):
        super().__init__(train_config=train_config)
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(H, P)
        self.linear_seasonal = nn.Linear(H, P)

    @property
    def name(self) -> str:
        return "DLinear"

    def _net_forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, H, 1)
        trend = self.decomp(X)           # (B, H, 1)
        seasonal = X - trend             # (B, H, 1)

        trend_out = self.linear_trend(trend.squeeze(-1))        # (B, P)
        seasonal_out = self.linear_seasonal(seasonal.squeeze(-1))  # (B, P)

        return (trend_out + seasonal_out).unsqueeze(-1)  # (B, P, 1)
