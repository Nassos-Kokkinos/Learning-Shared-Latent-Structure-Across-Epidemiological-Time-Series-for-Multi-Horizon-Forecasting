import torch
from torch import nn


class NBeatsBlock(nn.Module):
    """Basic N-BEATS block with backcast and forecast heads."""

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers = []
        in_features = input_window

        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, input_window)
        self.forecast_head = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NBeatsForecaster(nn.Module):
    """Simple generic N-BEATS forecaster."""

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(
                    input_window=input_window,
                    forecast_horizon=forecast_horizon,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        forecast = torch.zeros(
            x.size(0),
            self.forecast_horizon,
            device=x.device,
            dtype=x.dtype,
        )

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast