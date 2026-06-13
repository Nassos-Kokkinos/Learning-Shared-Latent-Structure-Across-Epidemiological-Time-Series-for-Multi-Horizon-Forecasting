import torch
import torch.nn.functional as F
from torch import nn


class NHiTSBlock(nn.Module):
    """Simplified N-HiTS-style block with pooling and interpolation."""

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        pooling_kernel_size: int = 1,
        downsample_frequency: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if pooling_kernel_size < 1:
            raise ValueError("pooling_kernel_size must be >= 1.")
        if downsample_frequency < 1:
            raise ValueError("downsample_frequency must be >= 1.")

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.pooling_kernel_size = pooling_kernel_size
        self.downsample_frequency = downsample_frequency

        pooled_input_size = (input_window + pooling_kernel_size - 1) // pooling_kernel_size
        n_theta_forecast = max(1, forecast_horizon // downsample_frequency)

        layers = []
        in_features = pooled_input_size

        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden_dim, pooled_input_size)
        self.forecast_head = nn.Linear(hidden_dim, n_theta_forecast)

    def _pool_history(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        pooled = F.max_pool1d(
            x,
            kernel_size=self.pooling_kernel_size,
            stride=self.pooling_kernel_size,
            ceil_mode=True,
        )
        return pooled.squeeze(1)

    def _upsample_forecast(self, theta: torch.Tensor) -> torch.Tensor:
        forecast = F.interpolate(
            theta.unsqueeze(1),
            size=self.forecast_horizon,
            mode="linear",
            align_corners=False,
        )
        return forecast.squeeze(1)

    def _upsample_backcast(self, theta: torch.Tensor) -> torch.Tensor:
        backcast = F.interpolate(
            theta.unsqueeze(1),
            size=self.input_window,
            mode="linear",
            align_corners=False,
        )
        return backcast.squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_x = self._pool_history(x)
        h = self.mlp(pooled_x)

        backcast_theta = self.backcast_head(h)
        forecast_theta = self.forecast_head(h)

        backcast = self._upsample_backcast(backcast_theta)
        forecast = self._upsample_forecast(forecast_theta)

        return backcast, forecast


class NHiTSForecaster(nn.Module):
    """Simplified N-HiTS-style global forecaster."""

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        hidden_dim: int = 256,
        n_stacks: int = 3,
        n_blocks_per_stack: int = 1,
        n_layers: int = 2,
        pooling_kernel_sizes: list[int] | None = None,
        downsample_frequencies: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if pooling_kernel_sizes is None:
            pooling_kernel_sizes = [1, 2, 4]
        if downsample_frequencies is None:
            downsample_frequencies = [4, 2, 1]

        if len(pooling_kernel_sizes) != n_stacks:
            raise ValueError("pooling_kernel_sizes length must equal n_stacks.")
        if len(downsample_frequencies) != n_stacks:
            raise ValueError("downsample_frequencies length must equal n_stacks.")

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        blocks = []
        for stack_idx in range(n_stacks):
            for _ in range(n_blocks_per_stack):
                blocks.append(
                    NHiTSBlock(
                        input_window=input_window,
                        forecast_horizon=forecast_horizon,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        pooling_kernel_size=pooling_kernel_sizes[stack_idx],
                        downsample_frequency=downsample_frequencies[stack_idx],
                        dropout=dropout,
                    )
                )

        self.blocks = nn.ModuleList(blocks)

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