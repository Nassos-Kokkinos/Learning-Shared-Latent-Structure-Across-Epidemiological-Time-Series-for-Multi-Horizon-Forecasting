import torch
from torch import nn


class PathogenSpecificHead(nn.Module):
    """Small pathogen-specific prediction head."""

    def __init__(
        self,
        latent_dim: int,
        head_hidden_dim: int,
        forecast_horizon: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, forecast_horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentSharedForecasterV2(nn.Module):
    """
    Shared encoder + common latent space + pathogen-specific heads.
    """

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        num_series: int,
        encoder_hidden_dim: int = 256,
        latent_dim: int = 128,
        series_embedding_dim: int = 16,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_window_normalization: bool = True,
        use_linear_skip: bool = True,
    ) -> None:
        super().__init__()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.num_series = num_series
        self.use_window_normalization = use_window_normalization
        self.use_linear_skip = use_linear_skip

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_window, encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.ReLU(),
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(encoder_hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.series_embedding = nn.Embedding(num_series, series_embedding_dim)

        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim + series_embedding_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList(
            [
                PathogenSpecificHead(
                    latent_dim=latent_dim,
                    head_hidden_dim=head_hidden_dim,
                    forecast_horizon=forecast_horizon,
                    dropout=dropout,
                )
                for _ in range(num_series)
            ]
        )

        if self.use_linear_skip:
            self.linear_skip = nn.Linear(input_window, forecast_horizon)

    def forward(self, x: torch.Tensor, series_idx: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_window)
        series_idx: (batch,)
        returns: (batch, forecast_horizon)
        """
        if self.use_window_normalization:
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            x_in = (x - mu) / sigma
        else:
            mu = torch.zeros_like(x[:, :1])
            sigma = torch.ones_like(x[:, :1])
            x_in = x

        features = self.shared_encoder(x_in)
        z = self.latent_layer(features)

        emb = self.series_embedding(series_idx)
        fused = self.fusion_layer(torch.cat([z, emb], dim=1))

        out = torch.zeros(
            x.size(0),
            self.forecast_horizon,
            device=x.device,
            dtype=x.dtype,
        )

        for head_idx, head in enumerate(self.heads):
            mask = series_idx == head_idx
            if mask.any():
                out[mask] = head(fused[mask])

        if self.use_window_normalization:
            out = out * sigma + mu

        if self.use_linear_skip:
            out = out + self.linear_skip(x)

        return out
