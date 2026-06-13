import torch
from torch import nn


class LatentSharedForecaster(nn.Module):
    """
    Simple shared latent forecasting model.

    Input:  past window of shape (batch, input_window)
    Output: forecast horizon of shape (batch, forecast_horizon)
    """

    def __init__(
        self,
        input_window: int,
        forecast_horizon: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_window, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.latent_layer = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch, input_window)
        returns: tensor of shape (batch, forecast_horizon)
        """
        features = self.encoder(x)
        z = self.latent_layer(features)
        y_hat = self.head(z)
        return y_hat