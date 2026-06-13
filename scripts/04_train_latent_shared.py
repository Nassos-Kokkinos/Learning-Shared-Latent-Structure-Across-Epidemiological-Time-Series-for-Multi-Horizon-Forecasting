from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.dataset import MultiSeriesWindowDataset
from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import temporal_split_by_series
from epi_forecasting.evaluation.metrics import mae, rmse
from epi_forecasting.models.latent_shared import LatentSharedForecaster
from epi_forecasting.training.engine import train_one_epoch
from epi_forecasting.training.utils import create_train_dataloader
from epi_forecasting.utils.paths import CONFIGS_DIR, RESULTS_DIR
from epi_forecasting.utils.seed import set_seed


DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"
MODEL_CONFIG_FILE = CONFIGS_DIR / "latent_shared.yaml"

MODEL_OUTPUT_FILE = RESULTS_DIR / "models" / "latent_shared.pt"
HISTORY_OUTPUT_FILE = RESULTS_DIR / "metrics" / "latent_shared_train_history.csv"
METRICS_OUTPUT_FILE = RESULTS_DIR / "metrics" / "latent_shared_eval_metrics.csv"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def forecast_one_series(
    model: nn.Module,
    history: np.ndarray,
    horizon: int,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
    y_hat = model(x)
    return y_hat.squeeze(0).cpu().numpy()


@torch.no_grad()
def evaluate_one_shot(
    model: nn.Module,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    split_name: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()

    rows = []

    for series_id in sorted(eval_df["series_id"].unique()):
        history_series = (
            history_df.loc[history_df["series_id"] == series_id]
            .sort_values("ds")
            .reset_index(drop=True)
        )
        eval_series = (
            eval_df.loc[eval_df["series_id"] == series_id]
            .sort_values("ds")
            .reset_index(drop=True)
        )

        history = history_series["y"].to_numpy(dtype=np.float32)
        target = eval_series["y"].to_numpy(dtype=np.float32)

        if len(history) < input_window:
            raise ValueError(
                f"Series '{series_id}' has only {len(history)} history points, "
                f"but input_window={input_window}."
            )

        if len(target) < forecast_horizon:
            raise ValueError(
                f"Series '{series_id}' has only {len(target)} eval points, "
                f"but forecast_horizon={forecast_horizon}."
            )

        history_window = history[-input_window:]
        target_window = target[:forecast_horizon]
        pred_window = forecast_one_series(
            model=model,
            history=history_window,
            horizon=forecast_horizon,
            device=device,
        )

        rows.append(
            {
                "split": split_name,
                "series_id": series_id,
                "mae": mae(target_window, pred_window),
                "rmse": rmse(target_window, pred_window),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    data_config = load_yaml(DATA_CONFIG_FILE)
    model_config = load_yaml(MODEL_CONFIG_FILE)

    seed = int(data_config["seed"])
    input_window = int(data_config["input_window"])
    forecast_horizon = int(data_config["forecast_horizon"])
    val_size = int(data_config["val_size"])
    test_size = int(data_config["test_size"])

    hidden_dim = int(model_config["hidden_dim"])
    latent_dim = int(model_config["latent_dim"])
    dropout = float(model_config["dropout"])
    batch_size = int(model_config["batch_size"])
    learning_rate = float(model_config["learning_rate"])
    weight_decay = float(model_config["weight_decay"])
    epochs = int(model_config["epochs"])

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_model_ready_dataset()
    train_df, val_df, test_df = temporal_split_by_series(
        df,
        val_size=val_size,
        test_size=test_size,
    )

    train_dataset = MultiSeriesWindowDataset(
        df=train_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    train_loader = create_train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = LatentSharedForecaster(
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    history_rows = []

    print("\n=== TRAINING START ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
            }
        )

        print(f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.6f}")

    history_df = pd.DataFrame(history_rows)

    val_metrics_df = evaluate_one_shot(
        model=model,
        history_df=train_df,
        eval_df=val_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        split_name="val",
        device=device,
    )

    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)

    test_metrics_df = evaluate_one_shot(
        model=model,
        history_df=train_plus_val_df,
        eval_df=test_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        split_name="test",
        device=device,
    )

    metrics_df = pd.concat([val_metrics_df, test_metrics_df], ignore_index=True)
    summary_df = (
        metrics_df.groupby("split")[["mae", "rmse"]]
        .mean()
        .reset_index()
    )

    MODEL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), MODEL_OUTPUT_FILE)
    history_df.to_csv(HISTORY_OUTPUT_FILE, index=False)
    metrics_df.to_csv(METRICS_OUTPUT_FILE, index=False)

    print("\nSaved model to:")
    print(MODEL_OUTPUT_FILE)

    print("\nSaved training history to:")
    print(HISTORY_OUTPUT_FILE)

    print("\nSaved evaluation metrics to:")
    print(METRICS_OUTPUT_FILE)

    print("\n=== PER-SERIES METRICS ===")
    print(metrics_df)

    print("\n=== AVERAGE METRICS BY SPLIT ===")
    print(summary_df)


if __name__ == "__main__":
    main()