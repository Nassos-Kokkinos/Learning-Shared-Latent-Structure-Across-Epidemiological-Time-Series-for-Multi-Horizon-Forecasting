from pathlib import Path
import sys
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.dataset_v2 import MultiSeriesWindowDatasetV2
from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import temporal_split_by_series
from epi_forecasting.evaluation.metrics import mae, rmse
from epi_forecasting.models.latent_shared_v2 import LatentSharedForecasterV2
from epi_forecasting.training.utils import create_train_dataloader
from epi_forecasting.utils.paths import CONFIGS_DIR, RESULTS_DIR
from epi_forecasting.utils.seed import set_seed


DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"
MODEL_CONFIG_FILE = CONFIGS_DIR / "latent_shared_v2.yaml"

MODEL_OUTPUT_FILE = RESULTS_DIR / "models" / "latent_shared_v2.pt"
HISTORY_OUTPUT_FILE = RESULTS_DIR / "metrics" / "latent_shared_v2_train_history.csv"
METRICS_OUTPUT_FILE = RESULTS_DIR / "metrics" / "latent_shared_v2_eval_metrics.csv"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch_series(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        series_idx = batch["series_idx"].to(device)

        optimizer.zero_grad()
        y_hat = model(x, series_idx)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("No samples were seen during training.")

    return running_loss / n_samples


@torch.no_grad()
def forecast_one_series(
    model: nn.Module,
    history: np.ndarray,
    series_idx: int,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
    series_idx_tensor = torch.tensor([series_idx], dtype=torch.long).to(device)
    y_hat = model(x, series_idx_tensor)
    return y_hat.squeeze(0).cpu().numpy()


@torch.no_grad()
def evaluate_one_shot(
    model: nn.Module,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    split_name: str,
    series_to_idx: dict[str, int],
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
        series_idx = series_to_idx[str(series_id)]

        pred_window = forecast_one_series(
            model=model,
            history=history_window,
            series_idx=series_idx,
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

    encoder_hidden_dim = int(model_config["encoder_hidden_dim"])
    latent_dim = int(model_config["latent_dim"])
    series_embedding_dim = int(model_config["series_embedding_dim"])
    head_hidden_dim = int(model_config["head_hidden_dim"])
    dropout = float(model_config["dropout"])
    use_window_normalization = bool(model_config["use_window_normalization"])
    use_linear_skip = bool(model_config["use_linear_skip"])

    batch_size = int(model_config["batch_size"])
    learning_rate = float(model_config["learning_rate"])
    weight_decay = float(model_config["weight_decay"])
    epochs = int(model_config["epochs"])
    patience = int(model_config["patience"])
    min_delta = float(model_config["min_delta"])

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_model_ready_dataset()
    train_df, val_df, test_df = temporal_split_by_series(
        df,
        val_size=val_size,
        test_size=test_size,
    )

    train_dataset = MultiSeriesWindowDatasetV2(
        df=train_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    train_loader = create_train_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = LatentSharedForecasterV2(
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        num_series=train_dataset.num_series,
        encoder_hidden_dim=encoder_hidden_dim,
        latent_dim=latent_dim,
        series_embedding_dim=series_embedding_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        use_window_normalization=use_window_normalization,
        use_linear_skip=use_linear_skip,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_mae = float("inf")
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    history_rows = []

    print("\n=== TRAINING START ===")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_series(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_metrics_df = evaluate_one_shot(
            model=model,
            history_df=train_df,
            eval_df=val_df,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            split_name="val",
            series_to_idx=train_dataset.series_to_idx,
            device=device,
        )

        val_mae = float(val_metrics_df["mae"].mean())
        val_rmse = float(val_metrics_df["rmse"].mean())

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_mae={val_mae:.6f} | "
            f"val_rmse={val_rmse:.6f}"
        )

        if val_mae < best_val_mae - min_delta:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_mae={best_val_mae:.6f}."
            )
            break

    model.load_state_dict(best_state_dict)

    history_df = pd.DataFrame(history_rows)

    val_metrics_df = evaluate_one_shot(
        model=model,
        history_df=train_df,
        eval_df=val_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        split_name="val",
        series_to_idx=train_dataset.series_to_idx,
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
        series_to_idx=train_dataset.series_to_idx,
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

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation MAE: {best_val_mae:.6f}")

    print("\n=== PER-SERIES METRICS ===")
    print(metrics_df)

    print("\n=== AVERAGE METRICS BY SPLIT ===")
    print(summary_df)


if __name__ == "__main__":
    main()
