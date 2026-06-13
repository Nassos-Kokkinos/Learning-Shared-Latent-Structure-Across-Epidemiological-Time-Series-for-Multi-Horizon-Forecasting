from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import temporal_split_by_series
from epi_forecasting.models.latent_shared_v2 import LatentSharedForecasterV2
from epi_forecasting.models.nbeats import NBeatsForecaster
from epi_forecasting.models.nhits import NHiTSForecaster
from epi_forecasting.utils.paths import CONFIGS_DIR, RESULTS_DIR


DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"
LATENT_V2_CONFIG_FILE = CONFIGS_DIR / "latent_shared_v2.yaml"
NBEATS_CONFIG_FILE = CONFIGS_DIR / "baseline_nbeats.yaml"
NHITS_CONFIG_FILE = CONFIGS_DIR / "baseline_nhits.yaml"

FORECASTS_DIR = RESULTS_DIR / "forecasts"
MODELS_DIR = RESULTS_DIR / "models"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_series_index_mapping(df: pd.DataFrame) -> dict[str, int]:
    series_ids = sorted(df["series_id"].astype(str).unique().tolist())
    return {series_id: idx for idx, series_id in enumerate(series_ids)}


def build_latent_v2_model(
    data_config: dict,
    model_config: dict,
    num_series: int,
    device: torch.device,
) -> torch.nn.Module:
    model = LatentSharedForecasterV2(
        input_window=int(data_config["input_window"]),
        forecast_horizon=int(data_config["forecast_horizon"]),
        num_series=num_series,
        encoder_hidden_dim=int(model_config["encoder_hidden_dim"]),
        latent_dim=int(model_config["latent_dim"]),
        series_embedding_dim=int(model_config["series_embedding_dim"]),
        head_hidden_dim=int(model_config["head_hidden_dim"]),
        dropout=float(model_config["dropout"]),
        use_window_normalization=bool(model_config["use_window_normalization"]),
        use_linear_skip=bool(model_config["use_linear_skip"]),
    )
    checkpoint_path = MODELS_DIR / "latent_shared_v2.pt"
    return load_checkpoint(model, checkpoint_path, device)


def build_nbeats_model(
    data_config: dict,
    model_config: dict,
    device: torch.device,
) -> torch.nn.Module:
    model = NBeatsForecaster(
        input_window=int(data_config["input_window"]),
        forecast_horizon=int(data_config["forecast_horizon"]),
        hidden_dim=int(model_config["hidden_dim"]),
        n_blocks=int(model_config["n_blocks"]),
        n_layers=int(model_config["n_layers"]),
        dropout=float(model_config["dropout"]),
    )
    checkpoint_path = MODELS_DIR / "nbeats.pt"
    return load_checkpoint(model, checkpoint_path, device)


def build_nhits_model(
    data_config: dict,
    model_config: dict,
    device: torch.device,
) -> torch.nn.Module:
    model = NHiTSForecaster(
        input_window=int(data_config["input_window"]),
        forecast_horizon=int(data_config["forecast_horizon"]),
        hidden_dim=int(model_config["hidden_dim"]),
        n_stacks=int(model_config["n_stacks"]),
        n_blocks_per_stack=int(model_config["n_blocks_per_stack"]),
        n_layers=int(model_config["n_layers"]),
        pooling_kernel_sizes=[int(x) for x in model_config["pooling_kernel_sizes"]],
        downsample_frequencies=[int(x) for x in model_config["downsample_frequencies"]],
        dropout=float(model_config["dropout"]),
    )
    checkpoint_path = MODELS_DIR / "nhits.pt"
    return load_checkpoint(model, checkpoint_path, device)


@torch.no_grad()
def forecast_one_series(
    model: torch.nn.Module,
    model_name: str,
    history: np.ndarray,
    device: torch.device,
    series_idx: int | None = None,
) -> np.ndarray:
    x = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)

    if model_name == "latent_shared_v2":
        if series_idx is None:
            raise ValueError("series_idx is required for latent_shared_v2.")
        series_idx_tensor = torch.tensor([series_idx], dtype=torch.long).to(device)
        y_hat = model(x, series_idx_tensor)
    else:
        y_hat = model(x)

    return y_hat.squeeze(0).cpu().numpy()


@torch.no_grad()
def export_one_shot_forecasts(
    model: torch.nn.Module,
    model_name: str,
    history_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    split_name: str,
    device: torch.device,
    series_to_idx: dict[str, int],
) -> pd.DataFrame:
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
        target_dates = pd.to_datetime(eval_series["ds"]).reset_index(drop=True)

        history_window = history[-input_window:]
        target_window = target[:forecast_horizon]
        date_window = target_dates.iloc[:forecast_horizon].tolist()

        series_idx = series_to_idx[str(series_id)]

        pred_window = forecast_one_series(
            model=model,
            model_name=model_name,
            history=history_window,
            device=device,
            series_idx=series_idx,
        )

        for step_idx in range(forecast_horizon):
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "series_id": series_id,
                    "horizon_step": step_idx + 1,
                    "ds": date_window[step_idx],
                    "y_true": float(target_window[step_idx]),
                    "y_pred": float(pred_window[step_idx]),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    data_config = load_yaml(DATA_CONFIG_FILE)
    latent_config = load_yaml(LATENT_V2_CONFIG_FILE)
    nbeats_config = load_yaml(NBEATS_CONFIG_FILE)
    nhits_config = load_yaml(NHITS_CONFIG_FILE)

    input_window = int(data_config["input_window"])
    forecast_horizon = int(data_config["forecast_horizon"])
    val_size = int(data_config["val_size"])
    test_size = int(data_config["test_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_model_ready_dataset()
    train_df, val_df, test_df = temporal_split_by_series(
        df,
        val_size=val_size,
        test_size=test_size,
    )
    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)

    series_to_idx = build_series_index_mapping(df)
    num_series = len(series_to_idx)

    latent_model = build_latent_v2_model(data_config, latent_config, num_series, device)
    nbeats_model = build_nbeats_model(data_config, nbeats_config, device)
    nhits_model = build_nhits_model(data_config, nhits_config, device)

    model_specs = [
        ("latent_shared_v2", latent_model),
        ("nbeats", nbeats_model),
        ("nhits", nhits_model),
    ]

    all_frames = []

    for model_name, model in model_specs:
        print(f"\nExporting forecasts for: {model_name}")

        val_forecasts = export_one_shot_forecasts(
            model=model,
            model_name=model_name,
            history_df=train_df,
            eval_df=val_df,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            split_name="val",
            device=device,
            series_to_idx=series_to_idx,
        )

        test_forecasts = export_one_shot_forecasts(
            model=model,
            model_name=model_name,
            history_df=train_plus_val_df,
            eval_df=test_df,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            split_name="test",
            device=device,
            series_to_idx=series_to_idx,
        )

        model_forecasts = pd.concat([val_forecasts, test_forecasts], ignore_index=True)
        model_output_path = FORECASTS_DIR / f"{model_name}_forecasts_final.csv"

        FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        model_forecasts.to_csv(model_output_path, index=False)

        print(f"Saved: {model_output_path}")
        print(f"Rows: {len(model_forecasts)}")

        all_frames.append(model_forecasts)

    all_forecasts = pd.concat(all_frames, ignore_index=True)
    all_output_path = FORECASTS_DIR / "all_model_forecasts_final.csv"
    all_forecasts.to_csv(all_output_path, index=False)

    print(f"\nSaved combined forecasts: {all_output_path}")
    print("\n=== FIRST 18 ROWS OF FINAL COMBINED FORECASTS ===")
    print(all_forecasts.head(18))


if __name__ == "__main__":
    main()