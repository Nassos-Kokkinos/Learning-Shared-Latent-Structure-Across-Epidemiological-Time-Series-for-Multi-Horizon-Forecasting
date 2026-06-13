from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import temporal_split_by_series
from epi_forecasting.evaluation.metrics import mae, rmse
from epi_forecasting.utils.paths import CONFIGS_DIR, RESULTS_DIR


DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"
OUTPUT_FILE = RESULTS_DIR / "metrics" / "last_value_baseline_metrics.csv"


def load_data_config() -> dict:
    with open(DATA_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def last_value_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for the whole forecast horizon."""
    last_value = history[-1]
    return np.repeat(last_value, horizon).astype(np.float32)


def evaluate_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    split_name: str,
) -> list[dict]:
    """
    Evaluate a last-value baseline on a split.

    For each series:
    - history = last input_window observations before the eval period
    - target = first forecast_horizon observations inside the eval period
    """
    results = []

    for series_id in sorted(eval_df["series_id"].unique()):
        train_series = train_df.loc[train_df["series_id"] == series_id].sort_values("ds")
        eval_series = eval_df.loc[eval_df["series_id"] == series_id].sort_values("ds")

        history = train_series["y"].to_numpy(dtype=np.float32)
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
        pred_window = last_value_forecast(history_window, forecast_horizon)

        results.append(
            {
                "split": split_name,
                "series_id": series_id,
                "mae": mae(target_window, pred_window),
                "rmse": rmse(target_window, pred_window),
            }
        )

    return results


def main() -> None:
    config = load_data_config()

    input_window = int(config["input_window"])
    forecast_horizon = int(config["forecast_horizon"])
    val_size = int(config["val_size"])
    test_size = int(config["test_size"])

    df = load_model_ready_dataset()
    train_df, val_df, test_df = temporal_split_by_series(
        df,
        val_size=val_size,
        test_size=test_size,
    )

    val_results = evaluate_split(
        train_df=train_df,
        eval_df=val_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        split_name="val",
    )

    train_plus_val_df = pd.concat([train_df, val_df], ignore_index=True)

    test_results = evaluate_split(
        train_df=train_plus_val_df,
        eval_df=test_df,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
        split_name="test",
    )

    results_df = pd.DataFrame(val_results + test_results)

    summary_df = (
        results_df.groupby("split")[["mae", "rmse"]]
        .mean()
        .reset_index()
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved baseline metrics to:")
    print(OUTPUT_FILE)

    print("\n=== PER-SERIES RESULTS ===")
    print(results_df)

    print("\n=== AVERAGE RESULTS BY SPLIT ===")
    print(summary_df)


if __name__ == "__main__":
    main()