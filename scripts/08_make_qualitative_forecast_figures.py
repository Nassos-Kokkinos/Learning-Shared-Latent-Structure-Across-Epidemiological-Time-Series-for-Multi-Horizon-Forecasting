from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.utils.paths import CONFIGS_DIR, RESULTS_DIR


DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"
FORECASTS_FILE = RESULTS_DIR / "forecasts" / "all_model_forecasts.csv"
FIGURES_DIR = RESULTS_DIR / "figures"

FOCUS_SERIES = [
    "Adenovirus",
    "HMPV",
    "RV/EV",
]

MODEL_ORDER = ["nbeats", "nhits", "latent_shared"]
MODEL_LABELS = {
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "latent_shared": "Proposed Model",
}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def get_test_history_and_truth(
    full_df: pd.DataFrame,
    series_id: str,
    input_window: int,
    test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_df = (
        full_df.loc[full_df["series_id"] == series_id]
        .sort_values("ds")
        .reset_index(drop=True)
        .copy()
    )

    test_start = len(series_df) - test_size

    history_df = series_df.iloc[test_start - input_window:test_start].copy()
    truth_df = series_df.iloc[test_start:test_start + 4].copy()

    return history_df, truth_df


def make_forecast_plot(
    series_id: str,
    history_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))

    history_df = history_df.copy()
    truth_df = truth_df.copy()
    forecast_df = forecast_df.copy()

    history_df["ds"] = pd.to_datetime(history_df["ds"])
    truth_df["ds"] = pd.to_datetime(truth_df["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    # Past history
    ax.plot(
        history_df["ds"],
        history_df["y"],
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Past history",
    )

    # True future
    ax.plot(
        truth_df["ds"],
        truth_df["y"],
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=4,
        label="Observed future",
    )

    # Forecast origin
    forecast_origin = truth_df["ds"].iloc[0]
    ax.axvline(
        forecast_origin,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )

    # Model forecasts
    for model_name in MODEL_ORDER:
        model_sub = (
            forecast_df.loc[forecast_df["model"] == model_name]
            .sort_values("horizon_step")
            .reset_index(drop=True)
        )

        ax.plot(
            model_sub["ds"],
            model_sub["y_pred"],
            linewidth=2,
            marker="o",
            markersize=4,
            label=MODEL_LABELS[model_name],
        )

    ax.set_title(f"Test Forecast Comparison - {series_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent positivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not FORECASTS_FILE.exists():
        raise FileNotFoundError(f"Forecast file not found: {FORECASTS_FILE}")

    data_config = load_yaml(DATA_CONFIG_FILE)
    input_window = int(data_config["input_window"])
    test_size = int(data_config["test_size"])
    forecast_horizon = int(data_config["forecast_horizon"])

    full_df = load_model_ready_dataset()
    forecasts_df = pd.read_csv(FORECASTS_FILE, parse_dates=["ds"])

    forecasts_df = forecasts_df.loc[forecasts_df["split"] == "test"].copy()

    print("Creating qualitative test forecast figures for:")
    for series_id in FOCUS_SERIES:
        print(f"- {series_id}")

    for series_id in FOCUS_SERIES:
        history_df, truth_df = get_test_history_and_truth(
            full_df=full_df,
            series_id=series_id,
            input_window=input_window,
            test_size=test_size,
        )

        forecast_sub = (
            forecasts_df.loc[forecasts_df["series_id"] == series_id]
            .sort_values(["model", "horizon_step"])
            .copy()
        )

        expected_models = set(MODEL_ORDER)
        found_models = set(forecast_sub["model"].unique())
        missing_models = expected_models - found_models
        if missing_models:
            raise ValueError(
                f"Missing forecasts for series '{series_id}' and models: {sorted(missing_models)}"
            )

        for model_name in MODEL_ORDER:
            model_rows = forecast_sub.loc[forecast_sub["model"] == model_name]
            if len(model_rows) != forecast_horizon:
                raise ValueError(
                    f"Series '{series_id}', model '{model_name}' has {len(model_rows)} "
                    f"rows instead of forecast_horizon={forecast_horizon}."
                )

        output_path = FIGURES_DIR / f"fig_forecast_test_{safe_filename(series_id)}.png"

        make_forecast_plot(
            series_id=series_id,
            history_df=history_df,
            truth_df=truth_df,
            forecast_df=forecast_sub,
            output_path=output_path,
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
