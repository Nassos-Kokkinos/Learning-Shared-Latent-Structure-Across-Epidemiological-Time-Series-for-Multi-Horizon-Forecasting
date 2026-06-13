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

FOCUS_SERIES = ["Adenovirus", "HMPV", "RV/EV"]

MODEL_ORDER = ["nbeats", "nhits", "latent_shared"]
MODEL_LABELS = {
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "latent_shared": "Proposed Model",
}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_test_history_and_truth(
    full_df: pd.DataFrame,
    series_id: str,
    input_window: int,
    test_size: int,
    forecast_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    series_df = (
        full_df.loc[full_df["series_id"] == series_id]
        .sort_values("ds")
        .reset_index(drop=True)
        .copy()
    )

    test_start = len(series_df) - test_size
    history_df = series_df.iloc[test_start - input_window:test_start].copy()
    truth_df = series_df.iloc[test_start:test_start + forecast_horizon].copy()

    return history_df, truth_df


def plot_one_panel(
    ax,
    series_id: str,
    history_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> None:
    history_df = history_df.copy()
    truth_df = truth_df.copy()
    forecast_df = forecast_df.copy()

    history_df["ds"] = pd.to_datetime(history_df["ds"])
    truth_df["ds"] = pd.to_datetime(truth_df["ds"])
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    ax.plot(
        history_df["ds"],
        history_df["y"],
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Past history",
    )

    ax.plot(
        truth_df["ds"],
        truth_df["y"],
        linewidth=2.0,
        linestyle="--",
        marker="o",
        markersize=3.5,
        label="Observed future",
    )

    forecast_origin = truth_df["ds"].iloc[0]
    ax.axvline(
        forecast_origin,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    for model_name in MODEL_ORDER:
        model_sub = (
            forecast_df.loc[forecast_df["model"] == model_name]
            .sort_values("horizon_step")
            .reset_index(drop=True)
        )

        ax.plot(
            model_sub["ds"],
            model_sub["y_pred"],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=MODEL_LABELS[model_name],
        )

    ax.set_title(series_id)
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent positivity")
    ax.grid(alpha=0.25)


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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=False)

    for ax, series_id in zip(axes, FOCUS_SERIES):
        history_df, truth_df = get_test_history_and_truth(
            full_df=full_df,
            series_id=series_id,
            input_window=input_window,
            test_size=test_size,
            forecast_horizon=forecast_horizon,
        )

        forecast_sub = (
            forecasts_df.loc[forecasts_df["series_id"] == series_id]
            .sort_values(["model", "horizon_step"])
            .copy()
        )

        plot_one_panel(
            ax=ax,
            series_id=series_id,
            history_df=history_df,
            truth_df=truth_df,
            forecast_df=forecast_sub,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Qualitative Comparison of Test Forecasts", fontsize=16)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    output_path = FIGURES_DIR / "fig_forecast_comparison_paper.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
