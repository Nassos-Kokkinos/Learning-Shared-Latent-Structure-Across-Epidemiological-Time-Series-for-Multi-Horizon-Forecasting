from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.utils.paths import RESULTS_DIR


FORECASTS_FILE = RESULTS_DIR / "forecasts" / "all_model_forecasts.csv"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

PAPER_MODELS = ["nbeats", "nhits", "latent_shared"]
PAPER_LABELS = {
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "latent_shared": "Proposed Model",
}


def compute_horizon_metrics(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = forecasts_df.groupby(["model", "split", "horizon_step"], sort=True)

    for (model, split, horizon_step), group in grouped:
        errors = group["y_true"] - group["y_pred"]
        mae = errors.abs().mean()
        rmse = (errors.pow(2).mean()) ** 0.5

        rows.append(
            {
                "model": model,
                "split": split,
                "horizon_step": int(horizon_step),
                "mae": float(mae),
                "rmse": float(rmse),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["split", "horizon_step", "model"]
    ).reset_index(drop=True)

    return out


def make_paper_horizon_table(horizon_df: pd.DataFrame) -> pd.DataFrame:
    test_df = horizon_df.loc[
        (horizon_df["split"] == "test") & (horizon_df["model"].isin(PAPER_MODELS))
    ].copy()

    blocks = []

    for model_key in PAPER_MODELS:
        label = PAPER_LABELS[model_key]
        sub = (
            test_df.loc[test_df["model"] == model_key, ["horizon_step", "mae", "rmse"]]
            .copy()
            .rename(
                columns={
                    "mae": f"{label}_MAE",
                    "rmse": f"{label}_RMSE",
                }
            )
        )
        blocks.append(sub)

    merged = blocks[0]
    for sub in blocks[1:]:
        merged = merged.merge(sub, on="horizon_step", how="inner")

    merged = merged.sort_values("horizon_step").reset_index(drop=True)
    return merged


def plot_horizon_metric(
    horizon_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    plot_df = horizon_df.loc[
        (horizon_df["split"] == "test") & (horizon_df["model"].isin(PAPER_MODELS))
    ].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for model_key in PAPER_MODELS:
        sub = (
            plot_df.loc[plot_df["model"] == model_key]
            .sort_values("horizon_step")
            .reset_index(drop=True)
        )

        ax.plot(
            sub["horizon_step"],
            sub[metric],
            marker="o",
            linewidth=2,
            label=PAPER_LABELS[model_key],
        )

    ax.set_title(title)
    ax.set_xlabel("Horizon step")
    ax.set_ylabel(metric.upper())
    ax.set_xticks(sorted(plot_df["horizon_step"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not FORECASTS_FILE.exists():
        raise FileNotFoundError(f"Forecast file not found: {FORECASTS_FILE}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    forecasts_df = pd.read_csv(FORECASTS_FILE)

    expected_columns = {
        "model",
        "split",
        "series_id",
        "horizon_step",
        "ds",
        "y_true",
        "y_pred",
    }
    missing_columns = expected_columns - set(forecasts_df.columns)
    if missing_columns:
        raise ValueError(
            f"Forecast file is missing columns: {sorted(missing_columns)}"
        )

    horizon_df = compute_horizon_metrics(forecasts_df)
    paper_horizon_df = make_paper_horizon_table(horizon_df)

    long_table_path = TABLES_DIR / "table_horizon_results_long.csv"
    paper_table_path = TABLES_DIR / "table_horizon_results_paper.csv"
    fig_mae_path = FIGURES_DIR / "fig_horizon_mae.png"
    fig_rmse_path = FIGURES_DIR / "fig_horizon_rmse.png"

    horizon_df.to_csv(long_table_path, index=False)
    paper_horizon_df.to_csv(paper_table_path, index=False)

    plot_horizon_metric(
        horizon_df=horizon_df,
        metric="mae",
        output_path=fig_mae_path,
        title="Test MAE by Forecast Horizon",
    )

    plot_horizon_metric(
        horizon_df=horizon_df,
        metric="rmse",
        output_path=fig_rmse_path,
        title="Test RMSE by Forecast Horizon",
    )

    print("Saved:")
    print(long_table_path)
    print(paper_table_path)
    print(fig_mae_path)
    print(fig_rmse_path)

    print("\n=== HORIZON RESULTS (LONG) ===")
    print(horizon_df)

    print("\n=== HORIZON RESULTS (PAPER, TEST ONLY) ===")
    print(paper_horizon_df)


if __name__ == "__main__":
    main()
