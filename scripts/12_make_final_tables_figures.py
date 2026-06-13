from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.utils.paths import RESULTS_DIR


METRIC_FILES = {
    "nbeats": RESULTS_DIR / "metrics" / "nbeats_eval_metrics.csv",
    "nhits": RESULTS_DIR / "metrics" / "nhits_eval_metrics.csv",
    "latent_shared_v2": RESULTS_DIR / "metrics" / "latent_shared_v2_eval_metrics.csv",
}

DISPLAY_NAMES = {
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
    "latent_shared_v2": "Proposed Model",
}

MODEL_ORDER = ["nbeats", "nhits", "latent_shared_v2"]

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_all_metrics() -> pd.DataFrame:
    frames = []

    for model_key, path in METRIC_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        df = pd.read_csv(path)
        expected_columns = {"split", "series_id", "mae", "rmse"}
        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(f"Metrics file {path} is missing columns: {sorted(missing)}")

        df = df.copy()
        df["model_key"] = model_key
        df["model"] = DISPLAY_NAMES[model_key]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def make_main_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["model_key", "model", "split"])[["mae", "rmse"]]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(
        index=["model_key", "model"],
        columns="split",
        values=["mae", "rmse"],
    )

    expected_cols = [
        ("mae", "val"),
        ("rmse", "val"),
        ("mae", "test"),
        ("rmse", "test"),
    ]
    for col in expected_cols:
        if col not in pivot.columns:
            raise ValueError(f"Missing expected summary column: {col}")

    pivot = pivot[expected_cols]
    pivot.columns = ["val_mae", "val_rmse", "test_mae", "test_rmse"]
    pivot = pivot.reset_index()

    order_map = {key: i for i, key in enumerate(MODEL_ORDER)}
    pivot["order"] = pivot["model_key"].map(order_map)
    pivot = pivot.sort_values("order").drop(columns=["order", "model_key"]).reset_index(drop=True)

    return pivot


def make_per_series_test_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    subset = metrics_df.loc[metrics_df["split"] == "test"].copy()

    pieces = []
    for model_key in MODEL_ORDER:
        model_name = DISPLAY_NAMES[model_key]
        sub = (
            subset.loc[subset["model_key"] == model_key, ["series_id", "mae", "rmse"]]
            .copy()
            .rename(
                columns={
                    "mae": f"{model_name}_MAE",
                    "rmse": f"{model_name}_RMSE",
                }
            )
        )
        pieces.append(sub)

    merged = pieces[0]
    for sub in pieces[1:]:
        merged = merged.merge(sub, on="series_id", how="inner")

    merged = merged.sort_values("series_id").reset_index(drop=True)
    return merged


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_test_metric(main_df: pd.DataFrame, metric_col: str, title: str, output_path: Path) -> None:
    plot_df = main_df[["model", metric_col]].copy().sort_values(metric_col).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plot_df["model"], plot_df[metric_col])

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_col.upper())
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, plot_df[metric_col]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics_df = load_all_metrics()

    main_table = make_main_table(metrics_df)
    per_series_test_table = make_per_series_test_table(metrics_df)

    main_path = TABLES_DIR / "final_table_main_results_paper.csv"
    per_series_path = TABLES_DIR / "final_table_per_series_results_paper_test.csv"
    fig_mae_path = FIGURES_DIR / "final_fig_model_comparison_test_mae.png"
    fig_rmse_path = FIGURES_DIR / "final_fig_model_comparison_test_rmse.png"

    save_table(main_table, main_path)
    save_table(per_series_test_table, per_series_path)

    plot_test_metric(
        main_df=main_table,
        metric_col="test_mae",
        title="Final Test MAE Comparison",
        output_path=fig_mae_path,
    )
    plot_test_metric(
        main_df=main_table,
        metric_col="test_rmse",
        title="Final Test RMSE Comparison",
        output_path=fig_rmse_path,
    )

    print("Saved:")
    print(main_path)
    print(per_series_path)
    print(fig_mae_path)
    print(fig_rmse_path)

    print("\n=== FINAL MAIN TABLE ===")
    print(main_table)

    print("\n=== FINAL PER-SERIES TEST TABLE ===")
    print(per_series_test_table)


if __name__ == "__main__":
    main()
