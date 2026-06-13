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
    "last_value_baseline": RESULTS_DIR / "metrics" / "last_value_baseline_metrics.csv",
    "latent_shared": RESULTS_DIR / "metrics" / "latent_shared_eval_metrics.csv",
    "nbeats": RESULTS_DIR / "metrics" / "nbeats_eval_metrics.csv",
    "nhits": RESULTS_DIR / "metrics" / "nhits_eval_metrics.csv",
}

DISPLAY_NAMES = {
    "last_value_baseline": "last_value_baseline",
    "latent_shared": "latent_shared",
    "nbeats": "nbeats",
    "nhits": "nhits",
}

PAPER_DISPLAY_NAMES = {
    "latent_shared": "Proposed Model",
    "nbeats": "N-BEATS",
    "nhits": "N-HiTS",
}

MODEL_ORDER = [
    "last_value_baseline",
    "latent_shared",
    "nbeats",
    "nhits",
]

PAPER_MODEL_ORDER = [
    "nbeats",
    "nhits",
    "latent_shared",
]

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_all_metrics() -> pd.DataFrame:
    frames = []

    for model_key, path in METRIC_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

        df = pd.read_csv(path)

        expected_columns = {"split", "series_id", "mae", "rmse"}
        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Metrics file {path} is missing columns: {sorted(missing_columns)}"
            )

        df = df.copy()
        df["model"] = DISPLAY_NAMES[model_key]
        frames.append(df[["model", "split", "series_id", "mae", "rmse"]])

    combined = pd.concat(frames, ignore_index=True)

    combined["model"] = pd.Categorical(
        combined["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    combined = combined.sort_values(["model", "split", "series_id"]).reset_index(drop=True)
    return combined


def make_main_results_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        metrics_df.groupby(["model", "split"])[["mae", "rmse"]]
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(
        index="model",
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

    pivot.columns = [
        "val_mae",
        "val_rmse",
        "test_mae",
        "test_rmse",
    ]

    pivot = pivot.reset_index()
    pivot["model"] = pd.Categorical(
        pivot["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    pivot = pivot.sort_values("model").reset_index(drop=True)
    pivot["model"] = pivot["model"].astype(str)

    return pivot


def make_per_series_long_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    table = metrics_df.copy()
    table["model"] = table["model"].astype(str)
    table = table[["model", "split", "series_id", "mae", "rmse"]]
    table = table.sort_values(["model", "split", "series_id"]).reset_index(drop=True)
    return table


def make_paper_main_table(main_df: pd.DataFrame) -> pd.DataFrame:
    paper_df = main_df.loc[
        main_df["model"].isin(PAPER_MODEL_ORDER)
    ].copy()

    paper_df["model"] = pd.Categorical(
        paper_df["model"],
        categories=PAPER_MODEL_ORDER,
        ordered=True,
    )
    paper_df = paper_df.sort_values("model").reset_index(drop=True)

    paper_df["model"] = paper_df["model"].astype(str).map(PAPER_DISPLAY_NAMES)
    return paper_df


def make_paper_per_series_test_table(per_series_df: pd.DataFrame) -> pd.DataFrame:
    paper_df = per_series_df.loc[
        (per_series_df["split"] == "test")
        & (per_series_df["model"].isin(PAPER_MODEL_ORDER))
    ].copy()

    metric_frames = []

    for model_key in PAPER_MODEL_ORDER:
        model_name = PAPER_DISPLAY_NAMES[model_key]
        sub = (
            paper_df.loc[paper_df["model"] == model_key, ["series_id", "mae", "rmse"]]
            .copy()
            .rename(
                columns={
                    "mae": f"{model_name}_MAE",
                    "rmse": f"{model_name}_RMSE",
                }
            )
        )
        metric_frames.append(sub)

    merged = metric_frames[0]
    for sub in metric_frames[1:]:
        merged = merged.merge(sub, on="series_id", how="inner")

    merged = merged.sort_values("series_id").reset_index(drop=True)
    return merged


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_test_metric(
    main_df: pd.DataFrame,
    metric_column: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = main_df[["model", metric_column]].copy()
    plot_df = plot_df.sort_values(metric_column).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(plot_df["model"], plot_df[metric_column])

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_column.upper())
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, plot_df[metric_column]):
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
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = load_all_metrics()

    main_table = make_main_results_table(metrics_df)
    per_series_table = make_per_series_long_table(metrics_df)

    paper_main_table = make_paper_main_table(main_table)
    paper_per_series_test_table = make_paper_per_series_test_table(per_series_table)

    main_table_path = TABLES_DIR / "table_main_results.csv"
    per_series_table_path = TABLES_DIR / "table_per_series_results.csv"
    paper_main_table_path = TABLES_DIR / "table_main_results_paper.csv"
    paper_per_series_test_table_path = TABLES_DIR / "table_per_series_results_paper_test.csv"

    fig_mae_path = FIGURES_DIR / "fig_model_comparison_test_mae.png"
    fig_rmse_path = FIGURES_DIR / "fig_model_comparison_test_rmse.png"

    save_table(main_table, main_table_path)
    save_table(per_series_table, per_series_table_path)
    save_table(paper_main_table, paper_main_table_path)
    save_table(paper_per_series_test_table, paper_per_series_test_table_path)

    plot_test_metric(
        main_df=main_table,
        metric_column="test_mae",
        title="Overall Model Comparison - Test MAE",
        output_path=fig_mae_path,
    )

    plot_test_metric(
        main_df=main_table,
        metric_column="test_rmse",
        title="Overall Model Comparison - Test RMSE",
        output_path=fig_rmse_path,
    )

    print("Saved:")
    print(main_table_path)
    print(per_series_table_path)
    print(paper_main_table_path)
    print(paper_per_series_test_table_path)
    print(fig_mae_path)
    print(fig_rmse_path)

    print("\n=== PAPER MAIN TABLE ===")
    print(paper_main_table)

    print("\n=== PAPER PER-SERIES TEST TABLE ===")
    print(paper_per_series_test_table)


if __name__ == "__main__":
    main()