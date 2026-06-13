import pandas as pd

from epi_forecasting.data.dataset import validate_model_ready_dataframe


def temporal_split_by_series(
    df: pd.DataFrame,
    val_size: int = 24,
    test_size: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each series into train/validation/test by time order."""
    validate_model_ready_dataframe(df)

    train_parts = []
    val_parts = []
    test_parts = []

    for series_id, group in df.groupby("series_id", sort=False):
        group = group.sort_values("ds").reset_index(drop=True)
        n_obs = len(group)

        if n_obs <= val_size + test_size:
            raise ValueError(
                f"Series '{series_id}' is too short: {n_obs} observations, "
                f"but val_size + test_size = {val_size + test_size}."
            )

        train_end = n_obs - val_size - test_size
        val_end = n_obs - test_size

        train_parts.append(group.iloc[:train_end].copy())
        val_parts.append(group.iloc[train_end:val_end].copy())
        test_parts.append(group.iloc[val_end:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


def summarize_split_lengths(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize train/val/test lengths for each series."""
    train_counts = (
        train_df.groupby("series_id")
        .size()
        .reset_index(name="train_n")
    )
    val_counts = (
        val_df.groupby("series_id")
        .size()
        .reset_index(name="val_n")
    )
    test_counts = (
        test_df.groupby("series_id")
        .size()
        .reset_index(name="test_n")
    )

    summary = train_counts.merge(val_counts, on="series_id").merge(test_counts, on="series_id")
    summary["total_n"] = summary["train_n"] + summary["val_n"] + summary["test_n"]

    return summary.sort_values("series_id").reset_index(drop=True)