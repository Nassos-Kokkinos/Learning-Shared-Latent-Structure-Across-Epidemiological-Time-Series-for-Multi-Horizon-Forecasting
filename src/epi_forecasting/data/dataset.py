from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def validate_model_ready_dataframe(df: pd.DataFrame) -> None:
    """Validate the model-ready forecasting dataframe."""
    expected_columns = ["ds", "series_id", "y"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    if df.empty:
        raise ValueError("The dataframe is empty.")

    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        raise TypeError("Column 'ds' must be datetime.")

    if df["series_id"].isna().any():
        raise ValueError("Column 'series_id' contains missing values.")

    if df["y"].isna().any():
        raise ValueError("Column 'y' contains missing values.")

    duplicated_rows = df.duplicated(subset=["series_id", "ds"]).sum()
    if duplicated_rows > 0:
        raise ValueError(
            f"Found {duplicated_rows} duplicated (series_id, ds) rows."
        )

    for series_id, group in df.groupby("series_id"):
        if not group["ds"].is_monotonic_increasing:
            raise ValueError(
                f"Dates are not sorted for series_id='{series_id}'."
            )


def summarize_series_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Return the number of observations for each time series."""
    summary = (
        df.groupby("series_id")
        .size()
        .reset_index(name="n_obs")
        .sort_values("n_obs", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def get_series_date_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Return start date, end date, and length for each series."""
    summary = (
        df.groupby("series_id")
        .agg(
            start_date=("ds", "min"),
            end_date=("ds", "max"),
            n_obs=("ds", "size"),
        )
        .reset_index()
        .sort_values(["n_obs", "series_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


@dataclass
class WindowSample:
    series_id: str
    x: np.ndarray
    y: np.ndarray
    start_idx: int


def build_window_samples(
    df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
) -> list[WindowSample]:
    """Build sliding-window samples from a model-ready dataframe."""
    validate_model_ready_dataframe(df)

    samples: list[WindowSample] = []

    for series_id, group in df.groupby("series_id", sort=False):
        group = group.sort_values("ds").reset_index(drop=True)
        y_values = group["y"].to_numpy(dtype=np.float32)

        n_obs = len(y_values)
        n_windows = n_obs - input_window - forecast_horizon + 1

        if n_windows <= 0:
            continue

        for start_idx in range(n_windows):
            x = y_values[start_idx : start_idx + input_window]
            y = y_values[
                start_idx + input_window : start_idx + input_window + forecast_horizon
            ]

            samples.append(
                WindowSample(
                    series_id=series_id,
                    x=x.copy(),
                    y=y.copy(),
                    start_idx=start_idx,
                )
            )

    return samples


class MultiSeriesWindowDataset(Dataset):
    """PyTorch dataset for multi-series forecasting windows."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_window: int,
        forecast_horizon: int,
    ) -> None:
        self.df = df.copy()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon
        self.samples = build_window_samples(
            df=self.df,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
        )

        if len(self.samples) == 0:
            raise ValueError(
                "No window samples were created. "
                "Check input_window / forecast_horizon / dataset length."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        x_tensor = torch.tensor(sample.x, dtype=torch.float32)
        y_tensor = torch.tensor(sample.y, dtype=torch.float32)

        return {
            "series_id": sample.series_id,
            "x": x_tensor,
            "y": y_tensor,
            "start_idx": sample.start_idx,
        }