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


def build_series_index_mapping(df: pd.DataFrame) -> dict[str, int]:
    """Create a stable mapping from series_id to integer index."""
    series_ids = sorted(df["series_id"].astype(str).unique().tolist())
    return {series_id: idx for idx, series_id in enumerate(series_ids)}


@dataclass
class WindowSampleV2:
    series_id: str
    series_idx: int
    x: np.ndarray
    y: np.ndarray
    start_idx: int


def build_window_samples_v2(
    df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    series_to_idx: dict[str, int] | None = None,
) -> list[WindowSampleV2]:
    """Build sliding-window samples with series indices."""
    validate_model_ready_dataframe(df)

    if series_to_idx is None:
        series_to_idx = build_series_index_mapping(df)

    samples: list[WindowSampleV2] = []

    for series_id, group in df.groupby("series_id", sort=False):
        series_id = str(series_id)
        group = group.sort_values("ds").reset_index(drop=True)
        y_values = group["y"].to_numpy(dtype=np.float32)

        n_obs = len(y_values)
        n_windows = n_obs - input_window - forecast_horizon + 1

        if n_windows <= 0:
            continue

        series_idx = series_to_idx[series_id]

        for start_idx in range(n_windows):
            x = y_values[start_idx : start_idx + input_window]
            y = y_values[
                start_idx + input_window : start_idx + input_window + forecast_horizon
            ]

            samples.append(
                WindowSampleV2(
                    series_id=series_id,
                    series_idx=series_idx,
                    x=x.copy(),
                    y=y.copy(),
                    start_idx=start_idx,
                )
            )

    return samples


class MultiSeriesWindowDatasetV2(Dataset):
    """PyTorch dataset for multi-series forecasting windows with series indices."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_window: int,
        forecast_horizon: int,
    ) -> None:
        self.df = df.copy()
        self.input_window = input_window
        self.forecast_horizon = forecast_horizon

        self.series_to_idx = build_series_index_mapping(self.df)
        self.idx_to_series = {
            idx: series_id for series_id, idx in self.series_to_idx.items()
        }
        self.num_series = len(self.series_to_idx)

        self.samples = build_window_samples_v2(
            df=self.df,
            input_window=input_window,
            forecast_horizon=forecast_horizon,
            series_to_idx=self.series_to_idx,
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
        series_idx_tensor = torch.tensor(sample.series_idx, dtype=torch.long)

        return {
            "series_id": sample.series_id,
            "series_idx": series_idx_tensor,
            "x": x_tensor,
            "y": y_tensor,
            "start_idx": sample.start_idx,
        }
