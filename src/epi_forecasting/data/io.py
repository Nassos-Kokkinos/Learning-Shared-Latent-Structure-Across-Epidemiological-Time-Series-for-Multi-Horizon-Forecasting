from pathlib import Path

import pandas as pd

from epi_forecasting.utils.paths import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_dataset(filename: str = "cleaned_epi_dataset.csv") -> pd.DataFrame:
    """Load the original cleaned dataset from data/processed."""
    file_path = PROCESSED_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {file_path}")

    df = pd.read_csv(file_path, sep=";")
    return df


def load_prepared_dataset(filename: str = "prepared_epi_dataset.csv") -> pd.DataFrame:
    """Load the prepared dataset from data/interim."""
    file_path = INTERIM_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["ds"])
    return df


def load_model_ready_dataset(filename: str = "model_ready_dataset.csv") -> pd.DataFrame:
    """Load the model-ready forecasting dataset from data/interim."""
    file_path = INTERIM_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Model-ready dataset not found: {file_path}")

    df = pd.read_csv(file_path, parse_dates=["ds"])

    expected_columns = ["ds", "series_id", "y"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df = df.sort_values(["series_id", "ds"]).reset_index(drop=True)
    return df