from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import (
    summarize_split_lengths,
    temporal_split_by_series,
)
from epi_forecasting.utils.paths import CONFIGS_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


RAW_INPUT_FILE = PROCESSED_DATA_DIR / "cleaned_epi_dataset.csv"
PREPARED_OUTPUT_FILE = INTERIM_DATA_DIR / "prepared_epi_dataset.csv"
MODEL_READY_OUTPUT_FILE = INTERIM_DATA_DIR / "model_ready_dataset.csv"

TRAIN_OUTPUT_FILE = INTERIM_DATA_DIR / "train_dataset.csv"
VAL_OUTPUT_FILE = INTERIM_DATA_DIR / "val_dataset.csv"
TEST_OUTPUT_FILE = INTERIM_DATA_DIR / "test_dataset.csv"

DATA_CONFIG_FILE = CONFIGS_DIR / "data.yaml"


def load_data_config() -> dict:
    if not DATA_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {DATA_CONFIG_FILE}")

    with open(DATA_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    if not RAW_INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {RAW_INPUT_FILE}")

    config = load_data_config()
    val_size = int(config["val_size"])
    test_size = int(config["test_size"])

    print(f"Reading raw file: {RAW_INPUT_FILE}")
    df = pd.read_csv(RAW_INPUT_FILE, sep=";")

    print("\nInitial shape:", df.shape)

    expected_columns = ["ds", "series_id", "tests", "detections", "y"]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    df["ds"] = pd.to_datetime(df["ds"], dayfirst=True, errors="coerce")
    df["series_id"] = df["series_id"].astype(str)
    df["tests"] = pd.to_numeric(df["tests"], errors="coerce")
    df["detections"] = pd.to_numeric(df["detections"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    print("\nMissing after type conversion:")
    print(df.isna().sum())

    df = df.drop_duplicates()
    df = df.sort_values(["series_id", "ds"]).reset_index(drop=True)

    model_df = df[["ds", "series_id", "y"]].copy()

    if model_df["ds"].isna().any():
        raise ValueError("Column 'ds' contains invalid dates after parsing.")

    if model_df["y"].isna().any():
        raise ValueError("Column 'y' contains missing values after conversion.")

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PREPARED_OUTPUT_FILE, index=False)
    model_df.to_csv(MODEL_READY_OUTPUT_FILE, index=False)

    print("\nSaved full prepared dataset to:")
    print(PREPARED_OUTPUT_FILE)

    print("\nSaved model-ready dataset to:")
    print(MODEL_READY_OUTPUT_FILE)

    model_df = load_model_ready_dataset()

    train_df, val_df, test_df = temporal_split_by_series(
        model_df,
        val_size=val_size,
        test_size=test_size,
    )

    train_df.to_csv(TRAIN_OUTPUT_FILE, index=False)
    val_df.to_csv(VAL_OUTPUT_FILE, index=False)
    test_df.to_csv(TEST_OUTPUT_FILE, index=False)

    print("\nSaved train split to:")
    print(TRAIN_OUTPUT_FILE)

    print("\nSaved validation split to:")
    print(VAL_OUTPUT_FILE)

    print("\nSaved test split to:")
    print(TEST_OUTPUT_FILE)

    print("\n=== SPLIT SUMMARY ===")
    print(summarize_split_lengths(train_df, val_df, test_df))

    print("\nFinal full shape:", df.shape)
    print("Final model-ready shape:", model_df.shape)
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)


if __name__ == "__main__":
    main()