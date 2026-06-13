from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epi_forecasting.data.dataset import (
    MultiSeriesWindowDataset,
    get_series_date_ranges,
    summarize_series_lengths,
    validate_model_ready_dataframe,
)
from epi_forecasting.data.io import load_model_ready_dataset
from epi_forecasting.data.preprocessing import temporal_split_by_series


INPUT_WINDOW = 52
FORECAST_HORIZON = 12
VAL_SIZE = 24
TEST_SIZE = 24


def main() -> None:
    df = load_model_ready_dataset()

    print("Loaded model-ready dataset successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    validate_model_ready_dataframe(df)
    print("\nValidation passed.")

    print("\n=== FIRST 5 ROWS ===")
    print(df.head())

    print("\n=== SERIES LENGTHS ===")
    print(summarize_series_lengths(df))

    print("\n=== SERIES DATE RANGES ===")
    print(get_series_date_ranges(df))

    train_df, val_df, test_df = temporal_split_by_series(
        df,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
    )

    print("\nTrain shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    train_dataset = MultiSeriesWindowDataset(
        df=train_df,
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    print("\n=== TRAIN WINDOW DATASET ===")
    print(f"Number of train windows: {len(train_dataset)}")

    first_sample = train_dataset[0]
    print("\nFirst sample keys:", list(first_sample.keys()))
    print("First sample series_id:", first_sample["series_id"])
    print("First sample x shape:", tuple(first_sample["x"].shape))
    print("First sample y shape:", tuple(first_sample["y"].shape))
    print("First sample start_idx:", first_sample["start_idx"])
    print("First sample x[:5]:", first_sample["x"][:5])
    print("First sample y[:5]:", first_sample["y"][:5])


if __name__ == "__main__":
    main()