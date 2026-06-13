import torch
from torch.utils.data import DataLoader

from epi_forecasting.data.dataset import MultiSeriesWindowDataset


def create_train_dataloader(
    dataset: MultiSeriesWindowDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader for training."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )