from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()

    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("No samples were seen during training.")

    return running_loss / n_samples


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on a dataloader and return average loss."""
    model.eval()

    running_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        batch_size = x.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    if n_samples == 0:
        raise ValueError("No samples were seen during evaluation.")

    return running_loss / n_samples


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Run batched prediction and return stacked tensors."""
    model.eval()

    all_preds = []
    all_targets = []
    all_series_ids = []
    all_start_idxs = []

    for batch in dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x)

        all_preds.append(y_hat.cpu())
        all_targets.append(y.cpu())
        all_series_ids.extend(batch["series_id"])
        all_start_idxs.extend(batch["start_idx"].tolist())

    return {
        "predictions": torch.cat(all_preds, dim=0),
        "targets": torch.cat(all_targets, dim=0),
        "series_id": all_series_ids,
        "start_idx": all_start_idxs,
    }