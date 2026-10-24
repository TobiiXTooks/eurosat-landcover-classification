"""Training and evaluation loops shared by the CLIs."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress: bool = True,
) -> tuple[float, float]:
    """Run one training epoch. Returns (average loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    iterator = loader
    if progress:
        iterator = tqdm(loader, desc="train", leave=False, unit="batch")

    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress: bool = True,
    desc: str = "eval",
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model. Returns (average loss, predictions, true labels)."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    iterator = loader
    if progress:
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch")

    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = running_loss / total if total else 0.0
    return avg_loss, np.asarray(all_preds), np.asarray(all_targets)
