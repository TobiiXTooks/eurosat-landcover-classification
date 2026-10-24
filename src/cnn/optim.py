"""Optimizer construction from config."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    config: dict,
) -> torch.optim.Optimizer:
    """Build the configured optimizer.

    Supported optimizers are deliberately small because the experiment plan only
    needs Adam and SGD+momentum for the course comparison.
    """
    name = str(config.get("name", "adam")).lower()
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {name!r} (expected 'adam' or 'sgd')")
