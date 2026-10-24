"""From-scratch CNN model definitions.

Design notes (for the report's "model 1" justification):

* Four convolutional blocks with a doubling channel width (32 -> 64 -> 128 ->
  256) progressively trade spatial resolution for semantic depth -- a standard,
  well-understood pattern for small RGB tiles.
* ``3x3`` convolutions with ``padding=1`` keep spatial size within a block; max
  pooling halves it between the first three blocks.
* The fourth block ends in an adaptive average pool to ``1x1`` so the classifier
  is independent of the exact input resolution.
* Dropout (0.3) before the linear layer regularizes the classifier.
* The model outputs **raw logits** -- no Softmax. Softmax is applied implicitly
  by :class:`torch.nn.CrossEntropyLoss` during training.
"""

from __future__ import annotations

import torch
from torch import nn


CHANNELS = (32, 64, 128, 256)
SUPPORTED_MODELS = ("scratch_cnn", "depthwise_separable_cnn")


class ScratchCNN(nn.Module):
    """A compact convolutional classifier trained from scratch."""

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32, then halve spatial size.
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128 -> 256, then collapse to 1x1.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)  # raw logits


class DepthwiseSeparableConv2d(nn.Module):
    """A 3x3 depthwise convolution followed by a 1x1 pointwise convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableCNN(nn.Module):
    """A compact scratch CNN using depthwise-separable convolution blocks."""

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        c1, c2, c3, c4 = CHANNELS
        self.features = nn.Sequential(
            # Block 1: 3 -> 32, then halve spatial size.
            DepthwiseSeparableConv2d(3, c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64.
            DepthwiseSeparableConv2d(c1, c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128.
            DepthwiseSeparableConv2d(c2, c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4: 128 -> 256, then collapse to 1x1.
            DepthwiseSeparableConv2d(c3, c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)  # raw logits


def build_model(config: dict, num_classes: int) -> nn.Module:
    """Build a model from the ``model`` section of a resolved config."""
    name = config.get("name", "scratch_cnn")
    dropout = float(config.get("dropout", 0.3))

    if name == "scratch_cnn":
        return ScratchCNN(num_classes=num_classes, dropout=dropout)
    if name == "depthwise_separable_cnn":
        return DepthwiseSeparableCNN(num_classes=num_classes, dropout=dropout)

    supported = ", ".join(SUPPORTED_MODELS)
    raise ValueError(f"Unknown model.name {name!r}. Supported values: {supported}.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in ``model``."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
