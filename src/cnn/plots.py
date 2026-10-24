"""Plotting helpers for run artifacts.

All figures are written to disk; nothing is shown interactively (Agg backend),
so this works on headless GPU/Colab runtimes.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .data import EuroSATDataset  # noqa: E402


def plot_loss_curve(history: dict, path: Path) -> None:
    """Training vs validation loss per epoch."""
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="train")
    ax.plot(epochs, history["val_loss"], marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_accuracy_curve(history: dict, path: Path) -> None:
    """Training vs validation accuracy per epoch."""
    epochs = history["epoch"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_acc"], marker="o", label="train")
    ax.plot(epochs, history["val_acc"], marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Training and validation accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(
    cm: list[list[int]], class_order: list[str], path: Path
) -> None:
    """Confusion matrix heatmap with counts annotated."""
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(class_order)), max(5, len(class_order))))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_order)))
    ax.set_yticks(range(len(class_order)))
    ax.set_xticklabels(class_order, rotation=45, ha="right")
    ax.set_yticklabels(class_order)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")

    threshold = cm_arr.max() / 2.0 if cm_arr.max() else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                int(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_class_distribution(
    splits: dict[str, list[tuple[str, int]]],
    multiclass_order: list[str],
    class_order: list[str],
    task: str,
    path: Path,
) -> None:
    """Grouped bar chart of class counts across train/val/test for the task."""
    from . import labels

    def label_of(multiclass_idx: int) -> int:
        if task == "multiclass":
            return multiclass_idx
        return labels.to_binary_index(multiclass_order[multiclass_idx])

    split_names = ["train", "val", "test"]
    counts = {
        name: Counter(label_of(idx) for _, idx in splits[name]) for name in split_names
    }

    x = np.arange(len(class_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(class_order)), 5))
    for offset, name in zip((-width, 0, width), split_names):
        heights = [counts[name].get(i, 0) for i in range(len(class_order))]
        ax.bar(x + offset, heights, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title(f"Class distribution per split ({task})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _denormalize(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """Undo Normalize and return an HxWxC array clipped to [0, 1]."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    img = image * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def plot_misclassified_grid(
    dataset: EuroSATDataset,
    preds: np.ndarray,
    targets: np.ndarray,
    class_order: list[str],
    mean: list[float],
    std: list[float],
    path: Path,
    max_examples: int = 16,
) -> None:
    """Grid of misclassified test images with predicted/true labels."""
    wrong = np.where(preds != targets)[0]
    if len(wrong) == 0:
        # Still emit a placeholder so the run directory is complete.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No misclassified examples", ha="center", va="center")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return

    wrong = wrong[:max_examples]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, sample_idx in enumerate(wrong):
        image, _ = dataset[int(sample_idx)]
        axes[ax_idx].imshow(_denormalize(image, mean, std))
        axes[ax_idx].set_title(
            f"pred: {class_order[preds[sample_idx]]}\ntrue: {class_order[targets[sample_idx]]}",
            fontsize=8,
        )
        axes[ax_idx].axis("off")

    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.suptitle("Misclassified test examples")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
