"""Train EfficientNet-B0 transfer-learning models for the EuroSAT project.

This script implements Model 2 for the project work:

* task="multiclass": original 10-class EuroSAT land-use classification
* task="binary": vegetation vs. non-vegetation classification
* task="all": run both tasks one after another

The underlying data loader uses the EuroSAT_MS GeoTIFF files, but only the
natural RGB bands are selected (B04, B03, B02). This is required because the
ImageNet-pretrained EfficientNet-B0 expects 3-channel RGB input.

Typical usage from the project root:

    python with_help/train_efficientnet_b0.py --task multiclass
    python with_help/train_efficientnet_b0.py --task binary
    python with_help/train_efficientnet_b0.py --task all --epochs 20

Outputs are written to:

    outputs/experiments/efficientnet_b0/<task>/

The script deliberately uses the same stratified split helper as Model 1 so the
final comparison remains fair.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from tqdm.auto import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp

# macOS leaks file descriptors with the default 'file_descriptor' sharing
# strategy when DataLoader workers are recreated each epoch. Over a long run
# (many epochs / multiple tasks) this exhausts the open-file limit and crashes
# with "Too many open files". 'file_system' avoids the per-tensor fd leak.
mp.set_sharing_strategy("file_system")
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from lib.constants import CLASSES, PATH_TO_DATASET, S2_REFLECTANCE_SCALE, SEED
from lib.dataset import EuroSATDataset, discover_samples
from lib.splits import make_stratified_splits

TaskName = Literal["multiclass", "binary"]

VEGETATION_CLASSES = {
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Pasture",
    "PermanentCrop",
}

BINARY_CLASS_NAMES = ["Non-vegetation", "Vegetation"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class TrainConfig:
    task: TaskName
    data_root: str
    output_dir: str
    image_size: int
    batch_size: int
    epochs: int
    freeze_epochs: int
    unfreeze_blocks: int
    lr_head: float
    lr_backbone: float
    weight_decay: float
    num_workers: int
    seed: int
    device: str
    pretrained: bool


class ScaleSentinelReflectance:
    """Scale Sentinel-2 reflectance values to approximately [0, 1]."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor / float(S2_REFLECTANCE_SCALE)
        return torch.clamp(tensor, 0.0, 1.0)


class RelabelDataset(Dataset):
    """Wrap a dataset and replace its original EuroSAT labels."""

    def __init__(self, dataset: Dataset, label_mapper: Callable[[int], int]) -> None:
        self.dataset = dataset
        self.label_mapper = label_mapper

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        return image, self.label_mapper(int(label))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 transfer-learning models on EuroSAT RGB bands."
    )
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary", "all"],
        default="all",
        help="Training task. Use 'all' to train multiclass and binary sequentially.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PATH_TO_DATASET,
        help="Path to extracted EuroSAT_MS folder with class subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "experiments" / "efficientnet_b0",
        help="Root folder for experiment outputs.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="Number of initial epochs with frozen EfficientNet feature extractor.",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=2,
        help="How many final EfficientNet feature blocks to unfreeze after freeze phase.",
    )
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, ...",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet weights. Mainly for debugging; not recommended for final Model 2.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assert_dataset_available(data_root: Path) -> None:
    if data_root.is_dir():
        return

    zip_candidate = data_root.with_suffix(".zip")
    if zip_candidate.exists():
        content = zip_candidate.read_bytes()[:200]
        if b"version https://git-lfs.github.com/spec" in content:
            raise FileNotFoundError(
                f"Dataset folder not found: {data_root}\n"
                f"Found {zip_candidate}, but it is only a Git LFS pointer.\n"
                "Download EuroSAT_MS.zip from https://github.com/phelber/eurosat and extract it to data/eurosat/."
            )
        raise FileNotFoundError(
            f"Dataset folder not found: {data_root}\n"
            f"A zip file exists at {zip_candidate}. Extract it so the folder EuroSAT_MS is available."
        )

    raise FileNotFoundError(
        f"Dataset folder not found: {data_root}\n"
        "Expected structure: data/eurosat/EuroSAT_MS/<class_name>/*.tif"
    )


def make_transforms(image_size: int) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    base = [
        ScaleSentinelReflectance(),
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    train_transform = transforms.Compose(
        [
            *base,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            *base,
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def binary_label_mapper(label: int) -> int:
    class_name = CLASSES[label]
    return 1 if class_name in VEGETATION_CLASSES else 0


def make_dataloaders(
    task: TaskName,
    data_root: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[dict[str, DataLoader], list[str]]:
    samples = discover_samples(data_root)
    splits = make_stratified_splits(samples, seed=seed)
    train_transform, eval_transform = make_transforms(image_size)

    train_dataset: Dataset = EuroSATDataset(
        splits["train"], transform=train_transform, rgb_only=True
    )
    val_dataset: Dataset = EuroSATDataset(
        splits["val"], transform=eval_transform, rgb_only=True
    )
    test_dataset: Dataset = EuroSATDataset(
        splits["test"], transform=eval_transform, rgb_only=True
    )

    if task == "binary":
        train_dataset = RelabelDataset(train_dataset, binary_label_mapper)
        val_dataset = RelabelDataset(val_dataset, binary_label_mapper)
        test_dataset = RelabelDataset(test_dataset, binary_label_mapper)
        class_names = BINARY_CLASS_NAMES
    else:
        class_names = CLASSES

    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        # Keep worker processes alive across epochs. Recreating them every epoch
        # leaks file descriptors and, over a long (50-epoch) run, exhausts the
        # open-file limit with "Too many open files" on os.pipe() worker spawn.
        "persistent_workers": num_workers > 0,
    }

    loaders = {
        "train": DataLoader(
            train_dataset,
            shuffle=True,
            generator=generator,
            drop_last=False,
            **loader_kwargs,
        ),
        "val": DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs),
    }
    return loaders, class_names


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, num_blocks: int) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if num_blocks <= 0:
        return

    feature_blocks = list(model.features.children())
    for block in feature_blocks[-num_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def make_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = [
        parameter
        for parameter in model.features.parameters()
        if parameter.requires_grad
    ]
    head_params = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    param_groups: list[dict[str, Any]] = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Epoch",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    progress_bar = tqdm(loader, desc=description, leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        predictions = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

        current_loss = total_loss / max(total_samples, 1)
        current_acc = accuracy_score(y_true, y_pred)

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}",
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
    return y_true, y_pred


def save_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNet-B0 loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="train")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EfficientNet-B0 accuracy curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    size = max(6, min(12, len(class_names) * 0.8))
    plt.figure(figsize=(size, size))
    plt.imshow(matrix)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            text_color = "white" if value > threshold else "black"
            plt.text(col, row, value, ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_one_task(args: argparse.Namespace, task: TaskName) -> None:
    set_seed(args.seed)
    data_root = Path(args.data_root)
    assert_dataset_available(data_root)

    output_dir = Path(args.output_root) / task
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    loaders, class_names = make_dataloaders(
        task=task,
        data_root=data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    config = TrainConfig(
        task=task,
        data_root=str(data_root),
        output_dir=str(output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        unfreeze_blocks=args.unfreeze_blocks,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=str(device),
        pretrained=not args.no_pretrained,
    )

    print(f"\n=== EfficientNet-B0 | task={task} | device={device} ===")
    print(json.dumps(asdict(config), indent=2))

    model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    optimizer: torch.optim.Optimizer | None = None
    current_phase = ""

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if epoch <= args.freeze_epochs:
            phase = "frozen_head"
            if phase != current_phase:
                freeze_backbone(model)
                optimizer = make_optimizer(
                    model, args.lr_head, args.lr_backbone, args.weight_decay
                )
                current_phase = phase
        else:
            phase = "finetune_last_blocks"
            if phase != current_phase:
                unfreeze_last_blocks(model, args.unfreeze_blocks)
                optimizer = make_optimizer(
                    model, args.lr_head, args.lr_backbone, args.weight_decay
                )
                current_phase = phase

        if optimizer is None:
            raise RuntimeError("Optimizer was not initialized.")

        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer, description="Training")
        val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None, description="Validation")
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_time_seconds": epoch_time,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"[{phase}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "class_names": class_names,
                    "best_epoch": best_epoch,
                    "best_val_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

    save_history_csv(history, output_dir / "history.csv")
    plot_training_curves(history, output_dir)

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None)
    test_metrics = run_epoch(model, loaders["test"], criterion, device, optimizer=None)
    y_true, y_pred = predict(model, loaders["test"], device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    metrics = {
        "config": asdict(config),
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val": val_metrics,
        "test": test_metrics,
        "test_confusion_matrix": cm.tolist(),
        "test_classification_report": report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nFinal metrics from best checkpoint:")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))
    print(f"Saved outputs to: {output_dir}")


def main() -> None:
    args = parse_args()
    tasks: list[TaskName]
    if args.task == "all":
        tasks = ["multiclass", "binary"]
    else:
        tasks = [args.task]

    for task in tasks:
        train_one_task(args, task)


if __name__ == "__main__":
    main()
