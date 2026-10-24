"""Dataset preparation and loading for EuroSAT.

This module owns everything between the raw zip and ready-to-train tensors:

* extracting the archive (only when missing),
* discovering the class order,
* building one stratified 70/15/15 split (seed 42) reused by both tasks,
* computing the RGB normalization statistics from the *training split only*,
* the :class:`EuroSATDataset` and the train/val/test transforms.

The prepared artifacts (class order, split, normalization) are persisted as
JSON so training and evaluation are fully reproducible and the split is shared
across tasks.
"""

from __future__ import annotations

import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from . import labels
from .utils import read_json, write_json

# Default on-disk locations (relative to the data root, typically ``data/``).
EXTRACTED_DIRNAME = "EuroSAT_RGB"
PROCESSED_DIRNAME = "processed"
CLASS_ORDER_FILE = "class_order.json"
SPLIT_FILE = "split.json"
NORMALIZATION_FILE = "normalization.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# A sample is a (relative image path, multiclass index) pair.
Sample = tuple[str, int]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


# --------------------------------------------------------------------------- #
# Extraction and discovery
# --------------------------------------------------------------------------- #
def extract_dataset(zip_path: Path, data_root: Path) -> Path:
    """Extract ``zip_path`` into ``data_root`` only when missing.

    Returns the path to the extracted ``EuroSAT_RGB`` directory.
    """
    extracted_dir = data_root / EXTRACTED_DIRNAME
    if extracted_dir.is_dir() and any(extracted_dir.iterdir()):
        return extracted_dir

    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {zip_path}")

    data_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)

    if not extracted_dir.is_dir():
        raise RuntimeError(
            f"Expected '{EXTRACTED_DIRNAME}/' after extracting {zip_path}, "
            f"but it was not found in {data_root}."
        )
    return extracted_dir


def discover_classes(extracted_dir: Path) -> list[str]:
    """Return the EuroSAT class names in a stable (alphabetical) order."""
    classes = sorted(
        p.name for p in extracted_dir.iterdir() if p.is_dir()
    )
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {extracted_dir}")
    return classes


def list_samples(extracted_dir: Path, class_order: list[str]) -> list[Sample]:
    """Collect all (relative path, class index) samples in a stable order."""
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}
    samples: list[Sample] = []
    for class_name in class_order:
        class_dir = extracted_dir / class_name
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for file_path in files:
            rel = file_path.relative_to(extracted_dir).as_posix()
            samples.append((rel, class_to_idx[class_name]))
    return samples


# --------------------------------------------------------------------------- #
# Stratified split
# --------------------------------------------------------------------------- #
def stratified_split(
    samples: list[Sample],
    seed: int = 42,
    ratios: dict[str, float] = SPLIT_RATIOS,
) -> dict[str, list[Sample]]:
    """Create a deterministic stratified train/val/test split.

    Stratification uses the 10-class labels, which keeps both the multiclass
    and (coarser) binary class balances stable across splits.
    """
    paths = [s[0] for s in samples]
    targets = [s[1] for s in samples]

    # First carve off the test set, then split the remainder into train/val.
    rel_val = ratios["val"] / (ratios["train"] + ratios["val"])
    train_val_paths, test_paths, train_val_targets, test_targets = train_test_split(
        paths,
        targets,
        test_size=ratios["test"],
        random_state=seed,
        stratify=targets,
    )
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        train_val_paths,
        train_val_targets,
        test_size=rel_val,
        random_state=seed,
        stratify=train_val_targets,
    )

    return {
        "train": list(zip(train_paths, train_targets)),
        "val": list(zip(val_paths, val_targets)),
        "test": list(zip(test_paths, test_targets)),
    }


# --------------------------------------------------------------------------- #
# Normalization statistics (training split only)
# --------------------------------------------------------------------------- #
def compute_normalization(
    extracted_dir: Path,
    train_samples: list[Sample],
    show_progress: bool = True,
) -> tuple[list[float], list[float]]:
    """Compute per-channel RGB mean/std over the *training* split only.

    Images are read as float tensors in [0, 1] (ToTensor scaling). Statistics
    are accumulated with running sums to keep memory flat.
    """
    to_tensor = transforms.ToTensor()
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    iterator = train_samples
    if show_progress:
        iterator = tqdm(train_samples, desc="Computing normalization", unit="img")

    for rel, _ in iterator:
        with Image.open(extracted_dir / rel) as img:
            tensor = to_tensor(img.convert("RGB")).to(torch.float64)
        # tensor: (3, H, W)
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sq_sum += (tensor**2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    if pixel_count == 0:
        raise RuntimeError("No training pixels available for normalization.")

    mean = channel_sum / pixel_count
    var = channel_sq_sum / pixel_count - mean**2
    std = torch.sqrt(var.clamp_min(0.0))
    return mean.tolist(), std.tolist()


# --------------------------------------------------------------------------- #
# Transforms and dataset
# --------------------------------------------------------------------------- #
def build_transforms(
    mean: list[float],
    std: list[float],
    train: bool,
    image_size: int = 64,
    rotation_degrees: int = 15,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
) -> transforms.Compose:
    """Build the transform pipeline. Augmentation is training-only."""
    steps: list = []
    if train:
        if horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip())
        if vertical_flip:
            steps.append(transforms.RandomVerticalFlip())
        if rotation_degrees:
            steps.append(transforms.RandomRotation(rotation_degrees))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


class EuroSATDataset(Dataset):
    """EuroSAT image dataset for either the multiclass or binary task.

    The same underlying samples (with their 10-class index) are reused for both
    tasks; only the returned label changes according to ``task``.
    """

    def __init__(
        self,
        samples: list[Sample],
        root: Path,
        task: str,
        multiclass_order: list[str],
        transform: transforms.Compose,
    ) -> None:
        if task not in ("multiclass", "binary"):
            raise ValueError(f"Unknown task: {task!r}")
        self.samples = samples
        self.root = Path(root)
        self.task = task
        self.multiclass_order = multiclass_order
        self.transform = transform
        self.class_order = labels.class_order_for_task(task, multiclass_order)
        # Precompute the task labels (also used for class-distribution plots).
        self.targets = [self._label_for(idx) for _, idx in samples]

    def _label_for(self, multiclass_idx: int) -> int:
        if self.task == "multiclass":
            return multiclass_idx
        class_name = self.multiclass_order[multiclass_idx]
        return labels.to_binary_index(class_name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        rel, _ = self.samples[index]
        with Image.open(self.root / rel) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.targets[index]


# --------------------------------------------------------------------------- #
# Artifact persistence
# --------------------------------------------------------------------------- #
@dataclass
class DatasetArtifacts:
    """In-memory view of the prepared dataset artifacts."""

    extracted_dir: Path
    class_order: list[str]
    splits: dict[str, list[Sample]]
    mean: list[float]
    std: list[float]
    seed: int
    image_size: int


def save_artifacts(
    processed_dir: Path,
    extracted_dir: Path,
    class_order: list[str],
    splits: dict[str, list[Sample]],
    mean: list[float],
    std: list[float],
    seed: int,
    image_size: int,
) -> None:
    """Persist class order, split and normalization to ``processed_dir``."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    write_json(processed_dir / CLASS_ORDER_FILE, {"classes": class_order})

    write_json(
        processed_dir / SPLIT_FILE,
        {
            "seed": seed,
            "ratios": SPLIT_RATIOS,
            "extracted_dir": str(extracted_dir),
            "class_order": class_order,
            "counts": {name: len(items) for name, items in splits.items()},
            # Samples stored as [relative_path, multiclass_index].
            "samples": {
                name: [[rel, idx] for rel, idx in items]
                for name, items in splits.items()
            },
        },
    )

    write_json(
        processed_dir / NORMALIZATION_FILE,
        {
            "mean": mean,
            "std": std,
            "image_size": image_size,
            "computed_from": "train",
            "num_train_images": len(splits["train"]),
        },
    )


def load_artifacts(processed_dir: Path) -> DatasetArtifacts:
    """Load prepared artifacts previously written by :func:`save_artifacts`."""
    processed_dir = Path(processed_dir)
    split_path = processed_dir / SPLIT_FILE
    norm_path = processed_dir / NORMALIZATION_FILE
    if not split_path.is_file() or not norm_path.is_file():
        raise FileNotFoundError(
            f"Prepared artifacts not found in {processed_dir}. "
            "Run 'python -m cnn.prepare_data --zip <archive>' first."
        )

    split = read_json(split_path)
    norm = read_json(norm_path)

    splits = {
        name: [(rel, int(idx)) for rel, idx in items]
        for name, items in split["samples"].items()
    }
    return DatasetArtifacts(
        extracted_dir=Path(split["extracted_dir"]),
        class_order=list(split["class_order"]),
        splits=splits,
        mean=list(norm["mean"]),
        std=list(norm["std"]),
        seed=int(split["seed"]),
        image_size=int(norm["image_size"]),
    )


# --------------------------------------------------------------------------- #
# Loader construction
# --------------------------------------------------------------------------- #
def limit_per_class(
    samples: list[Sample], limit: int
) -> list[Sample]:
    """Keep at most ``limit`` samples per (10-class) label. Used by smoke mode."""
    counts: Counter[int] = Counter()
    kept: list[Sample] = []
    for rel, idx in samples:
        if counts[idx] < limit:
            kept.append((rel, idx))
            counts[idx] += 1
    return kept


def balanced_subset(
    samples: list[Sample],
    total: int,
    class_count: int,
) -> list[Sample]:
    """Return exactly ``total`` samples balanced over the 10-class labels."""
    if total <= 0:
        raise ValueError("Balanced subset size must be positive.")
    if total % class_count != 0:
        raise ValueError(
            f"Balanced subset size {total} is not divisible by {class_count} classes."
        )

    per_class = total // class_count
    counts: Counter[int] = Counter()
    kept: list[Sample] = []
    for rel, idx in samples:
        if counts[idx] < per_class:
            kept.append((rel, idx))
            counts[idx] += 1

    missing = [idx for idx in range(class_count) if counts[idx] != per_class]
    if missing:
        raise ValueError(
            f"Could not build balanced subset of {total}; missing enough samples "
            f"for class indices {missing}."
        )
    return kept


def build_dataloaders(
    artifacts: DatasetArtifacts,
    task: str,
    batch_size: int,
    num_workers: int,
    augmentation: dict,
    limit: int | None = None,
    split_overrides: dict[str, list[Sample]] | None = None,
) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders for a task from prepared artifacts."""
    train_tf = build_transforms(
        artifacts.mean,
        artifacts.std,
        train=True,
        image_size=artifacts.image_size,
        rotation_degrees=augmentation.get("random_rotation_degrees", 15),
        horizontal_flip=augmentation.get("random_horizontal_flip", True),
        vertical_flip=augmentation.get("random_vertical_flip", True),
    )
    eval_tf = build_transforms(
        artifacts.mean, artifacts.std, train=False, image_size=artifacts.image_size
    )

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        samples = (
            split_overrides[name]
            if split_overrides is not None and name in split_overrides
            else artifacts.splits[name]
        )
        if limit is not None:
            samples = limit_per_class(samples, limit)
        dataset = EuroSATDataset(
            samples=samples,
            root=artifacts.extracted_dir,
            task=task,
            multiclass_order=artifacts.class_order,
            transform=train_tf if name == "train" else eval_tf,
        )
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
        )
    return loaders
