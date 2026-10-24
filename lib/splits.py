from collections.abc import Sequence
from typing import TypedDict

from sklearn.model_selection import train_test_split

from lib.constants import SEED
from lib.dataset import Sample


class Splits(TypedDict):
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]

# ! AI Generated helper function
def make_stratified_splits(
    samples: Sequence[Sample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> Splits:
    """Stratified train/val/test split.

    Identical inputs + seed produce identical splits, so Model 1 and Model 2
    are evaluated on the exact same test set — required for a fair comparison.
    """
    if not 0 < val_ratio + test_ratio < 1:
        raise ValueError("val_ratio + test_ratio must lie in (0, 1)")

    labels = [label for _, label in samples]
    train_val, test = train_test_split(
        list(samples),
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )
    train_val_labels = [label for _, label in train_val]
    val_relative = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val_labels,
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}
