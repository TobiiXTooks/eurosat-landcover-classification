"""Label definitions for the two EuroSAT tasks.

The 10-class label order is *discovered* from the extracted dataset directory
(see :func:`cnn.data.discover_classes`) and saved for reproducibility.
The binary mapping below is fixed by the project specification.
"""

from __future__ import annotations

# EuroSAT classes that depict vegetation. The remaining five classes are
# treated as non-vegetation. This mapping is fixed by the project spec.
VEGETATION_CLASSES: frozenset[str] = frozenset(
    {
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Pasture",
        "PermanentCrop",
    }
)

NON_VEGETATION_CLASSES: frozenset[str] = frozenset(
    {
        "Highway",
        "Industrial",
        "Residential",
        "River",
        "SeaLake",
    }
)

# Stable binary class order. ``vegetation`` is the positive class (index 1).
BINARY_CLASS_ORDER: list[str] = ["non_vegetation", "vegetation"]


def to_binary_name(class_name: str) -> str:
    """Map a 10-class EuroSAT label to its binary label name."""
    if class_name in VEGETATION_CLASSES:
        return "vegetation"
    if class_name in NON_VEGETATION_CLASSES:
        return "non_vegetation"
    raise KeyError(f"Unknown EuroSAT class: {class_name!r}")


def to_binary_index(class_name: str) -> int:
    """Map a 10-class EuroSAT label to its binary class index (0 or 1)."""
    return BINARY_CLASS_ORDER.index(to_binary_name(class_name))


def class_order_for_task(task: str, multiclass_order: list[str]) -> list[str]:
    """Return the ordered class names used for a given task."""
    if task == "multiclass":
        return list(multiclass_order)
    if task == "binary":
        return list(BINARY_CLASS_ORDER)
    raise ValueError(f"Unknown task: {task!r} (expected 'multiclass' or 'binary')")


def num_classes_for_task(task: str, multiclass_order: list[str]) -> int:
    """Return the number of output classes for a given task."""
    return len(class_order_for_task(task, multiclass_order))
