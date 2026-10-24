"""Metric computation helpers built on scikit-learn."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_order: list[str],
) -> dict:
    """Return a dict of accuracy, macro-F1, per-class report and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    label_indices = list(range(len(class_order)))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=label_indices, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=class_order,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }
