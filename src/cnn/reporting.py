"""Generate the per-run evaluation artifacts (metrics, plots).

Shared by :mod:`cnn.train` (after training) and
:mod:`cnn.evaluate` (standalone re-evaluation of an existing run).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from . import plots
from .data import DatasetArtifacts
from .engine import evaluate
from .metrics import compute_metrics
from .model import count_trainable_parameters
from .utils import write_json


def generate_run_report(
    run_dir: Path,
    model: nn.Module,
    artifacts: DatasetArtifacts,
    task: str,
    device: torch.device,
    loaders: dict,
    criterion: nn.Module,
    history: dict | None = None,
    best_val_loss: float | None = None,
    epochs_completed: int | None = None,
    progress: bool = True,
    evaluate_test: bool = True,
) -> dict:
    """Evaluate the run, write metrics + plots, and return a summary dict."""
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_split = "test" if evaluate_test else "val"
    class_order = loaders[plot_split].dataset.class_order

    val_loss, val_preds, val_targets = evaluate(
        model, loaders["val"], criterion, device, progress=progress, desc="val"
    )

    val_metrics = compute_metrics(val_targets, val_preds, class_order)
    test_loss = None
    test_metrics = None
    test_preds = None
    test_targets = None
    if evaluate_test:
        test_loss, test_preds, test_targets = evaluate(
            model, loaders["test"], criterion, device, progress=progress, desc="test"
        )
        test_metrics = compute_metrics(test_targets, test_preds, class_order)

    # --- metrics.json (detailed) ---
    trainable_parameters = count_trainable_parameters(model)
    metrics_payload = {
        "task": task,
        "evaluation": "val_test" if evaluate_test else "val_only",
        "model": {
            "class_name": model.__class__.__name__,
            "trainable_parameters": trainable_parameters,
        },
        "class_order": class_order,
        "epochs_completed": epochs_completed,
        "best_val_loss": best_val_loss,
        "val": {"loss": val_loss, **val_metrics},
    }
    if evaluate_test and test_metrics is not None:
        metrics_payload["test"] = {"loss": test_loss, **test_metrics}
    write_json(run_dir / "metrics.json", metrics_payload)

    # --- metrics.csv (flat summary, one row per split) ---
    summary_rows = [
        {
            "split": "val",
            "trainable_parameters": trainable_parameters,
            "loss": val_loss,
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
        },
    ]
    if evaluate_test and test_metrics is not None:
        summary_rows.append(
            {
                "split": "test",
                "trainable_parameters": trainable_parameters,
                "loss": test_loss,
                "accuracy": test_metrics["accuracy"],
                "macro_f1": test_metrics["macro_f1"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics.csv", index=False)

    # --- plots ---
    if history is not None:
        plots.plot_loss_curve(history, plots_dir / "loss_curve.png")
        plots.plot_accuracy_curve(history, plots_dir / "accuracy_curve.png")

    plot_metrics = test_metrics if evaluate_test and test_metrics is not None else val_metrics
    plot_preds = test_preds if evaluate_test and test_preds is not None else val_preds
    plot_targets = test_targets if evaluate_test and test_targets is not None else val_targets

    plots.plot_confusion_matrix(
        plot_metrics["confusion_matrix"], class_order, plots_dir / "confusion_matrix.png"
    )
    plots.plot_class_distribution(
        artifacts.splits,
        artifacts.class_order,
        class_order,
        task,
        plots_dir / "class_distribution.png",
    )
    plots.plot_misclassified_grid(
        loaders[plot_split].dataset,
        plot_preds,
        plot_targets,
        class_order,
        artifacts.mean,
        artifacts.std,
        plots_dir / "misclassified_grid.png",
    )

    return {
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"] if test_metrics is not None else None,
        "test_macro_f1": test_metrics["macro_f1"] if test_metrics is not None else None,
        "trainable_parameters": trainable_parameters,
    }
