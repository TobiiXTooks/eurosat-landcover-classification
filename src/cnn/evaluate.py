"""CLI: re-evaluate a finished run from its run directory.

Usage::

    python -m cnn.evaluate --run-dir outputs/runs/<run_id>

Loads the run's resolved config and best checkpoint, rebuilds the val/test
loaders from the same prepared split, and regenerates the metrics + plots
(reusing the training history if present for the loss/accuracy curves).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from . import data as data_mod
from .labels import num_classes_for_task
from .model import build_model
from .reporting import generate_run_report
from .utils import read_json, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a finished EuroSAT run.")
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Path to outputs/runs/<run_id>."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override prepared-artifacts directory (default: from split_reference.json).",
    )
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu' or 'cuda'.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config = read_json(run_dir / "config.json")
    split_ref = read_json(run_dir / "split_reference.json")
    task = config["_runtime"]["task"]
    seed = config.get("seed", 42)
    set_seed(seed)
    device = resolve_device(args.device)

    processed_dir = args.processed_dir or Path(split_ref["processed_dir"])
    artifacts = data_mod.load_artifacts(processed_dir)

    limit = config["_runtime"].get("limit_per_class")
    loaders = data_mod.build_dataloaders(
        artifacts=artifacts,
        task=task,
        batch_size=config["training"]["batch_size"],
        num_workers=0,
        augmentation=config.get("augmentation", {}),
        limit=limit,
    )

    num_classes = num_classes_for_task(task, artifacts.class_order)
    model = build_model(config.get("model", {}), num_classes=num_classes).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    history_path = run_dir / "history.json"
    history = read_json(history_path) if history_path.is_file() else None

    criterion = nn.CrossEntropyLoss()
    summary = generate_run_report(
        run_dir=run_dir,
        model=model,
        artifacts=artifacts,
        task=task,
        device=device,
        loaders=loaders,
        criterion=criterion,
        history=history,
        best_val_loss=checkpoint.get("val_loss"),
        epochs_completed=len(history["epoch"]) if history else None,
        progress=True,
    )

    print(f"Re-evaluated {run_dir}/")
    print(
        f"  val   acc={summary['val_accuracy']:.4f} f1={summary['val_macro_f1']:.4f}\n"
        f"  test  acc={summary['test_accuracy']:.4f} f1={summary['test_macro_f1']:.4f}"
    )
    return run_dir


if __name__ == "__main__":
    main()
