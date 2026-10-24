"""CLI: train a configured scratch CNN model on the multiclass or binary task.

Examples::

    python -m cnn.train --task multiclass --config configs/scratch_cnn.yaml
    python -m cnn.train --task binary --config configs/scratch_cnn.yaml
    python -m cnn.train --task multiclass --smoke   # fast CPU sanity run

Each run writes to ``outputs/runs/<timestamp>_<task>/`` and appends a row to the
shared comparison CSV.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch import nn

from . import data as data_mod
from .comparison import update_comparison
from .engine import train_one_epoch, evaluate
from .labels import num_classes_for_task
from .model import build_model, count_trainable_parameters
from .optim import build_optimizer
from .reporting import generate_run_report
from .utils import resolve_device, set_seed, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a EuroSAT CNN model.")
    parser.add_argument(
        "--task", choices=["multiclass", "binary"], required=True, help="Which task to train."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scratch_cnn.yaml"),
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with prepared artifacts (from prepare_data).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for run outputs and the comparison CSV.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run directory name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--device", default="auto", help="Device: 'auto', 'cpu' or 'cuda'."
    )
    parser.add_argument(
        "--val-only",
        action="store_true",
        help="Evaluate validation only; do not compute or write test metrics.",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=None,
        help="Train and validate on a balanced subset from the training split.",
    )
    # Smoke / override flags.
    parser.add_argument(
        "--smoke", action="store_true", help="Fast CPU sanity run (few epochs, few samples)."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Cap samples per class per split (for quick runs).",
    )
    return parser


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_settings(config: dict, args: argparse.Namespace) -> dict:
    """Merge config defaults with CLI/smoke overrides into a resolved dict."""
    resolved = copy.deepcopy(config)
    smoke_cfg = resolved.get("smoke", {})

    max_epochs = resolved["training"]["max_epochs"]
    limit = None
    if args.smoke:
        max_epochs = smoke_cfg.get("epochs", 1)
        limit = smoke_cfg.get("limit_per_class", 8)
    # Explicit CLI flags take precedence over both defaults and smoke.
    if args.epochs is not None:
        max_epochs = args.epochs
    if args.limit_per_class is not None:
        limit = args.limit_per_class

    resolved["training"]["max_epochs"] = max_epochs
    resolved["_runtime"] = {
        "task": args.task,
        "smoke": args.smoke,
        "limit_per_class": limit,
        "device_request": args.device,
        "run_name": args.run_name,
        "evaluate_test": not args.val_only,
        "overfit_samples": args.overfit_samples,
    }
    return resolved


def describe_augmentation(settings: dict) -> str:
    """Return a compact label for the configured training augmentation."""
    aug = settings.get("augmentation", {})
    aug_desc = "+".join(
        name
        for name, key in [
            ("hflip", "random_horizontal_flip"),
            ("vflip", "random_vertical_flip"),
        ]
        if aug.get(key)
    )
    if aug.get("random_rotation_degrees"):
        aug_desc += f"+rot{aug['random_rotation_degrees']}"
    return aug_desc.strip("+") or "none"


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    settings = resolve_settings(config, args)

    seed = settings.get("seed", 42)
    set_seed(seed)
    device = resolve_device("cpu" if args.smoke and args.device == "auto" else args.device)

    # --- data ---
    artifacts = data_mod.load_artifacts(args.processed_dir)
    num_classes = num_classes_for_task(args.task, artifacts.class_order)
    limit = settings["_runtime"]["limit_per_class"]
    split_overrides = None
    if args.overfit_samples is not None:
        subset = data_mod.balanced_subset(
            artifacts.splits["train"],
            total=args.overfit_samples,
            class_count=len(artifacts.class_order),
        )
        split_overrides = {"train": subset, "val": subset, "test": subset}

    loaders = data_mod.build_dataloaders(
        artifacts=artifacts,
        task=args.task,
        batch_size=settings["training"]["batch_size"],
        num_workers=0 if args.smoke else settings["data"]["num_workers"],
        augmentation=settings.get("augmentation", {}),
        limit=limit,
        split_overrides=split_overrides,
    )

    # --- run directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_name or f"{timestamp}_{args.task}" + ("_smoke" if args.smoke else "")
    run_dir = args.output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", settings)
    write_json(
        run_dir / "split_reference.json",
        {
            "seed": artifacts.seed,
            "ratios": data_mod.SPLIT_RATIOS,
            "processed_dir": str(args.processed_dir.resolve()),
            "class_order": artifacts.class_order,
            "counts": {name: len(items) for name, items in artifacts.splits.items()},
            "effective_counts": {
                name: len(loader.dataset) for name, loader in loaders.items()
            },
            "overfit_samples": args.overfit_samples,
            "normalization": {"mean": artifacts.mean, "std": artifacts.std},
        },
    )

    # --- model / optimizer / loss ---
    model = build_model(settings["model"], num_classes=num_classes).to(device)
    trainable_parameters = count_trainable_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), settings["optimizer"])

    # --- training loop with early stopping on val loss ---
    max_epochs = settings["training"]["max_epochs"]
    patience = settings["training"]["early_stopping_patience"]
    progress = not args.smoke

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    checkpoint_path = run_dir / "best_model.pt"

    print(f"Training task={args.task} on {device} for up to {max_epochs} epoch(s).")
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, progress=progress
        )
        val_loss, val_preds, val_targets = evaluate(
            model, loaders["val"], criterion, device, progress=progress, desc="val"
        )
        val_acc = float((val_preds == val_targets).mean()) if len(val_targets) else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:>3}/{max_epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "task": args.task,
                    "num_classes": num_classes,
                    "dropout": settings["model"]["dropout"],
                    "model_name": settings["model"].get("name", "scratch_cnn"),
                    "trainable_parameters": trainable_parameters,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "class_order": loaders["test"].dataset.class_order,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience}).")
                break

    epochs_completed = len(history["epoch"])
    write_json(run_dir / "history.json", history)

    # --- load best checkpoint and produce the evaluation report ---
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {best_epoch} (val_loss={best_val_loss:.4f}).")

    summary = generate_run_report(
        run_dir=run_dir,
        model=model,
        artifacts=artifacts,
        task=args.task,
        device=device,
        loaders=loaders,
        criterion=criterion,
        history=history,
        best_val_loss=best_val_loss,
        epochs_completed=epochs_completed,
        progress=progress,
        evaluate_test=not args.val_only,
    )

    # --- comparison CSV ---
    aug_desc = describe_augmentation(settings)

    update_comparison(
        args.output_root / "comparison.csv",
        {
            "run_id": run_id,
            "task": args.task,
            "model_name": settings["model"].get("name", "scratch_cnn"),
            "approach": "scratch",
            "trainable_parameters": summary["trainable_parameters"],
            "augmentation": aug_desc,
            "optimizer": settings["optimizer"]["name"],
            "lr": settings["optimizer"]["lr"],
            "weight_decay": settings["optimizer"]["weight_decay"],
            "dropout": settings["model"]["dropout"],
            "epochs_completed": epochs_completed,
            "best_val_loss": best_val_loss,
            "val_accuracy": summary["val_accuracy"],
            "val_macro_f1": summary["val_macro_f1"],
            "test_accuracy": (
                summary["test_accuracy"] if summary["test_accuracy"] is not None else ""
            ),
            "test_macro_f1": (
                summary["test_macro_f1"] if summary["test_macro_f1"] is not None else ""
            ),
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )

    print(f"Run complete. Artifacts written to {run_dir}/")
    print(f"  val   acc={summary['val_accuracy']:.4f} f1={summary['val_macro_f1']:.4f}")
    if summary["test_accuracy"] is not None:
        print(
            f"  test  acc={summary['test_accuracy']:.4f} "
            f"f1={summary['test_macro_f1']:.4f}"
        )
    return run_dir


if __name__ == "__main__":
    main()
