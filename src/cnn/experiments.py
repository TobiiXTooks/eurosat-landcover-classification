"""Run the course-practical experiment campaign.

The campaign runner is intentionally thin: it creates resolved configs for each
experiment and delegates actual training/evaluation to ``cnn.train``.
That keeps ablation runs and final runs on the exact same training code path.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import train as train_mod
from .utils import read_json, write_json


SUMMARY_COLUMNS = [
    "campaign_id",
    "experiment_id",
    "task",
    "phase",
    "model_name",
    "trainable_parameters",
    "optimizer",
    "lr",
    "momentum",
    "weight_decay",
    "dropout",
    "augmentation",
    "epochs_completed",
    "best_val_loss",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "run_dir",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the EuroSAT experiment campaign.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/course_practical.yaml"),
        help="Path to the experiment campaign config.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Override processed artifacts directory from the campaign config.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override experiment output root from the campaign config.",
    )
    parser.add_argument("--device", default="auto", help="Device passed to training.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run every experiment as a tiny CPU smoke test.",
    )
    return parser


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def deep_merge(base: dict, overrides: dict | None) -> dict:
    """Recursively merge ``overrides`` into a copied ``base`` dict."""
    result = copy.deepcopy(base)
    if not overrides:
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _metric(metrics: dict, split: str, key: str) -> Any:
    if split not in metrics:
        return ""
    return metrics[split].get(key, "")


def _row_from_run(
    campaign_id: str,
    experiment_id: str,
    phase: str,
    run_dir: Path,
) -> dict:
    metrics = read_json(run_dir / "metrics.json")
    resolved = read_json(run_dir / "config.json")
    optimizer = resolved["optimizer"]
    model_metrics = metrics.get("model", {})

    row = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
        "task": metrics["task"],
        "phase": phase,
        "model_name": resolved.get("model", {}).get("name", "scratch_cnn"),
        "trainable_parameters": model_metrics.get("trainable_parameters", ""),
        "optimizer": optimizer["name"],
        "lr": optimizer["lr"],
        "momentum": optimizer.get("momentum", ""),
        "weight_decay": optimizer.get("weight_decay", ""),
        "dropout": resolved["model"]["dropout"],
        "augmentation": train_mod.describe_augmentation(resolved),
        "epochs_completed": metrics.get("epochs_completed", ""),
        "best_val_loss": metrics.get("best_val_loss", ""),
        "val_accuracy": _metric(metrics, "val", "accuracy"),
        "val_macro_f1": _metric(metrics, "val", "macro_f1"),
        "test_accuracy": _metric(metrics, "test", "accuracy"),
        "test_macro_f1": _metric(metrics, "test", "macro_f1"),
        "run_dir": str(run_dir.resolve()),
    }
    return {col: row.get(col, "") for col in SUMMARY_COLUMNS}


def _run_training(
    experiment_id: str,
    task: str,
    phase: str,
    resolved_config: dict,
    campaign_dir: Path,
    processed_dir: Path,
    device: str,
    evaluate_test: bool,
    smoke: bool,
    overfit_samples: int | None = None,
) -> Path:
    config_path = campaign_dir / "configs" / f"{experiment_id}.yaml"
    write_yaml(config_path, resolved_config)

    argv = [
        "--task",
        task,
        "--config",
        str(config_path),
        "--processed-dir",
        str(processed_dir),
        "--output-root",
        str(campaign_dir),
        "--run-name",
        experiment_id,
        "--device",
        device,
    ]
    if not evaluate_test:
        argv.append("--val-only")
    if overfit_samples is not None:
        argv.extend(["--overfit-samples", str(20 if smoke else overfit_samples)])
    if smoke:
        argv.append("--smoke")

    print(f"\n=== {experiment_id} ({phase}, task={task}) ===")
    return train_mod.main(argv)


def _select_best_ablation(rows: list[dict]) -> dict:
    ablations = [row for row in rows if row["phase"] == "ablation"]
    if not ablations:
        raise RuntimeError("No ablation rows available for model selection.")
    return max(
        ablations,
        key=lambda row: (
            float(row["val_macro_f1"]),
            -float(row["best_val_loss"]),
        ),
    )


def _write_summary_markdown(
    path: Path,
    rows: list[dict],
    selection: dict,
) -> None:
    lines = [
        f"# Experiment Campaign: {selection['campaign_id']}",
        "",
        "Validation metrics decide the ablation winner. Test metrics are present only "
        "for final runs.",
        "",
        f"Selected configuration: `{selection['selected_experiment_id']}` by "
        "`highest val_macro_f1, tie lowest best_val_loss`.",
        "",
        "| experiment | phase | task | model | params | optimizer | aug | val_f1 | test_f1 |",
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        val_f1 = f"{float(row['val_macro_f1']):.4f}"
        test_f1 = (
            f"{float(row['test_macro_f1']):.4f}"
            if row["test_macro_f1"] != ""
            else "-"
        )
        lines.append(
            f"| {row['experiment_id']} | {row['phase']} | {row['task']} | "
            f"{row['model_name']} | {row['trainable_parameters']} | "
            f"{row['optimizer']} | {row['augmentation']} | "
            f"{val_f1} | {test_f1} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- Sanity rows check whether the model and training loop can learn a tiny subset.",
            "- Ablation rows are validation-only and decide model selection.",
            "- Final rows use the selected validation configuration and include test metrics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_campaign(
    config_path: Path,
    processed_dir_override: Path | None = None,
    output_root_override: Path | None = None,
    device: str = "auto",
    smoke: bool = False,
) -> Path:
    campaign = load_yaml(config_path)
    campaign_id = str(campaign["campaign_id"])
    output_root = output_root_override or Path(campaign.get("output_root", "outputs/experiments"))
    campaign_dir = output_root / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = processed_dir_override or Path(campaign.get("processed_dir", "data/processed"))
    base_config_path = Path(campaign["base_config"])
    base_config = load_yaml(base_config_path)

    write_yaml(campaign_dir / "campaign_config.yaml", campaign)

    rows: list[dict] = []
    resolved_by_id: dict[str, dict] = {}
    for experiment in campaign["experiments"]:
        experiment_id = experiment["id"]
        phase = experiment.get("phase", "ablation")
        task = experiment.get("task", "multiclass")
        resolved = deep_merge(base_config, experiment.get("overrides", {}))
        resolved_by_id[experiment_id] = resolved

        run_dir = _run_training(
            experiment_id=experiment_id,
            task=task,
            phase=phase,
            resolved_config=resolved,
            campaign_dir=campaign_dir,
            processed_dir=processed_dir,
            device=device,
            evaluate_test=bool(experiment.get("evaluate_test", False)),
            smoke=smoke,
            overfit_samples=experiment.get("overfit_samples"),
        )
        rows.append(_row_from_run(campaign_id, experiment_id, phase, run_dir))

    selected = _select_best_ablation(rows)
    selected_config = resolved_by_id[selected["experiment_id"]]

    final_runs: list[str] = []
    for task in campaign.get("final", {}).get("tasks", ["multiclass", "binary"]):
        experiment_id = f"Final_{task}"
        run_dir = _run_training(
            experiment_id=experiment_id,
            task=task,
            phase="final",
            resolved_config=selected_config,
            campaign_dir=campaign_dir,
            processed_dir=processed_dir,
            device=device,
            evaluate_test=True,
            smoke=smoke,
        )
        final_runs.append(experiment_id)
        rows.append(_row_from_run(campaign_id, experiment_id, "final", run_dir))

    summary_path = campaign_dir / "summary.csv"
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(summary_path, index=False)

    selection = {
        "campaign_id": campaign_id,
        "selected_experiment_id": selected["experiment_id"],
        "selected_by": "highest val_macro_f1, tie lowest best_val_loss",
        "best_val_macro_f1": selected["val_macro_f1"],
        "best_val_loss": selected["best_val_loss"],
        "final_runs": final_runs,
    }
    write_json(campaign_dir / "selection.json", selection)
    _write_summary_markdown(campaign_dir / "summary.md", rows, selection)

    print(f"\nCampaign complete: {campaign_dir}/")
    print(f"Selected {selection['selected_experiment_id']} for final runs.")
    return campaign_dir


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return run_campaign(
        config_path=args.config,
        processed_dir_override=args.processed_dir,
        output_root_override=args.output_root,
        device=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
