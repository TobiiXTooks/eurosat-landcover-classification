"""Maintain the comparison-ready CSV across runs and approaches.

Teammate transfer-learning results can be appended to the same CSV later using
the identical column schema, keyed by ``run_id``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = [
    "run_id",
    "task",
    "model_name",
    "approach",
    "trainable_parameters",
    "augmentation",
    "optimizer",
    "lr",
    "weight_decay",
    "dropout",
    "epochs_completed",
    "best_val_loss",
    "val_accuracy",
    "val_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "checkpoint_path",
]


def update_comparison(csv_path: Path, row: dict) -> None:
    """Insert or replace the row for ``row['run_id']`` in the comparison CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {col: row.get(col, "") for col in COLUMNS}

    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        # Ensure schema compatibility, then drop any prior entry for this run.
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[df["run_id"] != new_row["run_id"]]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df = df[COLUMNS]
    df.to_csv(csv_path, index=False)
