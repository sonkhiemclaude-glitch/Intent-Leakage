"""Generalization plot from E2-compatible CSV results."""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import fill

from ..orchestration.provenance import SYNTHETIC_NOTICE


def plot_generalization_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    condition_column: str = "held_out_condition",
    metric_column: str = "accuracy",
    synthetic: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Generalization results are empty")
    synthetic = synthetic or any(
        str(row.get("synthetic", "")).lower() in {"1", "true", "yes"}
        for row in rows
    )
    for column in (condition_column, metric_column):
        if column not in rows[0]:
            raise KeyError(f"Missing generalization column: {column}")
    labels = [fill(row[condition_column].replace("_", " "), width=20) for row in rows]
    values = [float(row[metric_column]) for row in rows]
    fig, axis = plt.subplots(figsize=(max(8, 1.8 * len(rows)), 5.5))
    axis.bar(labels, values, color="#4472C4")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel(metric_column.replace("_", " ").title())
    axis.set_xlabel("Synthetic condition" if synthetic else "Held-out condition")
    title = "Synthetic held-out evaluation" if synthetic else "Grouped held-out generalization"
    if synthetic:
        title += f"\n{SYNTHETIC_NOTICE}"
    axis.set_title(title)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, metadata={"Description": title})
    plt.close(fig)
    return destination
