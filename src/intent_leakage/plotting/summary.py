"""Compact metric summary for synthetic or real E1 result CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

from ..orchestration.provenance import SYNTHETIC_NOTICE


def plot_metric_summary_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    metric_column: str = "accuracy",
    synthetic: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Metric results are empty")
    if "view" not in rows[0] or metric_column not in rows[0]:
        raise KeyError("Metric CSV requires view and the selected metric column")
    synthetic = synthetic or any(
        str(row.get("synthetic", "")).lower() in {"1", "true", "yes"}
        for row in rows
    )
    labels = [
        f"{row.get('condition')}: {row['view']}"
        if row.get("condition")
        else row["view"]
        for row in rows
    ]
    values = [float(row[metric_column]) for row in rows]
    fig, axis = plt.subplots(figsize=(max(7, len(labels) * 0.65), 4.5))
    axis.bar(labels, values, color="#4472C4")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Observer view")
    axis.set_ylabel(metric_column.replace("_", " ").title())
    title = "Held-out attack metric by observer view"
    if synthetic:
        title += f"\n{SYNTHETIC_NOTICE}"
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, metadata={"Description": title})
    plt.close(fig)
    return destination
