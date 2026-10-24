"""Confusion-matrix plot generated only from prediction records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ..orchestration.provenance import SYNTHETIC_NOTICE


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_confusion_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    true_column: str = "class_id",
    predicted_column: str = "predicted_class",
    view: str | None = "fused",
    condition: str | None = None,
    synthetic: bool = False,
) -> Path:
    """Plot a confusion matrix from actual predictions, never hard-coded values."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    rows = _read_rows(input_path)
    if view is not None and any("view" in row for row in rows):
        rows = [row for row in rows if row.get("view") == view]
    if condition is not None and any("condition" in row for row in rows):
        rows = [row for row in rows if row.get("condition") == condition]
    if not rows:
        raise ValueError("No prediction rows matched the requested view")
    synthetic = synthetic or any(
        str(row.get("synthetic", "")).lower() in {"1", "true", "yes"}
        for row in rows
    )
    missing = [name for name in (true_column, predicted_column) if name not in rows[0]]
    if missing:
        raise KeyError(f"Prediction CSV is missing columns: {missing}")
    y_true = [row[true_column] for row in rows]
    y_pred = [row[predicted_column] for row in rows]
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=90)
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xlabel("Predicted intent class")
    axis.set_ylabel("True intent class")
    title = "Intent-class confusion matrix"
    if view:
        title += f" ({view})"
    if condition:
        title += f", {condition}"
    if synthetic:
        title += f"\n{SYNTHETIC_NOTICE}"
    axis.set_title(title)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, metadata={"Description": title})
    plt.close(fig)
    return destination
