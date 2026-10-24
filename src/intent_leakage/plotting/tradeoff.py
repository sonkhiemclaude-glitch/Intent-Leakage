"""Privacy/latency/bandwidth tradeoff plot from frozen-profile results."""

from __future__ import annotations

import csv
from pathlib import Path

from ..orchestration.provenance import SYNTHETIC_NOTICE


def plot_tradeoff_from_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    synthetic: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"profile", "observed_mi", "added_latency", "bandwidth_overhead"}
    if not rows:
        raise ValueError("Tradeoff results are empty")
    synthetic = synthetic or any(
        str(row.get("synthetic", "")).lower() in {"1", "true", "yes"}
        for row in rows
    )
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"Tradeoff CSV is missing columns: {sorted(missing)}")
    x = [float(row["added_latency"]) for row in rows]
    y = [float(row["observed_mi"]) for row in rows]
    sizes = [40.0 + 160.0 * max(0.0, float(row["bandwidth_overhead"])) for row in rows]

    fig, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(x, y, s=sizes, alpha=0.75, color="#C55A11")
    for row, x_value, y_value in zip(rows, x, y):
        right_half = x_value > (min(x) + max(x)) / 2
        axis.annotate(row["profile"].replace("_", " "), (x_value, y_value),
                      xytext=(-8 if right_half else 8, 8), textcoords="offset points",
                      ha="right" if right_half else "left")
    axis.margins(x=0.12, y=0.12)
    axis.set_xlabel("Simulated added duration (s)" if synthetic else "Added assurance latency")
    axis.set_ylabel("Observed leakage I(C; Ĉ), bits")
    title = "Synthetic profile tradeoff" if synthetic else "Frozen shaping-profile tradeoff"
    if synthetic:
        title += f"\n{SYNTHETIC_NOTICE}"
    axis.set_title(title)
    fig.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, metadata={"Description": title})
    plt.close(fig)
    return destination
