"""Simple grouping of result rows for declared ablation dimensions."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..orchestration.provenance import SYNTHETIC_NOTICE
from ..utils.validation import validate_run_kind_values


def summarize_ablation(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group_columns: Sequence[str] = (
        "protocol_block",
        "topology_id",
        "model_id",
        "view",
        "condition",
        "attacker",
    ),
    required_group_columns: Sequence[str] = ("view", "condition", "attacker"),
    metric_columns: Sequence[str] = ("accuracy", "macro_f1"),
    synthetic: bool,
) -> Path:
    """Create a stratified descriptive ablation summary from metric rows."""

    import pandas as pd

    frame = pd.read_csv(input_path)
    missing = (
        set(required_group_columns) | set(metric_columns) | {"synthetic"}
    ) - set(frame.columns)
    if missing:
        raise KeyError(f"Ablation input is missing columns: {sorted(missing)}")
    validate_run_kind_values(
        frame["synthetic"].tolist(),
        expected_synthetic=synthetic,
        context="ablation input",
    )
    available_groups = [column for column in group_columns if column in frame]
    summary = frame.groupby(available_groups, dropna=False)[list(metric_columns)].agg(
        ["mean", "count"]
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = summary.reset_index()
    result["synthetic"] = synthetic
    result["run_kind"] = (
        "synthetic_smoke_test" if synthetic else "real_offline_analysis"
    )
    result["notice"] = (
        SYNTHETIC_NOTICE
        if synthetic
        else "real-data analysis; inspect frozen capture and run provenance"
    )
    result["paper_results_claimed"] = False
    result.to_csv(destination, index=False)
    return destination
