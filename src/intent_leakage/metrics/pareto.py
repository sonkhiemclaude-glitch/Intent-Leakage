"""Nondominated privacy-cost operating points for paper experiment E4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


DEFAULT_OBJECTIVES = ("leakage", "added_latency", "bandwidth_overhead")


def pareto_mask(values: np.ndarray, *, minimize: bool | Sequence[bool] = True) -> np.ndarray:
    """Return a mask for weakly nondominated rows.

    Duplicate points are all retained.  A point dominates another only when it
    is no worse on every objective and strictly better on at least one.
    """

    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("values must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(points)):
        raise ValueError("Pareto objectives must all be finite")
    if isinstance(minimize, bool):
        directions = np.full(points.shape[1], minimize, dtype=bool)
    else:
        directions = np.asarray(list(minimize), dtype=bool)
        if directions.size != points.shape[1]:
            raise ValueError("minimize must have one entry per objective")
    canonical = points.copy()
    canonical[:, ~directions] *= -1.0
    keep = np.ones(points.shape[0], dtype=bool)
    for index, point in enumerate(canonical):
        no_worse = np.all(canonical <= point, axis=1)
        strictly_better = np.any(canonical < point, axis=1)
        if np.any(no_worse & strictly_better):
            keep[index] = False
    return keep


def nondominated_profiles(
    records: Sequence[Mapping[str, Any]] | Any,
    *,
    objectives: Sequence[str] = DEFAULT_OBJECTIVES,
    minimize: bool | Sequence[bool] = True,
) -> Any:
    """Filter records to the nondominated E4 operating points.

    Pandas ``DataFrame`` inputs are returned as a reset-index DataFrame.
    Sequence-of-mapping inputs are returned as a list of shallow dictionaries.
    """

    objective_names = tuple(objectives)
    if not objective_names:
        raise ValueError("at least one objective is required")
    if hasattr(records, "loc") and hasattr(records, "columns"):
        missing = set(objective_names) - set(records.columns)
        if missing:
            raise ValueError(f"missing Pareto columns: {sorted(missing)}")
        values = records.loc[:, objective_names].to_numpy(dtype=float)
        return records.loc[pareto_mask(values, minimize=minimize)].reset_index(drop=True)
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("records must not be empty")
    try:
        values = np.asarray(
            [[row[name] for name in objective_names] for row in rows], dtype=float
        )
    except KeyError as error:
        raise ValueError(f"missing Pareto objective: {error.args[0]}") from error
    mask = pareto_mask(values, minimize=minimize)
    return [row for row, keep in zip(rows, mask.tolist()) if keep]
