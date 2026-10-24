"""Operational cost and overflow metrics from paper Section 5.7."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Overhead:
    """A paired absolute change and relative overhead fraction."""

    absolute: float
    relative: float


@dataclass(frozen=True)
class OverflowRates:
    """Aggregate and stratified envelope-overflow rates."""

    overall: float
    by_class: Mapping[Hashable, float]
    by_scope: Mapping[Hashable, float]
    counts_by_class: Mapping[Hashable, int]
    counts_by_scope: Mapping[Hashable, int]


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def latency_cost(baseline_latency: float, defended_latency: float) -> Overhead:
    """Return ``L_d-L_0`` and ``(L_d-L_0)/L_0`` from Section 5.7.

    Values are paired run-level measurements.  The baseline must be positive;
    the function never silently turns an undefined ratio into zero or infinity.
    """

    baseline = _finite_nonnegative(baseline_latency, name="baseline latency")
    defended = _finite_nonnegative(defended_latency, name="defended latency")
    if baseline == 0.0:
        raise ZeroDivisionError("relative latency overhead requires positive baseline latency")
    added = defended - baseline
    return Overhead(absolute=added, relative=added / baseline)


def bandwidth_overhead(baseline_bytes: float, defended_bytes: float) -> float:
    """Return relative on-wire byte overhead ``(B_d-B_0)/B_0``."""

    baseline = _finite_nonnegative(baseline_bytes, name="baseline bytes")
    defended = _finite_nonnegative(defended_bytes, name="defended bytes")
    if baseline == 0.0:
        raise ZeroDivisionError("relative bandwidth overhead requires positive baseline bytes")
    return (defended - baseline) / baseline


def paired_latency_costs(
    baseline_latency: Sequence[float] | np.ndarray,
    defended_latency: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized paired latency changes and ratios without aggregate-ratio ambiguity."""

    baseline = np.asarray(baseline_latency, dtype=float).reshape(-1)
    defended = np.asarray(defended_latency, dtype=float).reshape(-1)
    if baseline.size == 0 or baseline.size != defended.size:
        raise ValueError("paired latency arrays must be nonempty and have equal lengths")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(defended)):
        raise ValueError("latency arrays must be finite")
    if np.any(baseline <= 0.0) or np.any(defended < 0.0):
        raise ValueError("baseline latencies must be positive and defended latencies nonnegative")
    added = defended - baseline
    return added, added / baseline


def paired_bandwidth_overheads(
    baseline_bytes: Sequence[float] | np.ndarray,
    defended_bytes: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Vectorized paired byte-overhead fractions."""

    baseline = np.asarray(baseline_bytes, dtype=float).reshape(-1)
    defended = np.asarray(defended_bytes, dtype=float).reshape(-1)
    if baseline.size == 0 or baseline.size != defended.size:
        raise ValueError("paired byte arrays must be nonempty and have equal lengths")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(defended)):
        raise ValueError("byte arrays must be finite")
    if np.any(baseline <= 0.0) or np.any(defended < 0.0):
        raise ValueError("baseline bytes must be positive and defended bytes nonnegative")
    return (defended - baseline) / baseline


def _stratified_rates(
    overflow: np.ndarray,
    strata: Sequence[Any] | np.ndarray | None,
    *,
    name: str,
) -> tuple[dict[Hashable, float], dict[Hashable, int]]:
    if strata is None:
        return {}, {}
    values = np.asarray(strata, dtype=object).reshape(-1)
    if values.size != overflow.size:
        raise ValueError(f"{name} labels and overflow indicators must have equal lengths")
    order: list[Hashable] = []
    for value in values.tolist():
        if value not in order:
            order.append(value)
    rates: dict[Hashable, float] = {}
    counts: dict[Hashable, int] = {}
    for value in order:
        mask = values == value
        rates[value] = float(np.mean(overflow[mask]))
        counts[value] = int(np.sum(mask))
    return rates, counts


def overflow_rates(
    overflow: Sequence[bool | int] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray | None = None,
    scopes: Sequence[Any] | np.ndarray | None = None,
) -> OverflowRates:
    """Report overall, class-wise and scope-wise envelope overflow.

    Inputs use ``True``/1 for overflow, equivalently ``E=0`` in Section 4.6.
    Empty strata are not manufactured and therefore cannot be mistaken for a
    measured zero rate.
    """

    raw = np.asarray(overflow).reshape(-1)
    if raw.size == 0:
        raise ValueError("overflow rates require at least one run")
    if not all(value in (False, True, 0, 1) for value in raw.tolist()):
        raise ValueError("overflow indicators must be boolean or 0/1")
    indicator = raw.astype(float)
    by_class, class_counts = _stratified_rates(indicator, classes, name="class")
    by_scope, scope_counts = _stratified_rates(indicator, scopes, name="scope")
    return OverflowRates(
        overall=float(np.mean(indicator)),
        by_class=by_class,
        by_scope=by_scope,
        counts_by_class=class_counts,
        counts_by_scope=scope_counts,
    )


def outcome_counts(outcomes: Sequence[str] | np.ndarray) -> dict[str, int]:
    """Count success, failure and timeout outcomes without relabeling privacy."""

    values = np.asarray(outcomes, dtype=object).reshape(-1)
    if values.size == 0:
        raise ValueError("outcome counts require at least one run")
    return dict(Counter(str(value) for value in values.tolist()))
