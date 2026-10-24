"""Deterministic grouped and paired bootstrap utilities for Section 5.7."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


Statistic = Callable[[np.ndarray], float]
PairedStatistic = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class BootstrapInterval:
    """Point estimate and percentile bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    seed: int
    method: str = "grouped_percentile"


def _validate_settings(confidence: float, n_resamples: int) -> tuple[float, int]:
    confidence_value = float(confidence)
    resamples = int(n_resamples)
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if resamples < 2:
        raise ValueError("n_resamples must be at least two")
    return confidence_value, resamples


def _group_indices(groups: Sequence[Hashable] | np.ndarray, n_rows: int) -> list[np.ndarray]:
    labels = list(groups)
    if len(labels) != n_rows:
        raise ValueError("groups and observations must have equal lengths")
    ordered: list[Hashable] = []
    for label in labels:
        try:
            hash(label)
        except TypeError as error:
            raise ValueError("bootstrap group labels must be hashable") from error
        if label is None or (
            isinstance(label, (float, np.floating)) and np.isnan(label)
        ):
            raise ValueError("bootstrap group labels must not be missing")
        if label not in ordered:
            ordered.append(label)
    if len(ordered) < 2:
        raise ValueError("a grouped bootstrap requires at least two unique groups")
    return [
        np.asarray([index for index, value in enumerate(labels) if value == label], dtype=int)
        for label in ordered
    ]


def _percentile_interval(
    samples: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def grouped_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    *,
    statistic: Statistic = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Percentile CI after resampling whole phrase/topology groups.

    Sampling a group more than once duplicates all rows in that group, which
    preserves within-group dependence instead of treating repetitions as IID.
    """

    observations = np.asarray(values, dtype=float).reshape(-1)
    if observations.size == 0 or not np.all(np.isfinite(observations)):
        raise ValueError("values must be a nonempty finite vector")
    confidence_value, resamples = _validate_settings(confidence, n_resamples)
    blocks = _group_indices(groups, observations.size)
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=float)
    for replicate in range(resamples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in selected])
        value = float(statistic(observations[indices]))
        if not np.isfinite(value):
            raise ValueError("statistic returned a non-finite bootstrap replicate")
        replicates[replicate] = value
    estimate = float(statistic(observations))
    if not np.isfinite(estimate):
        raise ValueError("statistic returned a non-finite point estimate")
    lower, upper = _percentile_interval(replicates, confidence=confidence_value)
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence_value,
        n_resamples=resamples,
        seed=int(seed),
    )


def grouped_bootstrap_metric(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Grouped percentile interval for a two-vector prediction metric."""

    truth = np.asarray(y_true, dtype=object).reshape(-1)
    prediction = np.asarray(y_pred, dtype=object).reshape(-1)
    if truth.size == 0 or truth.size != prediction.size:
        raise ValueError("y_true and y_pred must be nonempty and equally sized")
    confidence_value, resamples = _validate_settings(confidence, n_resamples)
    blocks = _group_indices(groups, truth.size)
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=float)
    for replicate in range(resamples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in selected])
        value = float(metric(truth[indices], prediction[indices]))
        if not np.isfinite(value):
            raise ValueError("metric returned a non-finite bootstrap replicate")
        replicates[replicate] = value
    estimate = float(metric(truth, prediction))
    if not np.isfinite(estimate):
        raise ValueError("metric returned a non-finite point estimate")
    lower, upper = _percentile_interval(replicates, confidence=confidence_value)
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence_value,
        n_resamples=resamples,
        seed=int(seed),
    )


def paired_grouped_bootstrap_difference(
    baseline: Sequence[float] | np.ndarray,
    defended: Sequence[float] | np.ndarray,
    groups: Sequence[Hashable] | np.ndarray,
    *,
    statistic: PairedStatistic | None = None,
    confidence: float = 0.95,
    n_resamples: int = 1_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Paired grouped CI for defended minus baseline performance.

    The default statistic is the mean of pairwise differences.  A custom
    statistic receives aligned baseline and defended arrays.
    """

    base = np.asarray(baseline, dtype=float).reshape(-1)
    defense = np.asarray(defended, dtype=float).reshape(-1)
    if base.size == 0 or base.size != defense.size:
        raise ValueError("paired arrays must be nonempty and equally sized")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(defense)):
        raise ValueError("paired values must be finite")
    confidence_value, resamples = _validate_settings(confidence, n_resamples)
    blocks = _group_indices(groups, base.size)
    paired_statistic = statistic or (lambda left, right: float(np.mean(right - left)))
    rng = np.random.default_rng(seed)
    replicates = np.empty(resamples, dtype=float)
    for replicate in range(resamples):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in selected])
        value = float(paired_statistic(base[indices], defense[indices]))
        if not np.isfinite(value):
            raise ValueError("statistic returned a non-finite bootstrap replicate")
        replicates[replicate] = value
    estimate = float(paired_statistic(base, defense))
    if not np.isfinite(estimate):
        raise ValueError("statistic returned a non-finite point estimate")
    lower, upper = _percentile_interval(replicates, confidence=confidence_value)
    return BootstrapInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence_value,
        n_resamples=resamples,
        seed=int(seed),
        method="paired_grouped_percentile",
    )
