from __future__ import annotations

import numpy as np
import pytest

from intent_leakage.metrics import (
    class_accuracy,
    grouped_bootstrap_ci,
    grouped_bootstrap_metric,
    paired_grouped_bootstrap_difference,
)


def test_grouped_bootstrap_is_seed_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 10.0, 11.0, 20.0, 21.0])
    groups = ["p1", "p1", "p2", "p2", "p3", "p3"]
    first = grouped_bootstrap_ci(values, groups, n_resamples=200, seed=17)
    second = grouped_bootstrap_ci(values, groups, n_resamples=200, seed=17)
    assert first == second
    assert first.estimate == pytest.approx(np.mean(values))
    assert first.lower <= first.estimate <= first.upper


def test_grouped_prediction_metric() -> None:
    truth = ["a", "a", "b", "b", "c", "c"]
    prediction = ["a", "b", "b", "b", "c", "a"]
    groups = ["f1", "f1", "f2", "f2", "f3", "f3"]
    result = grouped_bootstrap_metric(
        truth,
        prediction,
        groups,
        class_accuracy,
        n_resamples=100,
        seed=3,
    )
    assert result.estimate == pytest.approx(4.0 / 6.0)
    assert 0.0 <= result.lower <= result.upper <= 1.0


def test_paired_grouped_difference_resamples_pairs() -> None:
    baseline = [1.0, 3.0, 5.0, 7.0]
    defended = [2.0, 4.0, 6.0, 8.0]
    groups = ["x", "x", "y", "y"]
    result = paired_grouped_bootstrap_difference(
        baseline, defended, groups, n_resamples=100, seed=11
    )
    assert result.estimate == pytest.approx(1.0)
    assert result.lower == pytest.approx(1.0)
    assert result.upper == pytest.approx(1.0)
    assert result.method == "paired_grouped_percentile"


def test_grouped_bootstrap_rejects_one_cluster() -> None:
    with pytest.raises(ValueError, match="at least two"):
        grouped_bootstrap_ci([1.0, 2.0], ["same", "same"], n_resamples=10)
