"""Empirical leakage and prediction metrics from Section 3.5 of the paper.

The mutual-information function in this module measures the observed leakage
``I(C; C_hat)`` of a particular classifier.  It is deliberately not named or
reported as ``I(C; O)``: a finite-sample confusion-matrix plug-in value is not
a formal lower confidence bound on the population leakage of the trace.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np


Label = Hashable


def _paired_arrays(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=object).reshape(-1)
    prediction = np.asarray(y_pred, dtype=object).reshape(-1)
    if truth.size == 0:
        raise ValueError("metrics require at least one observation")
    if truth.size != prediction.size:
        raise ValueError(
            f"y_true and y_pred must have equal lengths, got "
            f"{truth.size} and {prediction.size}"
        )
    return truth, prediction


def _ordered_union(*arrays: np.ndarray) -> list[Label]:
    labels: list[Label] = []
    for array in arrays:
        for value in array.tolist():
            if value not in labels:
                labels.append(value)
    return labels


def class_accuracy(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
) -> float:
    """Return intent-class accuracy from paper Equation (2)."""

    truth, prediction = _paired_arrays(y_true, y_pred)
    return float(np.mean(truth == prediction))


def scope_mae(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """Return numeric scope mean absolute error from paper Equation (3)."""

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.size == 0:
        raise ValueError("scope MAE requires at least one observation")
    if truth.size != prediction.size:
        raise ValueError(
            f"y_true and y_pred must have equal lengths, got "
            f"{truth.size} and {prediction.size}"
        )
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(prediction)):
        raise ValueError("scope values must all be finite")
    return float(np.mean(np.abs(prediction - truth)))


def confusion_matrix_counts(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Iterable[Label] | None = None,
) -> tuple[np.ndarray, tuple[Label, ...]]:
    """Build a confusion count matrix without silently dropping labels."""

    truth, prediction = _paired_arrays(y_true, y_pred)
    ordered = list(labels) if labels is not None else _ordered_union(truth, prediction)
    if not ordered:
        raise ValueError("labels must not be empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError("labels must be unique")
    unknown = [value for value in _ordered_union(truth, prediction) if value not in ordered]
    if unknown:
        raise ValueError(f"observed labels missing from declared label set: {unknown!r}")
    index = {label: position for position, label in enumerate(ordered)}
    matrix = np.zeros((len(ordered), len(ordered)), dtype=np.int64)
    for expected, observed in zip(truth.tolist(), prediction.tolist()):
        matrix[index[expected], index[observed]] += 1
    return matrix, tuple(ordered)


def macro_f1(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Iterable[Label] | None = None,
    zero_division: float = 0.0,
) -> float:
    """Return unweighted per-class F1, including every declared label.

    ``zero_division`` is used when a class has neither predicted nor true
    positives.  The paper reports macro-F1 to expose class imbalance; using a
    fixed declared label set ensures an absent class is not silently removed.
    """

    if not np.isfinite(zero_division):
        raise ValueError("zero_division must be finite")
    matrix, _ = confusion_matrix_counts(y_true, y_pred, labels=labels)
    scores: list[float] = []
    for idx in range(matrix.shape[0]):
        true_positive = float(matrix[idx, idx])
        false_positive = float(matrix[:, idx].sum() - matrix[idx, idx])
        false_negative = float(matrix[idx, :].sum() - matrix[idx, idx])
        denominator = 2.0 * true_positive + false_positive + false_negative
        scores.append(
            float(zero_division) if denominator == 0.0 else 2.0 * true_positive / denominator
        )
    return float(np.mean(scores))


def _normalised_prior(
    prior: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        list(prior.values()) if isinstance(prior, Mapping) else prior,
        dtype=float,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("prior must contain at least one class")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("prior probabilities must be finite and nonnegative")
    if not np.isclose(values.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"prior probabilities must sum to one, got {values.sum():.16g}")
    return values


def prior_only_accuracy(
    prior: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> float:
    """Return the most-probable-class baseline ``max_k pi_k``."""

    return float(np.max(_normalised_prior(prior)))


def empirical_class_prior(
    labels: Sequence[Any] | np.ndarray,
    *,
    class_order: Iterable[Label] | None = None,
) -> dict[Label, float]:
    """Estimate a class prior for an explicitly declared evaluation subset."""

    values = np.asarray(labels, dtype=object).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot estimate a prior from an empty sample")
    ordered = list(class_order) if class_order is not None else _ordered_union(values)
    if len(set(ordered)) != len(ordered):
        raise ValueError("class_order must contain unique labels")
    unknown = [value for value in _ordered_union(values) if value not in ordered]
    if unknown:
        raise ValueError(f"observed classes missing from class_order: {unknown!r}")
    return {label: float(np.mean(values == label)) for label in ordered}


def advantage_over_prior(
    accuracy: float,
    prior: Mapping[Label, float] | Sequence[float] | np.ndarray | float,
) -> float:
    """Return absolute advantage ``A_C - A_0`` from Section 3.5."""

    accuracy_value = float(accuracy)
    if not np.isfinite(accuracy_value) or not 0.0 <= accuracy_value <= 1.0:
        raise ValueError("accuracy must be finite and in [0, 1]")
    if np.isscalar(prior):
        baseline = float(prior)
        if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
            raise ValueError("prior-only accuracy must be finite and in [0, 1]")
    else:
        baseline = prior_only_accuracy(prior)
    return accuracy_value - baseline


def confusion_mutual_information_from_matrix(matrix: np.ndarray) -> float:
    """Compute the plug-in ``I(C; C_hat)`` in bits from confusion counts.

    This is the classifier-dependent observed leakage described in Sections
    3.5 and 5.7.  Zero cells contribute zero and no smoothing is applied.
    """

    counts = np.asarray(matrix, dtype=float)
    if counts.ndim != 2:
        raise ValueError("confusion matrix must be two-dimensional")
    if counts.shape[0] == 0 or counts.shape[1] == 0:
        raise ValueError("confusion matrix must have at least one row and column")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("confusion counts must be finite and nonnegative")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("confusion matrix must contain at least one observation")
    joint = counts / total
    truth_marginal = joint.sum(axis=1, keepdims=True)
    prediction_marginal = joint.sum(axis=0, keepdims=True)
    independent = truth_marginal @ prediction_marginal
    positive = joint > 0.0
    information = np.sum(joint[positive] * np.log2(joint[positive] / independent[positive]))
    # A coherent contingency table is nonnegative analytically.  Remove only
    # roundoff, never a material negative value.
    if information < -1e-12:
        raise ArithmeticError(f"computed negative mutual information: {information}")
    return float(max(0.0, information))


def confusion_mutual_information(
    y_true: Sequence[Any] | np.ndarray,
    y_pred: Sequence[Any] | np.ndarray,
    *,
    labels: Iterable[Label] | None = None,
) -> float:
    """Compute held-out classifier-observed mutual information in bits."""

    matrix, _ = confusion_matrix_counts(y_true, y_pred, labels=labels)
    return confusion_mutual_information_from_matrix(matrix)
