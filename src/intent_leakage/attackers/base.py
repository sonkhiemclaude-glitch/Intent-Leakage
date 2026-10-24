"""Shared interfaces and validation for trace attackers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class OptionalDependencyError(ImportError):
    """Raised when an explicitly optional real implementation is unavailable."""


class TraceAttacker(ABC):
    """Minimal sklearn-like interface used by experiment runners."""

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs: Any) -> "TraceAttacker":
        """Fit using training data only."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict labels for attacker-visible features."""

    def predict_proba(self, X: Any) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not provide probabilities")


def validate_feature_matrix(X: Any, *, allow_3d: bool = False) -> np.ndarray:
    """Validate an ordinary aggregate or sequence feature tensor."""

    values = np.asarray(X, dtype=float)
    allowed = (2, 3) if allow_3d else (2,)
    if values.ndim not in allowed or values.shape[0] == 0:
        dimensionality = "two- or three-dimensional" if allow_3d else "two-dimensional"
        raise ValueError(f"X must be a nonempty {dimensionality} array")
    if not np.all(np.isfinite(values)):
        raise ValueError("X must contain only finite values")
    return values


def validate_labels(y: Any, *, n_samples: int) -> np.ndarray:
    labels = np.asarray(y, dtype=object).reshape(-1)
    if labels.size != n_samples:
        raise ValueError(f"expected {n_samples} labels, got {labels.size}")
    if labels.size == 0:
        raise ValueError("at least one label is required")
    return labels


def stable_label_order(labels: np.ndarray) -> np.ndarray:
    """Return deterministic labels even if Python types are not comparable."""

    unique: list[Any] = []
    for label in labels.tolist():
        if label not in unique:
            unique.append(label)
    unique.sort(key=lambda value: (type(value).__name__, str(value)))
    return np.asarray(unique, dtype=object)
