"""Prior-only analytical attacker from paper Section 5.5."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TraceAttacker, stable_label_order


class PriorOnlyClassifier(TraceAttacker):
    """Always predict the most frequent training label.

    Ties are resolved by deterministic class-ID order.  The tie choice does not
    change prior-only accuracy, but deterministic behavior aids reproducibility.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "PriorOnlyClassifier":
        labels = np.asarray(y, dtype=object).reshape(-1)
        if labels.size == 0:
            raise ValueError("prior classifier requires at least one training label")
        classes = stable_label_order(labels)
        counts = np.asarray([np.sum(labels == label) for label in classes], dtype=int)
        self.classes_ = classes
        self.class_prior_ = counts.astype(float) / counts.sum()
        self.most_probable_class_ = classes[int(np.argmax(counts))]
        return self

    def _n_samples(self, X: Any) -> int:
        if not hasattr(self, "classes_"):
            raise RuntimeError("classifier must be fitted before prediction")
        if isinstance(X, (int, np.integer)):
            n_samples = int(X)
        else:
            try:
                n_samples = len(X)
            except TypeError as error:
                raise ValueError("X must have a length or be an integer sample count") from error
        if n_samples < 0:
            raise ValueError("sample count must be nonnegative")
        return n_samples

    def predict(self, X: Any) -> np.ndarray:
        return np.full(self._n_samples(X), self.most_probable_class_, dtype=object)

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.tile(self.class_prior_, (self._n_samples(X), 1))
