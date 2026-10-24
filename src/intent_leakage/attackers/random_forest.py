"""Random-forest aggregate/burst attacker from paper Section 5.6."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TraceAttacker, validate_feature_matrix, validate_labels


class RandomForestTraceAttacker(TraceAttacker):
    """Thin validated wrapper around scikit-learn's random forest.

    Hyperparameters are explicit engineering configuration and are never tuned
    on the test partition by this class.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | dict[Any, float] | None = None,
        random_state: int = 0,
        n_jobs: int | None = 1,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.random_state = int(random_state)
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "RandomForestTraceAttacker":
        from sklearn.ensemble import RandomForestClassifier

        features = validate_feature_matrix(X)
        labels = validate_labels(y, n_samples=features.shape[0])
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(features, labels.tolist())
        self.classes_ = np.asarray(self.model_.classes_, dtype=object)
        self.n_features_in_ = features.shape[1]
        return self

    def _validate_predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("attacker must be fitted before prediction")
        features = validate_feature_matrix(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {features.shape[1]}"
            )
        return features

    def predict(self, X: Any) -> np.ndarray:
        features = self._validate_predict(X)
        return np.asarray(self.model_.predict(features), dtype=object)

    def predict_proba(self, X: Any) -> np.ndarray:
        features = self._validate_predict(X)
        return np.asarray(self.model_.predict_proba(features), dtype=float)
