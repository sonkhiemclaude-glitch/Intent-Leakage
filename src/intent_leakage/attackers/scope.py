"""Scope attacks, including the paper's predicted-class two-stage model."""

from __future__ import annotations

import copy
from typing import Any, Literal

import numpy as np

from .base import validate_feature_matrix, validate_labels


def _clone_estimator(estimator: Any) -> Any:
    try:
        from sklearn.base import clone

        return clone(estimator)
    except (TypeError, AttributeError):
        return copy.deepcopy(estimator)


class TwoStagePredictedClassScopeModel:
    """Predict intent class, then use that predicted class's scope estimator.

    Class-specific scope estimators are trained using true training classes,
    while inference accepts only ``X`` and therefore cannot accidentally use a
    true test class.  This implements the joint-attack rule in Sections 3.5 and
    5.6.
    """

    def __init__(
        self,
        class_estimator: Any,
        scope_estimator: Any | None = None,
        *,
        task: Literal["regression", "classification"] = "regression",
        random_state: int = 0,
    ) -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        self.class_estimator = class_estimator
        self.scope_estimator = scope_estimator
        self.task = task
        self.random_state = int(random_state)

    def _default_scope_estimator(self) -> Any:
        if self.task == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=1
            )
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100, random_state=self.random_state, n_jobs=1
        )

    def fit(
        self,
        X: Any,
        intent_classes: Any,
        scopes: Any,
    ) -> "TwoStagePredictedClassScopeModel":
        features = validate_feature_matrix(X)
        class_labels = validate_labels(intent_classes, n_samples=features.shape[0])
        if self.task == "regression":
            scope_values = np.asarray(scopes, dtype=float).reshape(-1)
            if scope_values.size != features.shape[0] or not np.all(np.isfinite(scope_values)):
                raise ValueError("numeric scopes must be aligned and finite")
        else:
            scope_values = validate_labels(scopes, n_samples=features.shape[0])

        self.class_estimator_ = _clone_estimator(self.class_estimator)
        self.class_estimator_.fit(features, class_labels.tolist())
        prototype = (
            self.scope_estimator
            if self.scope_estimator is not None
            else self._default_scope_estimator()
        )
        self.global_scope_estimator_ = _clone_estimator(prototype)
        self.global_scope_estimator_.fit(features, scope_values.tolist())

        self.scope_estimators_: dict[Any, Any] = {}
        ordered_classes: list[Any] = []
        for label in class_labels.tolist():
            if label not in ordered_classes:
                ordered_classes.append(label)
        for label in ordered_classes:
            mask = class_labels == label
            estimator = _clone_estimator(prototype)
            estimator.fit(features[mask], scope_values[mask].tolist())
            self.scope_estimators_[label] = estimator
        self.n_features_in_ = features.shape[1]
        return self

    def _features(self, X: Any) -> np.ndarray:
        if not hasattr(self, "class_estimator_"):
            raise RuntimeError("scope model must be fitted before prediction")
        features = validate_feature_matrix(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {features.shape[1]}"
            )
        return features

    def predict_with_classes(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        features = self._features(X)
        predicted_classes = np.asarray(
            self.class_estimator_.predict(features), dtype=object
        ).reshape(-1)
        if predicted_classes.size != features.shape[0]:
            raise RuntimeError("class estimator returned the wrong number of predictions")
        if self.task == "regression":
            scope_predictions = np.empty(features.shape[0], dtype=float)
        else:
            scope_predictions = np.empty(features.shape[0], dtype=object)
        for label in set(predicted_classes.tolist()):
            mask = predicted_classes == label
            estimator = self.scope_estimators_.get(label, self.global_scope_estimator_)
            prediction = np.asarray(estimator.predict(features[mask])).reshape(-1)
            scope_predictions[mask] = prediction
        return predicted_classes, scope_predictions

    def predict(self, X: Any) -> np.ndarray:
        """Return scope predictions made from predicted, never true, classes."""

        return self.predict_with_classes(X)[1]


TwoStageScopeAttacker = TwoStagePredictedClassScopeModel
