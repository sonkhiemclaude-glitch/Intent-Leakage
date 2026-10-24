from __future__ import annotations

import numpy as np
import pytest

from intent_leakage.attackers import (
    CNN1DTraceAttacker,
    PriorOnlyClassifier,
    RandomForestTraceAttacker,
    SequenceKNNTraceAttacker,
    TwoStagePredictedClassScopeModel,
    sequence_distance,
)


def test_prior_only_classifier_is_deterministic_on_tie() -> None:
    attacker = PriorOnlyClassifier().fit(None, ["I02", "I01", "I02", "I01"])
    np.testing.assert_array_equal(attacker.predict(np.zeros((3, 1))), ["I01"] * 3)
    np.testing.assert_allclose(attacker.predict_proba(2), [[0.5, 0.5], [0.5, 0.5]])


def test_random_forest_reference_attacker() -> None:
    X = np.asarray([[-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0]])
    y = ["left", "left", "left", "right", "right", "right"]
    first = RandomForestTraceAttacker(n_estimators=40, random_state=9).fit(X, y)
    second = RandomForestTraceAttacker(n_estimators=40, random_state=9).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert np.mean(first.predict(X) == np.asarray(y, dtype=object)) >= 5.0 / 6.0
    np.testing.assert_allclose(first.predict_proba(X).sum(axis=1), 1.0)


def test_random_forest_unfitted_error_is_explicit() -> None:
    attacker = RandomForestTraceAttacker(n_estimators=5)
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        attacker.predict([[0.0]])
    with pytest.raises(RuntimeError, match="fitted before prediction"):
        attacker.predict_proba([[0.0]])


def test_sequence_distances_and_knn() -> None:
    left = np.asarray([[0.0, 0.0], [1.0, 0.1]])
    right = np.asarray([[0.0, 0.0], [2.0, 0.1]])
    assert sequence_distance(left, left, metric="dtw") == 0.0
    assert sequence_distance(left, right, metric="euclidean") == pytest.approx(1.0)
    assert sequence_distance(left, right, metric="manhattan") == pytest.approx(1.0)

    X = np.asarray(
        [
            [[-2.0, 0.0], [-1.8, 0.1]],
            [[-1.9, 0.0], [-2.1, 0.1]],
            [[2.0, 0.0], [1.8, 0.1]],
            [[1.9, 0.0], [2.1, 0.1]],
        ]
    )
    y = ["a", "a", "b", "b"]
    attacker = SequenceKNNTraceAttacker(n_neighbors=1, distance="dtw").fit(X, y)
    np.testing.assert_array_equal(attacker.predict(X), y)
    np.testing.assert_allclose(attacker.predict_proba(X).sum(axis=1), 1.0)


def test_knn_distance_selection_uses_validation_data() -> None:
    X = np.asarray([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
    y = ["a", "a", "b", "b"]
    attacker, scores = SequenceKNNTraceAttacker.select_distance_on_validation(
        X,
        y,
        X,
        y,
        candidates=("euclidean", "manhattan"),
        n_neighbors=1,
    )
    assert attacker.distance == "euclidean"  # deterministic first-candidate tie
    assert scores == {"euclidean": 1.0, "manhattan": 1.0}


class AlwaysBClassifier:
    def fit(self, X: np.ndarray, y: list[str]) -> "AlwaysBClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), "b", dtype=object)


class MeanScopeEstimator:
    def fit(self, X: np.ndarray, y: list[float]) -> "MeanScopeEstimator":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mean_)


def test_two_stage_scope_uses_predicted_not_true_class() -> None:
    X = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
    classes = ["a", "a", "b", "b"]
    scopes = [1.0, 1.0, 10.0, 10.0]
    model = TwoStagePredictedClassScopeModel(
        AlwaysBClassifier(), MeanScopeEstimator(), task="regression"
    ).fit(X, classes, scopes)
    predicted_classes, predicted_scope = model.predict_with_classes([[-100.0], [100.0]])
    np.testing.assert_array_equal(predicted_classes, ["b", "b"])
    np.testing.assert_allclose(predicted_scope, [10.0, 10.0])


def test_real_optional_cnn_smoke() -> None:
    pytest.importorskip("torch")
    X = np.asarray(
        [
            [[-2.0, 0.0], [-2.0, 0.1], [-1.5, 0.1]],
            [[-1.8, 0.0], [-2.2, 0.1], [-1.7, 0.1]],
            [[-2.1, 0.0], [-1.9, 0.1], [-1.6, 0.1]],
            [[2.0, 0.0], [2.0, 0.1], [1.5, 0.1]],
            [[1.8, 0.0], [2.2, 0.1], [1.7, 0.1]],
            [[2.1, 0.0], [1.9, 0.1], [1.6, 0.1]],
        ],
        dtype=float,
    )
    y = ["a", "a", "a", "b", "b", "b"]
    model = CNN1DTraceAttacker(
        hidden_channels=(4, 4),
        epochs=2,
        batch_size=3,
        patience=None,
        random_state=7,
    ).fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert len(model.training_history_) == 2
