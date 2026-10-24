"""Sequence-neighbour attacks with validation-selected distance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from .base import TraceAttacker, stable_label_order, validate_labels


DistanceName = Literal["euclidean", "manhattan", "dtw"]


def _as_sequence_dataset(X: Any) -> list[np.ndarray]:
    if isinstance(X, np.ndarray) and X.dtype != object:
        values = np.asarray(X, dtype=float)
        if values.ndim == 3:
            sequences = [values[index] for index in range(values.shape[0])]
        elif values.ndim == 2:
            # Rows are samples with one scalar feature per sequence step.
            sequences = [values[index, :, None] for index in range(values.shape[0])]
        else:
            raise ValueError("sequence X must have shape (n,length) or (n,length,features)")
    else:
        try:
            raw = list(X)
        except TypeError as error:
            raise ValueError("sequence X must be an array or iterable of arrays") from error
        sequences = []
        for item in raw:
            array = np.asarray(item, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2:
                raise ValueError("every sequence must be one- or two-dimensional")
            sequences.append(array)
    if not sequences:
        raise ValueError("at least one sequence is required")
    feature_count = sequences[0].shape[1]
    if feature_count == 0 or any(sequence.shape[0] == 0 for sequence in sequences):
        raise ValueError("sequences must have at least one step and one feature")
    if any(sequence.shape[1] != feature_count for sequence in sequences):
        raise ValueError("all sequences must have the same feature count")
    if any(not np.all(np.isfinite(sequence)) for sequence in sequences):
        raise ValueError("sequences must contain only finite values")
    return sequences


def _pad_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = max(left.shape[0], right.shape[0])
    padded_left = np.zeros((length, left.shape[1]), dtype=float)
    padded_right = np.zeros((length, right.shape[1]), dtype=float)
    padded_left[: left.shape[0]] = left
    padded_right[: right.shape[0]] = right
    return padded_left, padded_right


def sequence_distance(left: np.ndarray, right: np.ndarray, *, metric: DistanceName) -> float:
    """Compute one configured sequence distance.

    Euclidean and Manhattan distances use right-zero padding for unequal
    lengths.  DTW uses Euclidean local cost and normalizes by path length.  The
    paper leaves the candidate distance unspecified, so the chosen name must be
    captured in run provenance.
    """

    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.ndim == 1:
        first = first[:, None]
    if second.ndim == 1:
        second = second[:, None]
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("sequence pair must be 2D with matching feature counts")
    if first.shape[0] == 0 or second.shape[0] == 0:
        raise ValueError("sequence pair must be nonempty")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("sequence values must be finite")
    if metric in {"euclidean", "manhattan"}:
        padded_first, padded_second = _pad_pair(first, second)
        difference = padded_first - padded_second
        if metric == "euclidean":
            return float(np.linalg.norm(difference))
        return float(np.sum(np.abs(difference)))
    if metric != "dtw":
        raise ValueError(f"unsupported sequence distance: {metric!r}")

    rows, columns = first.shape[0], second.shape[0]
    costs = np.full((rows + 1, columns + 1), np.inf, dtype=float)
    path_lengths = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    costs[0, 0] = 0.0
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                (costs[row - 1, column], path_lengths[row - 1, column]),
                (costs[row, column - 1], path_lengths[row, column - 1]),
                (costs[row - 1, column - 1], path_lengths[row - 1, column - 1]),
            )
            predecessor_cost, predecessor_length = min(candidates, key=lambda item: item[0])
            local = float(np.linalg.norm(first[row - 1] - second[column - 1]))
            costs[row, column] = predecessor_cost + local
            path_lengths[row, column] = predecessor_length + 1
    return float(costs[rows, columns] / path_lengths[rows, columns])


class SequenceKNNTraceAttacker(TraceAttacker):
    """Exact k-nearest-neighbour attack over traffic sequences."""

    def __init__(
        self,
        *,
        n_neighbors: int = 3,
        distance: DistanceName = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if distance not in {"euclidean", "manhattan", "dtw"}:
            raise ValueError("distance must be euclidean, manhattan, or dtw")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be uniform or distance")
        self.n_neighbors = int(n_neighbors)
        self.distance = distance
        self.weights = weights

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "SequenceKNNTraceAttacker":
        sequences = _as_sequence_dataset(X)
        labels = validate_labels(y, n_samples=len(sequences))
        if self.n_neighbors > len(sequences):
            raise ValueError("n_neighbors cannot exceed the training sample count")
        self.sequences_ = [sequence.copy() for sequence in sequences]
        self.labels_ = labels.copy()
        self.classes_ = stable_label_order(labels)
        self.n_features_in_ = sequences[0].shape[1]
        return self

    def _distances(self, sequence: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                sequence_distance(sequence, training, metric=self.distance)
                for training in self.sequences_
            ],
            dtype=float,
        )

    def _vote(self, distances: np.ndarray) -> np.ndarray:
        order = np.argsort(distances, kind="stable")[: self.n_neighbors]
        selected_distances = distances[order]
        selected_labels = self.labels_[order]
        if self.weights == "distance" and np.any(selected_distances == 0.0):
            zero = selected_distances == 0.0
            selected_labels = selected_labels[zero]
            selected_distances = selected_distances[zero]
        probabilities = np.zeros(self.classes_.size, dtype=float)
        if self.weights == "uniform":
            weights = np.ones(selected_labels.size, dtype=float)
        else:
            weights = (
                np.ones(selected_labels.size, dtype=float)
                if np.all(selected_distances == 0.0)
                else 1.0 / selected_distances
            )
        for label, weight in zip(selected_labels.tolist(), weights.tolist()):
            index = int(np.flatnonzero(self.classes_ == label)[0])
            probabilities[index] += weight
        probabilities /= probabilities.sum()
        return probabilities

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "sequences_"):
            raise RuntimeError("attacker must be fitted before prediction")
        sequences = _as_sequence_dataset(X)
        if any(sequence.shape[1] != self.n_features_in_ for sequence in sequences):
            raise ValueError(f"expected {self.n_features_in_} features per sequence step")
        return np.stack([self._vote(self._distances(sequence)) for sequence in sequences])

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    @classmethod
    def select_distance_on_validation(
        cls,
        X_train: Any,
        y_train: Any,
        X_validation: Any,
        y_validation: Any,
        *,
        candidates: Sequence[DistanceName] = ("euclidean", "manhattan", "dtw"),
        n_neighbors: int = 3,
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> tuple["SequenceKNNTraceAttacker", dict[str, float]]:
        """Select the sequence distance on validation data only."""

        if not candidates:
            raise ValueError("at least one candidate distance is required")
        validation_labels = np.asarray(y_validation, dtype=object).reshape(-1)
        scores: dict[str, float] = {}
        fitted: dict[str, SequenceKNNTraceAttacker] = {}
        for candidate in candidates:
            attacker = cls(
                n_neighbors=n_neighbors,
                distance=candidate,
                weights=weights,
            ).fit(X_train, y_train)
            prediction = attacker.predict(X_validation)
            if prediction.size != validation_labels.size or prediction.size == 0:
                raise ValueError("validation features and labels must be nonempty and aligned")
            scores[candidate] = float(np.mean(prediction == validation_labels))
            fitted[candidate] = attacker
        # Candidate order deterministically resolves tied validation scores.
        best = max(candidates, key=lambda candidate: scores[candidate])
        return fitted[best], scores
