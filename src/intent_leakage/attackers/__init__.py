"""Reference passive attackers from paper Section 5.6."""

from .base import OptionalDependencyError, TraceAttacker
from .cnn import CNN1DTraceAttacker, CNNTraceAttacker
from .prior import PriorOnlyClassifier
from .random_forest import RandomForestTraceAttacker
from .scope import TwoStagePredictedClassScopeModel, TwoStageScopeAttacker
from .sequence_knn import SequenceKNNTraceAttacker, sequence_distance

__all__ = [
    "CNN1DTraceAttacker",
    "CNNTraceAttacker",
    "OptionalDependencyError",
    "PriorOnlyClassifier",
    "RandomForestTraceAttacker",
    "SequenceKNNTraceAttacker",
    "TraceAttacker",
    "TwoStagePredictedClassScopeModel",
    "TwoStageScopeAttacker",
    "sequence_distance",
]
