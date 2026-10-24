"""Experiment planning, synthetic smoke tests, and offline analysis."""

from .evaluate import evaluate_prediction_groups, evaluate_predictions
from .runner import run_experiment
from .synthetic_demo import run_synthetic_demo
from .train_attackers import (
    predict_grouped_random_forest_attackers_from_csv,
    train_grouped_random_forest_attackers_from_csv,
)

__all__ = [
    "evaluate_prediction_groups",
    "evaluate_predictions",
    "predict_grouped_random_forest_attackers_from_csv",
    "run_experiment",
    "run_synthetic_demo",
    "train_grouped_random_forest_attackers_from_csv",
]
