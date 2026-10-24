"""Result-driven plotting for manuscript Figures 2-4."""

from .confusion import plot_confusion_from_csv
from .generalization import plot_generalization_from_csv
from .summary import plot_metric_summary_from_csv
from .tradeoff import plot_tradeoff_from_csv

__all__ = [
    "plot_confusion_from_csv",
    "plot_generalization_from_csv",
    "plot_metric_summary_from_csv",
    "plot_tradeoff_from_csv",
]
