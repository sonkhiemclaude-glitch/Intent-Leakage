"""Offline calculation of attack metrics from prediction records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..orchestration.provenance import SYNTHETIC_NOTICE
from ..schemas import IntentClass
from ..utils.io import atomic_write_json
from ..utils.validation import validate_run_kind_values


_VALID_CLASS_IDS = frozenset(value.value for value in IntentClass)


def _require_paper_class_ids(values: list[str], *, name: str) -> None:
    invalid = sorted(set(values) - _VALID_CLASS_IDS)
    if invalid:
        raise ValueError(f"{name} contains class IDs outside I01-I15: {invalid}")


def evaluate_predictions(
    input_path: str | Path,
    *,
    true_column: str = "class_id",
    predicted_column: str = "predicted_class",
    prior_prediction_column: str = "prior_predicted_class",
    synthetic: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, prior advantage, and observed MI."""

    from sklearn.metrics import accuracy_score, f1_score

    from .synthetic_demo import _observed_mi

    with Path(input_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Prediction file is empty")
    for column in (true_column, predicted_column, prior_prediction_column):
        if column not in rows[0]:
            raise KeyError(f"Missing prediction column: {column}")
    if "synthetic" not in rows[0]:
        raise KeyError("Prediction file must contain an explicit synthetic column")
    validate_run_kind_values(
        [row["synthetic"] for row in rows],
        expected_synthetic=synthetic,
        context="prediction file",
    )
    y_true = [row[true_column] for row in rows]
    y_pred = [row[predicted_column] for row in rows]
    _require_paper_class_ids(y_true, name=true_column)
    _require_paper_class_ids(y_pred, name=predicted_column)
    prior_classes = {row[prior_prediction_column] for row in rows}
    if len(prior_classes) != 1:
        raise ValueError("A prediction group must carry one frozen prior class")
    prior_class = next(iter(prior_classes))
    _require_paper_class_ids([prior_class], name=prior_prediction_column)
    prior_accuracy = sum(value == prior_class for value in y_true) / len(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    result = {
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "prior_accuracy": prior_accuracy,
        "advantage": accuracy - prior_accuracy,
        "observed_mi_bits": _observed_mi(y_true, y_pred),
        "synthetic": synthetic,
        "run_kind": "synthetic_smoke_test" if synthetic else "real_offline_analysis",
        "notice": SYNTHETIC_NOTICE if synthetic else "real-data analysis; inspect provenance",
        "paper_results_claimed": False,
    }
    if output_path is not None:
        atomic_write_json(output_path, result)
    return result


def evaluate_prediction_groups(
    input_path: str | Path,
    output_path: str | Path,
    *,
    group_columns: tuple[str, ...] = (
        "protocol_block",
        "topology_id",
        "model_id",
        "view",
        "condition",
        "attacker",
    ),
    required_group_columns: tuple[str, ...] = ("view", "condition", "attacker"),
    evaluation_splits: tuple[str, ...] = ("test",),
    true_column: str = "class_id",
    predicted_column: str = "predicted_class",
    prior_prediction_column: str = "prior_predicted_class",
    synthetic: bool = False,
) -> Path:
    """Write paper metrics separately for each available declared result group."""

    import pandas as pd

    from ..metrics import (
        class_accuracy,
        confusion_mutual_information,
        macro_f1,
        scope_mae,
    )

    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError("Prediction file is empty")
    required = {
        true_column,
        predicted_column,
        prior_prediction_column,
        "synthetic",
        *required_group_columns,
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")
    validate_run_kind_values(
        frame["synthetic"].tolist(),
        expected_synthetic=synthetic,
        context="prediction file",
    )
    split_column = "partition" if "partition" in frame else "split" if "split" in frame else None
    if split_column is None:
        raise KeyError("Prediction file must contain partition or split provenance")
    if not evaluation_splits:
        raise ValueError("evaluation_splits must not be empty")
    frame = frame[frame[split_column].astype(str).isin(set(evaluation_splits))]
    if frame.empty:
        raise ValueError(
            f"Prediction file has no rows in evaluation splits {sorted(evaluation_splits)}"
        )
    # The evaluated split is always part of the result stratum. Otherwise a
    # caller requesting validation and test together would silently pool them.
    available_groups = tuple(
        dict.fromkeys(
            column
            for column in (*group_columns, split_column)
            if column in frame
        )
    )
    grouped = (
        frame.groupby(list(available_groups), dropna=False, sort=True)
        if available_groups
        else [((), frame)]
    )
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        keys = key if isinstance(key, tuple) else (key,)
        truth = group[true_column].astype(str).tolist()
        predicted = group[predicted_column].astype(str).tolist()
        _require_paper_class_ids(truth, name=true_column)
        _require_paper_class_ids(predicted, name=predicted_column)
        accuracy = class_accuracy(truth, predicted)
        prior_classes = group[prior_prediction_column].astype(str).drop_duplicates().tolist()
        if len(prior_classes) != 1:
            raise ValueError(
                "Each evaluation group must carry one frozen training-modal class"
            )
        prior_class = prior_classes[0]
        _require_paper_class_ids([prior_class], name=prior_prediction_column)
        prior_accuracy = sum(value == prior_class for value in truth) / len(truth)
        row: dict[str, Any] = dict(zip(available_groups, keys, strict=True))
        row.update(
            {
                "n": len(group),
                "accuracy": accuracy,
                "macro_f1": macro_f1(
                    truth, predicted, labels=tuple(value.value for value in IntentClass)
                ),
                "prior_accuracy": prior_accuracy,
                "advantage": accuracy - prior_accuracy,
                "observed_mi_bits": confusion_mutual_information(
                    truth, predicted, labels=tuple(value.value for value in IntentClass)
                ),
                "synthetic": synthetic,
                "run_kind": (
                    "synthetic_smoke_test" if synthetic else "real_offline_analysis"
                ),
                "notice": (
                    SYNTHETIC_NOTICE
                    if synthetic
                    else "real-data analysis; inspect frozen capture and run provenance"
                ),
                "paper_results_claimed": False,
            }
        )
        if {"scope_q", "predicted_scope_q"} <= set(group):
            true_scope = pd.to_numeric(group["scope_q"], errors="coerce")
            predicted_scope = pd.to_numeric(
                group["predicted_scope_q"], errors="coerce"
            )
            if (
                true_scope.isna().any()
                or predicted_scope.isna().any()
                or not true_scope.isin((1, 2, 4)).all()
                or not predicted_scope.isin((1, 2, 4)).all()
            ):
                raise ValueError("Scope truth and predictions must be q in {1, 2, 4}")
            row["scope_mae_primary_targets"] = scope_mae(
                true_scope, predicted_scope
            )
            row["scope_macro_f1"] = macro_f1(
                true_scope.astype(int).astype(str),
                predicted_scope.astype(int).astype(str),
                labels=("1", "2", "4"),
            )
        rows.append(row)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination
