from __future__ import annotations

import csv

import numpy as np
import pytest

from intent_leakage.experiments.evaluate import evaluate_prediction_groups
from intent_leakage.metrics import (
    advantage_over_prior,
    bandwidth_overhead,
    class_accuracy,
    confusion_matrix_counts,
    confusion_mutual_information,
    confusion_mutual_information_from_matrix,
    empirical_class_prior,
    latency_cost,
    macro_f1,
    nondominated_profiles,
    overflow_rates,
    paired_bandwidth_overheads,
    paired_latency_costs,
    prior_only_accuracy,
    scope_mae,
)


def test_accuracy_prior_advantage_and_scope_mae() -> None:
    truth = ["a", "b", "c", "c"]
    prediction = ["a", "a", "c", "b"]
    accuracy = class_accuracy(truth, prediction)
    assert accuracy == 0.5
    prior = empirical_class_prior(truth, class_order=["a", "b", "c"])
    assert prior == {"a": 0.25, "b": 0.25, "c": 0.5}
    assert prior_only_accuracy(prior) == 0.5
    assert advantage_over_prior(accuracy, prior) == 0.0
    assert advantage_over_prior(0.25, 0.5) == -0.25
    assert scope_mae([1, 2, 4], [1, 3, 2]) == pytest.approx(1.0)


def test_macro_f1_includes_declared_absent_class() -> None:
    # F1(a)=2/3, F1(b)=0 and absent declared c uses zero_division=0.
    value = macro_f1(["a", "b"], ["a", "a"], labels=["a", "b", "c"])
    assert value == pytest.approx((2.0 / 3.0) / 3.0)


def test_confusion_observed_mutual_information() -> None:
    labels = ["a", "b", "c", "d"] * 3
    assert confusion_mutual_information(labels, labels) == pytest.approx(2.0)
    assert confusion_mutual_information(labels, ["a"] * len(labels)) == pytest.approx(0.0)
    permuted = [{"a": "d", "b": "c", "c": "b", "d": "a"}[item] for item in labels]
    assert confusion_mutual_information(labels, permuted) == pytest.approx(2.0)
    matrix, order = confusion_matrix_counts(labels, permuted)
    assert order == ("a", "b", "c", "d")
    assert confusion_mutual_information_from_matrix(matrix) == pytest.approx(2.0)


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        class_accuracy([], [])
    with pytest.raises(ValueError, match="equal lengths"):
        class_accuracy([1], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        scope_mae([1.0], [np.nan])
    with pytest.raises(ValueError, match="at least one"):
        confusion_mutual_information_from_matrix(np.zeros((2, 2)))


def test_latency_bandwidth_and_vectorized_pairing() -> None:
    latency = latency_cost(2.0, 3.0)
    assert latency.absolute == 1.0
    assert latency.relative == 0.5
    assert bandwidth_overhead(100.0, 150.0) == 0.5
    added, ratios = paired_latency_costs([2.0, 4.0], [3.0, 3.0])
    np.testing.assert_allclose(added, [1.0, -1.0])
    np.testing.assert_allclose(ratios, [0.5, -0.25])
    np.testing.assert_allclose(paired_bandwidth_overheads([100, 200], [150, 100]), [0.5, -0.5])
    with pytest.raises(ZeroDivisionError):
        latency_cost(0.0, 1.0)
    with pytest.raises(ZeroDivisionError):
        bandwidth_overhead(0.0, 1.0)


def test_overflow_rates_overall_class_and_scope() -> None:
    result = overflow_rates(
        [False, True, False, True],
        classes=["a", "a", "b", "b"],
        scopes=["low", "high", "low", "high"],
    )
    assert result.overall == 0.5
    assert result.by_class == {"a": 0.5, "b": 0.5}
    assert result.by_scope == {"low": 0.0, "high": 1.0}
    assert result.counts_by_class == {"a": 2, "b": 2}


def test_three_objective_pareto_frontier() -> None:
    records = [
        {"profile": "a", "leakage": 0.2, "added_latency": 2.0, "bandwidth_overhead": 1.0},
        {"profile": "b", "leakage": 0.3, "added_latency": 3.0, "bandwidth_overhead": 1.2},
        {"profile": "c", "leakage": 0.1, "added_latency": 4.0, "bandwidth_overhead": 1.5},
        {"profile": "d", "leakage": 0.2, "added_latency": 1.0, "bandwidth_overhead": 2.0},
    ]
    frontier = nondominated_profiles(records)
    assert {row["profile"] for row in frontier} == {"a", "c", "d"}


def test_evaluation_scores_frozen_training_modal_predictor_on_test(tmp_path) -> None:
    source = tmp_path / "predictions.csv"
    rows = []
    for class_id in ("I01", "I02", "I02", "I02"):
        rows.append(
            {
                "class_id": class_id,
                "predicted_class": "I02",
                "prior_predicted_class": "I01",
                "partition": "test",
                "view": "CT",
                "condition": "no_shaping",
                "attacker": "random_forest",
                "synthetic": True,
            }
        )
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "evaluation.csv"
    evaluate_prediction_groups(source, output, synthetic=True)
    with output.open("r", encoding="utf-8", newline="") as stream:
        result = next(csv.DictReader(stream))
    assert float(result["prior_accuracy"]) == pytest.approx(0.25)
    assert float(result["accuracy"]) == pytest.approx(0.75)
    assert float(result["advantage"]) == pytest.approx(0.5)
    assert result["partition"] == "test"


def test_evaluation_never_pools_requested_splits(tmp_path) -> None:
    source = tmp_path / "predictions.csv"
    rows = [
        {
            "class_id": "I01",
            "predicted_class": "I01" if partition == "test" else "I02",
            "prior_predicted_class": "I01",
            "partition": partition,
            "view": "CT",
            "condition": "no_shaping",
            "attacker": "random_forest",
            "synthetic": True,
        }
        for partition in ("validation", "test")
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "evaluation.csv"
    evaluate_prediction_groups(
        source,
        output,
        evaluation_splits=("validation", "test"),
        synthetic=True,
    )
    with output.open("r", encoding="utf-8", newline="") as stream:
        evaluated = list(csv.DictReader(stream))
    assert {row["partition"] for row in evaluated} == {"validation", "test"}
    assert len(evaluated) == 2


def test_evaluation_rejects_invalid_frozen_prior_class(tmp_path) -> None:
    source = tmp_path / "predictions.csv"
    row = {
        "class_id": "I01",
        "predicted_class": "I01",
        "prior_predicted_class": "I99",
        "partition": "test",
        "view": "CT",
        "condition": "no_shaping",
        "attacker": "random_forest",
        "synthetic": True,
    }
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="outside I01-I15"):
        evaluate_prediction_groups(
            source, tmp_path / "evaluation.csv", synthetic=True
        )
