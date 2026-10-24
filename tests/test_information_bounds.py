from __future__ import annotations

import math

import pytest

from intent_leakage.metrics.bounds import (
    augmented_observation_information,
    binary_entropy_bits,
    conditional_mi_upper_bound,
    conservative_unconditional_mi_upper_bound,
    entropy_bits,
    fitted_accuracy_upper_bound,
    fitted_advantage_upper_bound,
    fitted_class_prior,
    mean_overflow_probability,
    overflow_class_distribution,
    overflow_indicator_information,
    overflow_conditional_mi_mixture,
    sharp_unconditional_mi_upper_bound,
    unconditional_accuracy_upper_bound,
)


def test_equations_7_12_and_13_balanced_fitted_population() -> None:
    fitted = {"I01": 0.25, "I02": 0.25, "I03": 0.25, "I04": 0.25}
    groups = {"G1": ["I01", "I02"], "G2": ["I03", "I04"]}
    assert fitted_accuracy_upper_bound(fitted, groups) == pytest.approx(0.5)
    assert fitted_advantage_upper_bound(fitted, groups) == pytest.approx(0.25)
    assert conditional_mi_upper_bound(fitted, groups) == pytest.approx(1.0)

    one_group = {"all": list(fitted)}
    assert fitted_accuracy_upper_bound(fitted, one_group) == pytest.approx(0.25)
    assert fitted_advantage_upper_bound(fitted, one_group) == pytest.approx(0.0)
    assert conditional_mi_upper_bound(fitted, one_group) == pytest.approx(0.0)


def test_equation_18_fitted_prior() -> None:
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    overflow = {"a": 0.1, "b": 0.2, "c": 0.4}
    assert mean_overflow_probability(prior, overflow) == pytest.approx(0.19)
    fitted = fitted_class_prior(prior, overflow)
    assert fitted == pytest.approx(
        {"a": 0.45 / 0.81, "b": 0.24 / 0.81, "c": 0.12 / 0.81}
    )
    with pytest.raises(ValueError, match=r"Pr\(E=1\)"):
        fitted_class_prior(prior, {key: 1.0 for key in prior})


def test_equation_19_overflow_accuracy_bound() -> None:
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    overflow = {"a": 0.1, "b": 0.2, "c": 0.4}
    groups = {"large": ["a", "b"], "single": ["c"]}
    # 0.19 overflow + max(0.45,0.24) + 0.12 fitted group maxima.
    assert unconditional_accuracy_upper_bound(prior, overflow, groups) == pytest.approx(0.76)
    assert unconditional_accuracy_upper_bound(
        prior, {key: 1.0 for key in prior}, groups
    ) == pytest.approx(1.0)


def test_equations_20_21_and_24() -> None:
    prior = {"a": 0.5, "b": 0.3, "c": 0.2}
    overflow = {"a": 0.1, "b": 0.2, "c": 0.4}
    groups = {"large": ["a", "b"], "single": ["c"]}
    mean_overflow = 0.19
    expected_indicator = binary_entropy_bits(mean_overflow) - sum(
        prior[key] * binary_entropy_bits(overflow[key]) for key in prior
    )
    assert overflow_indicator_information(prior, overflow) == pytest.approx(
        expected_indicator
    )

    sharp = sharp_unconditional_mi_upper_bound(prior, overflow, groups)
    conservative = conservative_unconditional_mi_upper_bound(prior, overflow, groups)
    assert 0.0 <= sharp <= conservative <= entropy_bits(prior)

    overflow_distribution = overflow_class_distribution(prior, overflow)
    assert overflow_distribution == pytest.approx(
        {key: prior[key] * overflow[key] / mean_overflow for key in prior}
    )


def test_equations_22_and_23_chain_rule_expansion() -> None:
    mixture = overflow_conditional_mi_mixture(0.2, 0.5, 1.5)
    assert mixture == pytest.approx(0.7)
    augmented = augmented_observation_information(0.1, 0.2, 0.5, 1.5)
    assert augmented == pytest.approx(0.8)
    with pytest.raises(ValueError, match="mean_overflow"):
        overflow_conditional_mi_mixture(1.1, 0.5, 1.5)
    with pytest.raises(ValueError, match="nonnegative"):
        augmented_observation_information(-0.1, 0.2, 0.5, 1.5)


def test_sharp_bound_equivalent_entropy_identity() -> None:
    prior = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
    overflow = {"a": 0.0, "b": 0.2, "c": 0.5, "d": 0.8}
    groups = {"g1": ["a", "b"], "g2": ["c", "d"]}
    fitted = fitted_class_prior(prior, overflow)
    fitted_mass = 1.0 - mean_overflow_probability(prior, overflow)
    within_group_entropy = 0.0
    for members in groups.values():
        group_mass = sum(fitted[key] for key in members)
        conditional = {
            key: fitted[key] / group_mass for key in members if fitted[key] > 0.0
        }
        within_group_entropy += group_mass * entropy_bits(conditional)
    expected = entropy_bits(prior) - fitted_mass * within_group_entropy
    assert sharp_unconditional_mi_upper_bound(prior, overflow, groups) == pytest.approx(
        expected
    )


def test_overflow_endpoints_and_class_independence() -> None:
    prior = {"a": 0.8, "b": 0.2}
    groups = {"all": ["a", "b"]}
    no_overflow = {"a": 0.0, "b": 0.0}
    all_overflow = {"a": 1.0, "b": 1.0}
    equal_overflow = {"a": 0.25, "b": 0.25}
    assert sharp_unconditional_mi_upper_bound(prior, no_overflow, groups) == 0.0
    assert conservative_unconditional_mi_upper_bound(prior, no_overflow, groups) == 0.0
    assert overflow_indicator_information(prior, equal_overflow) == pytest.approx(0.0)
    assert sharp_unconditional_mi_upper_bound(prior, all_overflow, groups) == pytest.approx(
        entropy_bits(prior)
    )
    assert conservative_unconditional_mi_upper_bound(
        prior, all_overflow, groups
    ) == pytest.approx(entropy_bits(prior))


def test_partition_and_probability_validation() -> None:
    prior = {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError, match="cover"):
        fitted_accuracy_upper_bound(prior, {"g": ["a"]})
    with pytest.raises(ValueError, match="multiple"):
        fitted_accuracy_upper_bound(prior, {"g1": ["a"], "g2": ["a", "b"]})
    with pytest.raises(ValueError, match="sum to one"):
        fitted_accuracy_upper_bound({"a": 0.4, "b": 0.4}, {"g": ["a", "b"]})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fitted_class_prior(prior, {"a": -0.1, "b": 0.0})


def test_binary_entropy_endpoints() -> None:
    assert binary_entropy_bits(0.0) == 0.0
    assert binary_entropy_bits(1.0) == 0.0
    assert binary_entropy_bits(0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        binary_entropy_bits(math.nan)


@pytest.mark.parametrize("weights", [
    [1.0] * 15,
    [1.0 / k for k in range(1, 16)],
    [0.7] + [0.3 / 14] * 14,
])
@pytest.mark.parametrize("normalization_drift", [0.0, 1e-11, -1e-11])
def test_multiclass_overflow_endpoints_tolerate_normalization_roundoff(
    weights, normalization_drift
) -> None:
    total = sum(weights)
    prior = {k: value / total * (1 + normalization_drift)
             for k, value in enumerate(weights)}
    groups = {"all": list(prior)}
    overflow = {k: 1.0 for k in prior}
    assert mean_overflow_probability(prior, overflow) == pytest.approx(1.0)
    assert overflow_indicator_information(prior, overflow) == pytest.approx(0.0, abs=1e-12)
    assert sharp_unconditional_mi_upper_bound(prior, overflow, groups) == pytest.approx(entropy_bits(prior))
    assert conservative_unconditional_mi_upper_bound(prior, overflow, groups) == pytest.approx(entropy_bits(prior))
    assert sharp_unconditional_mi_upper_bound(prior, {k: 0.0 for k in prior}, groups) == pytest.approx(0.0)
