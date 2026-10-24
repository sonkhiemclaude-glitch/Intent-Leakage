"""Information-theoretic certificates from paper Sections 4.4-4.6."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from math import fsum
from typing import Union
import numpy as np


Label = Hashable
ProbabilityInput = Union[Mapping[Label, float], Sequence[float], np.ndarray]
GroupPartition = Mapping[Hashable, Iterable[Label]]


def _probability_mapping(values: ProbabilityInput, *, name: str) -> dict[Label, float]:
    if isinstance(values, Mapping):
        result = {key: float(value) for key, value in values.items()}
    else:
        array = np.asarray(values, dtype=float).reshape(-1)
        result = {idx: float(value) for idx, value in enumerate(array.tolist())}
    if not result:
        raise ValueError(f"{name} must contain at least one class")
    array = np.asarray(list(result.values()), dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    if not np.isclose(array.sum(), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one, got {array.sum():.16g}")
    # The accepted normalization tolerance must not propagate into probabilities
    # above one, especially at the all-overflow endpoint of binary entropy.
    total = fsum(result.values())
    return {key: value / total for key, value in result.items()}


def _weighted_overflow_probability(
    prior: Mapping[Label, float], rates: Mapping[Label, float]
) -> float:
    """Accumulate validated probabilities without crossing [0, 1] by roundoff."""

    return min(1.0, max(0.0, fsum(prior[label] * rates[label] for label in prior)))


def _rate_mapping(
    rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
    labels: Iterable[Label],
) -> dict[Label, float]:
    ordered_labels = list(labels)
    if isinstance(rates, Mapping):
        missing = set(ordered_labels) - set(rates)
        extra = set(rates) - set(ordered_labels)
        if missing or extra:
            raise ValueError(
                "overflow-rate labels must match prior labels; "
                f"missing={sorted(map(str, missing))}, "
                f"extra={sorted(map(str, extra))}"
            )
        result = {label: float(rates[label]) for label in ordered_labels}
    else:
        values = np.asarray(rates, dtype=float).reshape(-1)
        if values.size != len(ordered_labels):
            raise ValueError(
                f"expected {len(ordered_labels)} overflow rates, got {values.size}"
            )
        result = dict(zip(ordered_labels, values.tolist()))
    array = np.asarray(list(result.values()), dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("overflow rates must be finite and in [0, 1]")
    return result


def _validated_groups(
    groups: GroupPartition,
    labels: Iterable[Label],
) -> dict[Hashable, tuple[Label, ...]]:
    expected = set(labels)
    if not groups:
        raise ValueError("groups must contain at least one nonempty group")
    result: dict[Hashable, tuple[Label, ...]] = {}
    seen: set[Label] = set()
    for group, members_iterable in groups.items():
        members = tuple(members_iterable)
        if not members:
            raise ValueError(f"group {group!r} is empty")
        if len(set(members)) != len(members):
            raise ValueError(f"group {group!r} contains duplicate classes")
        overlap = seen.intersection(members)
        if overlap:
            raise ValueError(f"classes occur in multiple groups: {sorted(map(str, overlap))}")
        unknown = set(members) - expected
        if unknown:
            raise ValueError(f"group {group!r} contains unknown classes: {unknown!r}")
        seen.update(members)
        result[group] = members
    missing = expected - seen
    if missing:
        raise ValueError(f"groups do not cover classes: {sorted(map(str, missing))}")
    return result


def binary_entropy_bits(probability: float) -> float:
    """Return binary entropy ``h_2(p)`` with the endpoint convention."""

    p = float(probability)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def entropy_bits(probabilities: ProbabilityInput) -> float:
    """Return Shannon entropy in bits for a normalized distribution."""

    distribution = _probability_mapping(probabilities, name="probabilities")
    positive = np.asarray([value for value in distribution.values() if value > 0.0])
    return float(-np.sum(positive * np.log2(positive)))


def mean_overflow_probability(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> float:
    """Return ``epsilon_bar = sum_k pi_k epsilon_k`` from Section 4.6."""

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    return _weighted_overflow_probability(pi, epsilon)


def fitted_class_prior(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> dict[Label, float]:
    """Compute the fitted prior in paper Equations (6) and (18).

    Raises ``ValueError`` when all probability mass overflows, because
    conditioning on ``E=1`` is then undefined.
    """

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    fitted_mass = 1.0 - _weighted_overflow_probability(pi, epsilon)
    if fitted_mass <= 1e-15:
        raise ValueError("fitted class prior is undefined because Pr(E=1) is zero")
    fitted = {
        label: pi[label] * (1.0 - epsilon[label]) / fitted_mass for label in pi
    }
    total = sum(fitted.values())
    # Renormalize only floating-point drift from the Bayes calculation.
    return {label: value / total for label, value in fitted.items()}


def fitted_group_distribution(
    fitted_prior: ProbabilityInput,
    groups: GroupPartition,
) -> dict[Hashable, float]:
    """Aggregate a fitted class prior into the shaping-group distribution."""

    pi_fitted = _probability_mapping(fitted_prior, name="fitted class prior")
    partition = _validated_groups(groups, pi_fitted)
    return {
        group: float(sum(pi_fitted[label] for label in members))
        for group, members in partition.items()
    }


def fitted_accuracy_upper_bound(
    fitted_prior: ProbabilityInput,
    groups: GroupPartition,
) -> float:
    """Return Proposition 1 / Equation (7)'s fitted accuracy bound."""

    pi_fitted = _probability_mapping(fitted_prior, name="fitted class prior")
    partition = _validated_groups(groups, pi_fitted)
    bound = sum(max(pi_fitted[label] for label in members) for members in partition.values())
    return float(min(1.0, max(0.0, bound)))


def fitted_advantage_upper_bound(
    fitted_prior: ProbabilityInput,
    groups: GroupPartition,
) -> float:
    """Return the prior-aware fitted advantage bound in Equation (12)."""

    pi_fitted = _probability_mapping(fitted_prior, name="fitted class prior")
    return fitted_accuracy_upper_bound(pi_fitted, groups) - max(pi_fitted.values())


def conditional_mi_upper_bound(
    fitted_prior: ProbabilityInput,
    groups: GroupPartition,
) -> float:
    """Return ``H(G | E=1)`` from Proposition 2 / Equation (13)."""

    group_distribution = fitted_group_distribution(fitted_prior, groups)
    return entropy_bits(group_distribution)


def unconditional_accuracy_upper_bound(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
    groups: GroupPartition,
) -> float:
    """Return the overflow-aware attacker accuracy bound in Equation (19)."""

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    partition = _validated_groups(groups, pi)
    mean_overflow = _weighted_overflow_probability(pi, epsilon)
    fitted_maxima = sum(
        max(pi[label] * (1.0 - epsilon[label]) for label in members)
        for members in partition.values()
    )
    bound = mean_overflow + fitted_maxima
    return float(min(1.0, max(0.0, bound)))


def overflow_indicator_information(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> float:
    """Return exact ``I(C;E)`` in bits from paper Equation (24)."""

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    mean_overflow = _weighted_overflow_probability(pi, epsilon)
    information = binary_entropy_bits(mean_overflow) - sum(
        pi[label] * binary_entropy_bits(epsilon[label]) for label in pi
    )
    if information < -1e-12:
        raise ArithmeticError(f"computed negative I(C;E): {information}")
    return float(max(0.0, information))


def overflow_class_distribution(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
) -> dict[Label, float]:
    """Return ``Pr(C=k | E=0)``; reject a zero-probability overflow event."""

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    mean_overflow = _weighted_overflow_probability(pi, epsilon)
    if mean_overflow <= 1e-15:
        raise ValueError("overflow class distribution is undefined because Pr(E=0) is zero")
    result = {label: pi[label] * epsilon[label] / mean_overflow for label in pi}
    total = sum(result.values())
    return {label: value / total for label, value in result.items()}


def overflow_conditional_mi_mixture(
    mean_overflow: float,
    fitted_conditional_mi: float,
    overflow_conditional_mi: float,
) -> float:
    """Evaluate the conditional-information expansion in Equation (23).

    This function accepts information quantities in bits and returns
    ``(1-epsilon_bar) I(C;O|E=1) + epsilon_bar I(C;O|E=0)``.  It is useful for
    auditing the Corollary 1 proof independently of the entropy bounds.
    """

    probability = float(mean_overflow)
    fitted = float(fitted_conditional_mi)
    overflow = float(overflow_conditional_mi)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("mean_overflow must be finite and in [0, 1]")
    if not np.isfinite(fitted) or fitted < 0.0:
        raise ValueError("fitted_conditional_mi must be finite and nonnegative")
    if not np.isfinite(overflow) or overflow < 0.0:
        raise ValueError("overflow_conditional_mi must be finite and nonnegative")
    return (1.0 - probability) * fitted + probability * overflow


def augmented_observation_information(
    indicator_information: float,
    mean_overflow: float,
    fitted_conditional_mi: float,
    overflow_conditional_mi: float,
) -> float:
    """Evaluate ``I(C;E)+I(C;O|E)`` in Equations (22)-(23).

    The returned quantity equals ``I(C;O,E)`` for exact inputs and upper-bounds
    ``I(C;O)``.  Equality with the latter requires the overflow indicator to be
    recoverable from the observation; the implementation does not assume that.
    """

    indicator = float(indicator_information)
    if not np.isfinite(indicator) or indicator < 0.0:
        raise ValueError("indicator_information must be finite and nonnegative")
    return indicator + overflow_conditional_mi_mixture(
        mean_overflow, fitted_conditional_mi, overflow_conditional_mi
    )


def sharp_unconditional_mi_upper_bound(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
    groups: GroupPartition,
) -> float:
    """Return Corollary 1's sharper unconditional bound, Equation (20)."""

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    partition = _validated_groups(groups, pi)
    mean_overflow = _weighted_overflow_probability(pi, epsilon)
    indicator_information = overflow_indicator_information(pi, epsilon)

    fitted_component = 0.0
    if mean_overflow < 1.0 - 1e-15:
        fitted = fitted_class_prior(pi, epsilon)
        fitted_component = (1.0 - mean_overflow) * conditional_mi_upper_bound(
            fitted, partition
        )

    overflow_component = 0.0
    if mean_overflow > 1e-15:
        overflow_component = mean_overflow * entropy_bits(
            overflow_class_distribution(pi, epsilon)
        )

    # Equations (22)-(23), with the two conditional MI branches replaced by
    # their entropy upper bounds.
    fitted_information_bound = (
        fitted_component / (1.0 - mean_overflow)
        if mean_overflow < 1.0 - 1e-15
        else 0.0
    )
    overflow_information_bound = (
        overflow_component / mean_overflow if mean_overflow > 1e-15 else 0.0
    )
    bound = augmented_observation_information(
        indicator_information,
        mean_overflow,
        fitted_information_bound,
        overflow_information_bound,
    )
    class_entropy = entropy_bits(pi)
    if bound > class_entropy and bound - class_entropy < 1e-10:
        bound = class_entropy
    return float(min(class_entropy, max(0.0, bound)))


def conservative_unconditional_mi_upper_bound(
    prior: ProbabilityInput,
    overflow_rates: Mapping[Label, float] | Sequence[float] | np.ndarray,
    groups: GroupPartition,
) -> float:
    """Return the capped conservative bound in paper Equation (21).

    The uncapped expression is ``h2(epsilon_bar)`` plus the fitted group
    entropy weighted by fitted probability plus ``epsilon_bar log2(K)``.
    Corollary 1 explicitly permits capping it at ``H(C)``.
    """

    pi = _probability_mapping(prior, name="class prior")
    epsilon = _rate_mapping(overflow_rates, pi)
    partition = _validated_groups(groups, pi)
    mean_overflow = _weighted_overflow_probability(pi, epsilon)

    fitted_component = 0.0
    if mean_overflow < 1.0 - 1e-15:
        fitted = fitted_class_prior(pi, epsilon)
        fitted_component = (1.0 - mean_overflow) * conditional_mi_upper_bound(
            fitted, partition
        )

    uncapped = (
        binary_entropy_bits(mean_overflow)
        + fitted_component
        + mean_overflow * np.log2(len(pi))
    )
    return float(min(entropy_bits(pi), max(0.0, uncapped)))


# Concise aliases matching prose used in some experiment code.
unconditional_mi_upper_bound = conservative_unconditional_mi_upper_bound
