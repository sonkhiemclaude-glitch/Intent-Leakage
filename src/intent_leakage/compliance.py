"""Section 7.3 intent-confidentiality compliance decision support.

The routines combine already computed one-sided uncertainty limits with the
paper's seven checks.  They do not turn empirical tests into a proof of the
group-constant trace-law assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Mapping

from .schemas import IntentClass


class CheckStatus(str, Enum):
    """Outcome of one compliance check."""

    PASS = "pass"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True, slots=True)
class ComplianceThresholds:
    """Predeclared deployment-specific thresholds from Section 7.3."""

    attack_slack_delta: float
    fitted_information_max_bits: float
    total_information_max_bits: float
    class_overflow_max: float
    added_latency_max_s: float
    bandwidth_overhead_max: float

    def __post_init__(self) -> None:
        values = (
            self.attack_slack_delta,
            self.fitted_information_max_bits,
            self.total_information_max_bits,
            self.class_overflow_max,
            self.added_latency_max_s,
            self.bandwidth_overhead_max,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("all compliance thresholds must be finite and nonnegative")
        if self.class_overflow_max > 1:
            raise ValueError("class_overflow_max must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ComplianceEvidence:
    """Evidence supplied to the seven-step compliance test.

    Fields ending in ``_ucl`` are precomputed one-sided upper confidence
    limits.  The caller is responsible for a predeclared, selection-aware
    uncertainty procedure.  In particular, the best-attack UCL should apply
    the maximum across attack families inside each resample rather than pick
    a test winner and then ignore selection.
    """

    declarations_frozen: bool
    unresolved_placeholders: tuple[str, ...]
    splits_frozen_before_fitting: bool
    test_influenced_fitting: bool
    all_declared_cells_captured: bool
    paired_unprotected_protected: bool
    encryption_confirmed: bool
    channel_separation_disclosed: bool
    envelope_checks_total: int
    envelope_checks_failed: int
    fitted_accuracy_bound: float | None
    best_attack_accuracy_ucl: float | None
    fitted_group_entropy_bits: float | None
    total_information_bound_bits: float | None
    class_overflow_ucl: Mapping[str, float] = field(default_factory=dict)
    expected_class_ids: tuple[str, ...] = tuple(value.value for value in IntentClass)
    added_latency_ucl_s: float | None = None
    bandwidth_overhead_ucl: float | None = None
    failures_published: bool = False
    group_constant_assumption_declared: bool = False
    empirical_limits_disclosed: bool = False


@dataclass(frozen=True, slots=True)
class ComplianceCheck:
    """Result and explanation for one numbered paper check."""

    number: int
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    """Complete decision-support report.

    ``passes`` means all seven operational checks pass under the supplied
    thresholds.  It is not an unconditional proof of trace-law equality.
    """

    checks: tuple[ComplianceCheck, ...]
    passes: bool
    formal_guarantee_is_conditional: bool = True
    statement: str = (
        "Schedule and reference-attack checks cannot prove equality of complete "
        "trace distributions; the formal result remains conditional on the "
        "declared group-constant trace-law assumption."
    )


def _finite_nonnegative(value: float | None) -> bool:
    return value is not None and isfinite(value) and value >= 0


def _finite(value: float | None) -> bool:
    """Return whether a possibly signed metric is present and finite."""

    return value is not None and isfinite(value)


def evaluate_compliance(
    evidence: ComplianceEvidence,
    thresholds: ComplianceThresholds,
) -> ComplianceReport:
    """Evaluate the seven steps in paper Section 7.3.

    This function only compares evidence and thresholds.  It never estimates
    a confidence interval, silently fills a missing value, or certifies
    Equation (5)'s distributional assumption.
    """

    declared = evidence.declarations_frozen and not evidence.unresolved_placeholders
    check_1 = ComplianceCheck(
        1,
        "declare",
        CheckStatus.PASS if declared else CheckStatus.FAIL,
        "All classes, scopes, priors, observer positions, profiles, policies, and thresholds are frozen."
        if declared
        else f"Unfrozen declaration or placeholders: {list(evidence.unresolved_placeholders)}",
    )

    separated = evidence.splits_frozen_before_fitting and not evidence.test_influenced_fitting
    check_2 = ComplianceCheck(
        2,
        "separate",
        CheckStatus.PASS if separated else CheckStatus.FAIL,
        "Test traces did not influence fitting." if separated else "Test/fitting separation failed or is undocumented.",
    )

    captured = all(
        (
            evidence.all_declared_cells_captured,
            evidence.paired_unprotected_protected,
            evidence.encryption_confirmed,
            evidence.channel_separation_disclosed,
        )
    )
    check_3 = ComplianceCheck(
        3,
        "capture",
        CheckStatus.PASS if captured else CheckStatus.FAIL,
        "Paired encrypted traces cover every declared cell and channel-separation status is disclosed."
        if captured
        else "Capture coverage, pairing, encryption, or channel-separation disclosure is incomplete.",
    )

    expected_classes = set(evidence.expected_class_ids)
    overflow_classes = set(evidence.class_overflow_ucl)
    complete_overflow_coverage = (
        bool(expected_classes) and overflow_classes == expected_classes
    )
    checked_envelope = (
        evidence.envelope_checks_total > 0
        and evidence.envelope_checks_failed == 0
        and complete_overflow_coverage
        and all(_finite_nonnegative(v) and v <= 1 for v in evidence.class_overflow_ucl.values())
    )
    check_4 = ComplianceCheck(
        4,
        "check_envelope",
        CheckStatus.PASS if checked_envelope else CheckStatus.FAIL,
        f"Envelope checks: {evidence.envelope_checks_total - evidence.envelope_checks_failed}/"
        f"{evidence.envelope_checks_total} passed; class-wise one-sided overflow bounds supplied."
        if checked_envelope
        else "A fitted trace failed, no trace was checked, or class-wise one-sided overflow bounds are missing.",
    )

    attack_evidence = _finite_nonnegative(evidence.best_attack_accuracy_ucl)
    check_5 = ComplianceCheck(
        5,
        "attack",
        CheckStatus.PASS if attack_evidence else CheckStatus.FAIL,
        "A one-sided upper limit for the held-out best reference attack is supplied."
        if attack_evidence
        else "Held-out, selection-aware best-attack evidence is missing.",
    )

    compared = all(
        _finite_nonnegative(value)
        for value in (
            evidence.fitted_accuracy_bound,
            evidence.best_attack_accuracy_ucl,
            evidence.fitted_group_entropy_bits,
            evidence.total_information_bound_bits,
        )
    )
    if compared:
        compared = bool(
            evidence.best_attack_accuracy_ucl
            <= evidence.fitted_accuracy_bound + thresholds.attack_slack_delta
            and evidence.fitted_group_entropy_bits
            <= thresholds.fitted_information_max_bits
            and evidence.total_information_bound_bits
            <= thresholds.total_information_max_bits
            and evidence.group_constant_assumption_declared
        )
    check_6 = ComplianceCheck(
        6,
        "compare",
        CheckStatus.PASS if compared else CheckStatus.FAIL,
        "Accuracy and information limits meet predeclared bounds, with the formal assumption explicitly declared."
        if compared
        else "One or more accuracy/information limits fail, are missing, or the conditional assumption is not declared.",
    )

    operational = (
        complete_overflow_coverage
        and all(value <= thresholds.class_overflow_max for value in evidence.class_overflow_ucl.values())
        # Paired overhead estimates are signed in Section 5.7. A negative
        # one-sided UCL is favorable evidence and must not be rejected merely
        # for being below zero; only the configured budgets are nonnegative.
        and _finite(evidence.added_latency_ucl_s)
        and evidence.added_latency_ucl_s <= thresholds.added_latency_max_s
        and _finite(evidence.bandwidth_overhead_ucl)
        and evidence.bandwidth_overhead_ucl <= thresholds.bandwidth_overhead_max
        and evidence.failures_published
        and evidence.empirical_limits_disclosed
    )
    check_7 = ComplianceCheck(
        7,
        "account_for_operation",
        CheckStatus.PASS if operational else CheckStatus.FAIL,
        "Overflow, latency, and bandwidth meet budgets; failures and empirical limitations are published."
        if operational
        else "An operational budget fails/is missing, or failures/limitations are not published.",
    )

    checks = (check_1, check_2, check_3, check_4, check_5, check_6, check_7)
    return ComplianceReport(
        checks=checks,
        passes=all(check.status is CheckStatus.PASS for check in checks),
    )
