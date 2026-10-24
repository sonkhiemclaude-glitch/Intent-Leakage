from intent_leakage.compliance import (
    CheckStatus,
    ComplianceEvidence,
    ComplianceThresholds,
    evaluate_compliance,
)
from intent_leakage.schemas import IntentClass


def _evidence(**changes):
    values = dict(
        declarations_frozen=True,
        unresolved_placeholders=(),
        splits_frozen_before_fitting=True,
        test_influenced_fitting=False,
        all_declared_cells_captured=True,
        paired_unprotected_protected=True,
        encryption_confirmed=True,
        channel_separation_disclosed=True,
        envelope_checks_total=20,
        envelope_checks_failed=0,
        fitted_accuracy_bound=0.3,
        best_attack_accuracy_ucl=0.32,
        fitted_group_entropy_bits=1.0,
        total_information_bound_bits=1.5,
        class_overflow_ucl={class_id.value: 0.03 for class_id in IntentClass},
        added_latency_ucl_s=0.5,
        bandwidth_overhead_ucl=0.4,
        failures_published=True,
        group_constant_assumption_declared=True,
        empirical_limits_disclosed=True,
    )
    values.update(changes)
    return ComplianceEvidence(**values)


def _thresholds():
    return ComplianceThresholds(0.05, 1.1, 1.6, 0.05, 0.6, 0.5)


def test_all_seven_checks_can_pass_but_report_remains_conditional():
    report = evaluate_compliance(_evidence(), _thresholds())
    assert report.passes
    assert report.formal_guarantee_is_conditional
    assert len(report.checks) == 7
    assert all(check.status is CheckStatus.PASS for check in report.checks)
    assert "cannot prove" in report.statement


def test_missing_failure_disclosure_fails_operational_check():
    report = evaluate_compliance(_evidence(failures_published=False), _thresholds())
    assert not report.passes
    assert report.checks[-1].status is CheckStatus.FAIL


def test_scope_certificate_requires_assumption_declaration():
    report = evaluate_compliance(
        _evidence(group_constant_assumption_declared=False), _thresholds()
    )
    assert report.checks[5].status is CheckStatus.FAIL


def test_missing_class_overflow_bound_fails_compliance():
    report = evaluate_compliance(
        _evidence(class_overflow_ucl={"I01": 0.01}), _thresholds()
    )
    assert report.checks[3].status is CheckStatus.FAIL
    assert report.checks[6].status is CheckStatus.FAIL


def test_negative_cost_upper_limits_are_valid_favorable_evidence():
    report = evaluate_compliance(
        _evidence(added_latency_ucl_s=-0.1, bandwidth_overhead_ucl=-0.2),
        _thresholds(),
    )
    assert report.passes
    assert report.checks[6].status is CheckStatus.PASS
