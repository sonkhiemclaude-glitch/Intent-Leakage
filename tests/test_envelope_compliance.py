from __future__ import annotations

from dataclasses import replace

import pytest

from intent_leakage.exceptions import UnsafeDecoyError
from intent_leakage.schemas import ApplicationMessage, Channel, Direction
from intent_leakage.shaping import (
    FixedCadenceScheduler,
    SafeDecoy,
    make_uniform_envelope,
    verify_envelope_structure,
)


def _result():
    profile = make_uniform_envelope(
        profile_id="compliance-profile",
        group_id="all-classes",
        round_count=1,
        period_s=1.0,
        bucket_bytes=256,
    )
    messages = (
        ApplicationMessage(
            "tool-request",
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            100,
            0.01,
            "tool_request",
        ),
    )
    return profile, FixedCadenceScheduler().schedule(profile, messages)


def test_structural_conformance_never_claims_distributional_proof() -> None:
    profile, result = _result()
    report = verify_envelope_structure(profile, result)
    assert report.structurally_conformant
    assert report.fitted_execution
    assert report.passes_fitted_envelope
    assert not report.distributional_equivalence_proven
    assert "cannot prove" in report.limitation


def test_verifier_rejects_changed_bucket_and_timing() -> None:
    profile, result = _result()
    tampered_first = replace(
        result.transmissions[0],
        wire_size_bytes=result.transmissions[0].wire_size_bytes + 1,
        scheduled_at_s=result.transmissions[0].scheduled_at_s + 0.1,
    )
    tampered = replace(
        result,
        transmissions=(tampered_first, *result.transmissions[1:]),
    )
    report = verify_envelope_structure(profile, tampered)
    assert not report.structurally_conformant
    assert {issue.code for issue in report.issues} >= {
        "bucket_mismatch",
        "timing_mismatch",
    }


def test_unsafe_decoy_is_rejected_at_definition_time() -> None:
    with pytest.raises(UnsafeDecoyError):
        SafeDecoy(
            decoy_id="unsafe",
            operation="write_flow",
            target="switch-1",
            nominal_size_bytes=64,
            read_only=False,
            side_effect_free=False,
        )


def test_string_decoy_safety_flags_are_rejected() -> None:
    with pytest.raises(UnsafeDecoyError, match="literal booleans"):
        SafeDecoy(
            decoy_id="misparsed",
            operation="read_policy",
            target="controller",
            nominal_size_bytes=64,
            authenticated="false",  # type: ignore[arg-type]
            authorized="false",  # type: ignore[arg-type]
            read_only="false",  # type: ignore[arg-type]
            side_effect_free="false",  # type: ignore[arg-type]
            excluded_from_truth_criterion="false",  # type: ignore[arg-type]
        )
