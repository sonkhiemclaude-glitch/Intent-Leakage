"""Structural envelope checks, distinct from distributional certification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from intent_leakage.shaping.envelope import EnvelopeProfile
from intent_leakage.shaping.scheduler import ShapingResult


_NO_DISTRIBUTIONAL_PROOF = (
    "A structural schedule check cannot prove the group-constant trace "
    "distribution required by Assumption 1. Packetization, retransmissions, "
    "compression, processing time, and other observable effects require "
    "separate empirical and implementation validation."
)


@dataclass(frozen=True, slots=True)
class ConformanceIssue:
    """One mismatch between a frozen profile and an emitted schedule."""

    code: str
    detail: str
    slot_id: str | None = None


@dataclass(frozen=True, slots=True)
class EnvelopeConformanceReport:
    """Result of a mechanical profile check."""

    structurally_conformant: bool
    fitted_execution: bool
    passes_fitted_envelope: bool
    issues: tuple[ConformanceIssue, ...]
    distributional_equivalence_proven: bool = False
    limitation: str = _NO_DISTRIBUTIONAL_PROOF


def verify_envelope_structure(
    profile: EnvelopeProfile,
    result: ShapingResult,
    *,
    timing_tolerance_s: float = 1e-9,
) -> EnvelopeConformanceReport:
    """Mechanically verify sizes, timing, directions, decoys, and termination.

    This implements the structural part of Section 7.3's envelope check.  It
    never returns a claim that Assumption 1 or distributional equality has
    been proved.
    """

    if timing_tolerance_s < 0:
        raise ValueError("timing_tolerance_s must be nonnegative")
    issues: list[ConformanceIssue] = []
    if result.profile_id != profile.profile_id:
        issues.append(
            ConformanceIssue(
                "profile_id_mismatch",
                f"expected {profile.profile_id!r}, got {result.profile_id!r}",
            )
        )
    if result.overflow_policy is not profile.overflow_policy:
        issues.append(
            ConformanceIssue(
                "overflow_policy_mismatch",
                f"expected {profile.overflow_policy.value!r}, got "
                f"{result.overflow_policy.value!r}",
            )
        )
    if len(result.transmissions) != len(profile.slots):
        issues.append(
            ConformanceIssue(
                "slot_count_mismatch",
                f"expected {len(profile.slots)}, got {len(result.transmissions)}",
            )
        )

    real_source_ids = [
        transmission.source_message_id
        for transmission in result.transmissions
        if not transmission.is_decoy and transmission.source_message_id is not None
    ]
    if len(real_source_ids) != len(set(real_source_ids)):
        issues.append(
            ConformanceIssue(
                "duplicate_real_message",
                "one real source message was emitted in more than one slot",
            )
        )

    for slot, transmission in zip(profile.slots, result.transmissions):
        if transmission.slot_id != slot.slot_id:
            issues.append(
                ConformanceIssue(
                    "slot_id_mismatch",
                    f"expected {slot.slot_id!r}, got {transmission.slot_id!r}",
                    slot.slot_id,
                )
            )
        expected_time = profile.absolute_slot_time(slot)
        if not isclose(
            transmission.scheduled_at_s,
            expected_time,
            rel_tol=0.0,
            abs_tol=timing_tolerance_s,
        ):
            issues.append(
                ConformanceIssue(
                    "timing_mismatch",
                    f"expected {expected_time}, got {transmission.scheduled_at_s}",
                    slot.slot_id,
                )
            )
        if transmission.channel is not slot.channel:
            issues.append(
                ConformanceIssue(
                    "channel_mismatch",
                    f"expected {slot.channel.value}, got {transmission.channel.value}",
                    slot.slot_id,
                )
            )
        if transmission.direction is not slot.direction:
            issues.append(
                ConformanceIssue(
                    "direction_mismatch",
                    f"expected {int(slot.direction)}, got {int(transmission.direction)}",
                    slot.slot_id,
                )
            )
        if transmission.wire_size_bytes != slot.bucket_bytes:
            issues.append(
                ConformanceIssue(
                    "bucket_mismatch",
                    f"expected {slot.bucket_bytes}, got {transmission.wire_size_bytes}",
                    slot.slot_id,
                )
            )
        if transmission.actual_size_bytes > transmission.wire_size_bytes:
            issues.append(
                ConformanceIssue(
                    "payload_exceeds_bucket",
                    "actual message or decoy is larger than the wire bucket",
                    slot.slot_id,
                )
            )
        if transmission.is_decoy:
            if transmission.decoy_id != slot.decoy.decoy_id:
                issues.append(
                    ConformanceIssue(
                        "undeclared_decoy",
                        "unused slot did not carry its declared safe decoy",
                        slot.slot_id,
                    )
                )
            if transmission.side_effecting:
                issues.append(
                    ConformanceIssue(
                        "side_effecting_decoy",
                        "a decoy must never be side-effecting",
                        slot.slot_id,
                    )
                )
        elif transmission.source_message_id is None:
            issues.append(
                ConformanceIssue(
                    "missing_real_message_id",
                    "a non-decoy transmission needs a source message ID",
                    slot.slot_id,
                )
            )
        elif not slot.accepts_kind(transmission.semantic_kind):
            issues.append(
                ConformanceIssue(
                    "semantic_kind_mismatch",
                    f"slot does not accept {transmission.semantic_kind!r}",
                    slot.slot_id,
                )
            )

    expected_termination = profile.public_start_epoch_s + profile.common_termination_s
    if not isclose(
        result.termination_at_s,
        expected_termination,
        rel_tol=0.0,
        abs_tol=timing_tolerance_s,
    ):
        issues.append(
            ConformanceIssue(
                "termination_mismatch",
                f"expected {expected_termination}, got {result.termination_at_s}",
            )
        )

    structurally_conformant = not issues
    return EnvelopeConformanceReport(
        structurally_conformant=structurally_conformant,
        fitted_execution=result.fitted,
        passes_fitted_envelope=structurally_conformant and result.fitted,
        issues=tuple(issues),
    )
