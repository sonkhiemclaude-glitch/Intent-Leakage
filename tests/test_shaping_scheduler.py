from __future__ import annotations

from intent_leakage.schemas import (
    ApplicationMessage,
    Channel,
    Direction,
    IntentClass,
    ScopeBucket,
    Split,
)
from intent_leakage.data import generate_application_messages, generate_intent_instance
from intent_leakage.shaping import (
    FixedCadenceScheduler,
    IntentAwareShaper,
    OverflowReason,
    all_class_partition,
    make_uniform_envelope,
)


def _profile(*, rounds: int = 2, bucket: int = 256):
    return make_uniform_envelope(
        profile_id="test-envelope",
        group_id="all-classes",
        round_count=rounds,
        period_s=1.0,
        bucket_bytes=bucket,
        admitted_scopes=frozenset(ScopeBucket),
    )


def test_scheduler_emits_every_fixed_slot_and_safe_decoys() -> None:
    profile = _profile(rounds=2)
    messages = (
        ApplicationMessage(
            "write-1",
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            100,
            0.05,
            "tool_request",
            side_effecting=True,
        ),
        ApplicationMessage(
            "reply-1",
            Channel.TOOL,
            Direction.TOWARD_AGENT,
            120,
            0.20,
            "tool_response",
        ),
        ApplicationMessage(
            "assure-1",
            Channel.ASSURANCE,
            Direction.AWAY_FROM_AGENT,
            80,
            0.40,
            "assurance_request",
        ),
        ApplicationMessage(
            "assure-reply-1",
            Channel.ASSURANCE,
            Direction.TOWARD_AGENT,
            90,
            0.70,
            "assurance_response",
        ),
    )
    result = FixedCadenceScheduler().schedule(profile, messages)
    assert result.fitted
    assert len(result.transmissions) == len(profile.slots)
    assert [tx.scheduled_at_s for tx in result.transmissions] == [
        profile.absolute_slot_time(slot) for slot in profile.slots
    ]
    assert {tx.wire_size_bytes for tx in result.transmissions} == {256}
    assert sum(not tx.is_decoy for tx in result.transmissions) == len(messages)
    assert all(not tx.side_effecting for tx in result.transmissions if tx.is_decoy)
    assert result.termination_at_s == profile.common_termination_s


def test_scheduler_records_oversize_and_late_overflow_without_extension() -> None:
    profile = _profile(rounds=1, bucket=128)
    messages = (
        ApplicationMessage(
            "oversize",
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            129,
            0.01,
            "tool_request",
        ),
        ApplicationMessage(
            "late",
            Channel.ASSURANCE,
            Direction.AWAY_FROM_AGENT,
            64,
            2.0,
            "assurance_request",
        ),
    )
    result = FixedCadenceScheduler().schedule(profile, messages)
    assert not result.fitted
    assert len(result.transmissions) == len(profile.slots)
    assert result.termination_at_s == 1.0
    assert {event.reason for event in result.overflow_events} == {
        OverflowReason.BUCKET_TOO_SMALL,
        OverflowReason.MISSED_DEADLINE,
    }
    assert set(result.unscheduled_message_ids) == {"oversize", "late"}


def test_scheduler_preserves_real_state_changing_message_order() -> None:
    profile = _profile(rounds=2)
    messages = (
        ApplicationMessage(
            "first-write",
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            100,
            0.05,
            "tool_request",
            side_effecting=True,
        ),
        ApplicationMessage(
            "second-write",
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            100,
            0.05,
            "tool_request",
            side_effecting=True,
        ),
    )
    result = FixedCadenceScheduler().schedule(profile, messages)
    write_order = [
        tx.source_message_id
        for tx in result.transmissions
        if tx.side_effecting and not tx.is_decoy
    ]
    assert write_order == ["first-write", "second-write"]


def test_i15_uses_tool_decoys_even_with_zero_real_tool_calls() -> None:
    profile = _profile(rounds=4, bucket=512)
    intent = generate_intent_instance(
        # The paper explicitly permits zero real C_T calls for I15.
        class_id=IntentClass.I15,
        scope=ScopeBucket.MEDIUM,
        split=Split.TEST,
        topology_seed=9,
    )
    messages = tuple(
        message
        for message in generate_application_messages(intent, seed=9)
        if message.channel in profile.protected_channels
    )
    result = FixedCadenceScheduler().schedule(profile, messages)
    assert result.fitted
    tool_transmissions = [
        tx for tx in result.transmissions if tx.channel is Channel.TOOL
    ]
    assert tool_transmissions
    assert all(tx.is_decoy for tx in tool_transmissions)


def test_intent_aware_shaper_selects_group_and_rejects_unadmitted_scope() -> None:
    partition = all_class_partition(group_id="all-classes")
    profile = make_uniform_envelope(
        profile_id="low-only",
        group_id="all-classes",
        round_count=4,
        period_s=1.0,
        bucket_bytes=1024,
        admitted_scopes=frozenset({ScopeBucket.LOW}),
    )
    shaper = IntentAwareShaper(partition, {"all-classes": profile})

    low = generate_intent_instance(
        IntentClass.I01, ScopeBucket.LOW, Split.TEST, topology_seed=3
    )
    low_messages = tuple(
        message
        for message in generate_application_messages(low, seed=3)
        if message.channel in profile.protected_channels
    )
    assert shaper.shape(low, low_messages).fitted

    medium = generate_intent_instance(
        IntentClass.I01, ScopeBucket.MEDIUM, Split.TEST, topology_seed=3
    )
    overflow = shaper.shape(medium, ())
    assert not overflow.fitted
    assert overflow.overflow_events[0].reason is OverflowReason.SCOPE_NOT_ADMITTED
