from __future__ import annotations

import pytest

from intent_leakage.baselines import FixedSizePaddingBaseline, NoShapingBaseline
from intent_leakage.data import generate_intent_instance
from intent_leakage.exceptions import SchemaValidationError
from intent_leakage.schemas import (
    ApplicationMessage,
    Channel,
    ChannelTrace,
    Direction,
    IntentClass,
    ScopeBucket,
    Split,
    TraceEvent,
)


def test_scope_mapping_is_fixed_to_one_two_four() -> None:
    assert ScopeBucket.LOW.q == 1
    assert ScopeBucket.MEDIUM.q == 2
    assert ScopeBucket.HIGH.q == 4
    assert ScopeBucket.from_q(4) is ScopeBucket.HIGH
    with pytest.raises(SchemaValidationError):
        ScopeBucket.from_q(3)


def test_intent_requires_scope_q_and_target_count_to_agree() -> None:
    valid = generate_intent_instance(
        IntentClass.I01, ScopeBucket.MEDIUM, Split.TRAIN
    )
    with pytest.raises(SchemaValidationError, match="exactly q=2"):
        type(valid)(
            instance_id=valid.instance_id,
            class_id=valid.class_id,
            scope=valid.scope,
            q=valid.q,
            targets=("only-one",),
            topology_id=valid.topology_id,
            topology_seed=valid.topology_seed,
            phrase_family_id=valid.phrase_family_id,
            split=valid.split,
            natural_language_intent=valid.natural_language_intent,
            expected_postcondition=valid.expected_postcondition,
            inverse_cleanup_operation=valid.inverse_cleanup_operation,
        )


def test_channel_trace_validates_order_and_exposes_equation_one_sequences() -> None:
    first = TraceEvent(0.5, Channel.TOOL, Direction.AWAY_FROM_AGENT, 100)
    second = TraceEvent(0.8, Channel.TOOL, Direction.TOWARD_AGENT, 200)
    trace = ChannelTrace(Channel.TOOL, (first, second))
    assert trace.signed_sizes == (100, -200)
    assert trace.interarrival_s == pytest.approx((0.0, 0.3))
    with pytest.raises(SchemaValidationError, match="chronological"):
        ChannelTrace(Channel.TOOL, (second, first))


def test_trace_event_rejects_coercible_invalid_types() -> None:
    with pytest.raises(SchemaValidationError, match="positive integer"):
        TraceEvent(0.0, Channel.TOOL, Direction.AWAY_FROM_AGENT, True)
    with pytest.raises(SchemaValidationError, match="positive integer"):
        TraceEvent(0.0, Channel.TOOL, Direction.AWAY_FROM_AGENT, 1.5)  # type: ignore[arg-type]
    with pytest.raises(SchemaValidationError, match="literal boolean"):
        TraceEvent(
            0.0,
            Channel.TOOL,
            Direction.AWAY_FROM_AGENT,
            1,
            retransmission="false",  # type: ignore[arg-type]
        )


def test_baselines_preserve_count_timing_direction_and_identity() -> None:
    messages = (
        ApplicationMessage(
            "m1", Channel.TOOL, Direction.AWAY_FROM_AGENT, 90, 0.1, "tool_request"
        ),
        ApplicationMessage(
            "m2", Channel.TOOL, Direction.TOWARD_AGENT, 140, 0.2, "tool_response"
        ),
        ApplicationMessage(
            "m3", Channel.LLM, Direction.AWAY_FROM_AGENT, 70, 0.3, "llm_request"
        ),
    )
    assert NoShapingBaseline.apply_messages(messages) is messages

    result = FixedSizePaddingBaseline(bucket_bytes=128).apply_messages(messages)
    assert len(result.messages) == len(messages)
    assert [item.ready_at_s for item in result.messages] == [0.1, 0.2, 0.3]
    assert [item.direction for item in result.messages] == [
        Direction.AWAY_FROM_AGENT,
        Direction.TOWARD_AGENT,
        Direction.AWAY_FROM_AGENT,
    ]
    assert [item.size_bytes for item in result.messages] == [128, 140, 70]
    assert result.oversize_message_ids == ("m2",)
