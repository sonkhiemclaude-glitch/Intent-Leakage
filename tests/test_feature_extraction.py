from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest

from intent_leakage.features import (
    TraceFeatureExtractor,
    burst_representation,
    extract_channel_features,
    extract_fused_features,
    fused_sequence_tensor,
    normalize_events,
    padded_sequence,
    sequence_representation,
)


class Channel(Enum):
    CL = "CL"
    CT = "CT"
    CA = "CA"


@dataclass(frozen=True)
class Event:
    timestamp_ns: int
    delta_time_s: float
    direction: int
    length_bytes: int
    channel: Channel
    retransmission: bool = False
    plaintext: str = "must never become a feature"
    run_id: str = "forbidden"


def sample_events() -> list[Event]:
    return [
        Event(0, 0.0, 1, 100, Channel.CL),
        Event(100_000_000, 0.1, 1, 50, Channel.CL),
        Event(200_000_000, 0.1, -1, 40, Channel.CT, retransmission=True),
        Event(350_000_000, 0.15, 1, 20, Channel.CT),
        Event(500_000_000, 0.15, -1, 10, Channel.CA),
    ]


def test_signed_size_interarrival_and_bursts() -> None:
    sequence = sequence_representation(sample_events())
    np.testing.assert_array_equal(sequence.signed_sizes, [100, 50, -40, 20, -10])
    np.testing.assert_allclose(sequence.inter_arrivals_s, [0.0, 0.1, 0.1, 0.15, 0.15])

    bursts = burst_representation(sample_events())
    np.testing.assert_array_equal(bursts.signed_bytes, [150, -40, 20, -10])
    np.testing.assert_allclose(bursts.durations_s, [0.1, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(bursts.event_counts, [2, 1, 1, 1])


def test_channel_and_fused_aggregate_features() -> None:
    cl = extract_channel_features(sample_events(), channel="C_L")
    assert cl["event_count"] == 2
    assert cl["total_bytes"] == 150
    assert cl["bytes_direction_pos"] == 150
    assert cl["bytes_direction_neg"] == 0
    assert cl["elapsed_time_s"] == pytest.approx(0.1)

    fused = extract_fused_features(sample_events())
    assert fused["cl__event_count"] == 2
    assert fused["ct__event_count"] == 2
    assert fused["ca__event_count"] == 1
    assert fused["fused__event_count"] == 5
    assert fused["fused__retransmission_count"] == 1
    assert all("plaintext" not in key and "run_id" not in key for key in fused)


def test_forbidden_metadata_does_not_change_features() -> None:
    first = [event.__dict__ for event in sample_events()]
    second = [dict(event, plaintext="different", run_id="different") for event in first]
    assert extract_fused_features(first) == extract_fused_features(second)


def test_padded_and_common_clock_fused_sequences() -> None:
    padded = padded_sequence(sample_events(), max_length=4, channel="CL")
    assert padded.shape == (4, 2)
    np.testing.assert_allclose(padded[:2], [[100, 0.0], [50, 0.1]])
    np.testing.assert_array_equal(padded[2:], 0.0)

    fused = fused_sequence_tensor(sample_events(), max_length=6)
    assert fused.shape == (6, 6)
    np.testing.assert_allclose(fused[0], [100, 0.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(fused[2], [0.0, 0.0, -40, 0.1, 0.0, 0.0])
    np.testing.assert_allclose(fused[4], [0.0, 0.0, 0.0, 0.0, -10, 0.15])


def test_trace_feature_extractor_has_stable_schema() -> None:
    extractor = TraceFeatureExtractor(max_sequence_length=8, fused=True)
    aggregate = extractor.transform([sample_events(), sample_events()])
    sequence = extractor.transform_sequences([sample_events(), sample_events()])
    assert aggregate.shape[0] == 2
    assert sequence.shape == (2, 8, 6)
    assert aggregate.shape[1] == extractor.get_feature_names_out().size
    np.testing.assert_array_equal(aggregate[0], aggregate[1])


def test_event_validation_and_delta_only_input() -> None:
    delta_only = [
        {"delta_time_s": 0.0, "direction": 1, "length": 4, "channel": "CT"},
        {"delta_time_s": 0.25, "direction": -1, "length": 5, "channel": "CT"},
    ]
    normalized = normalize_events(delta_only)
    assert [event.timestamp_s for event in normalized] == [0.0, 0.25]
    with pytest.raises(ValueError, match="direction"):
        normalize_events(
            [{"timestamp_s": 0.0, "direction": 0, "length": 1, "channel": "CL"}]
        )
    with pytest.raises(ValueError, match="positive integers"):
        normalize_events(
            [{"timestamp_s": 0.0, "direction": 1, "length": -1, "channel": "CL"}]
        )
    with pytest.raises(ValueError, match="positive integers"):
        normalize_events(
            [{"timestamp_s": 0.0, "direction": 1, "length": 0, "channel": "CL"}]
        )
    with pytest.raises(ValueError, match="positive integers"):
        normalize_events(
            [{"timestamp_s": 0.0, "direction": 1, "length": 1.5, "channel": "CL"}]
        )
    with pytest.raises(ValueError, match="literal boolean"):
        normalize_events(
            [
                {
                    "timestamp_s": 0.0,
                    "direction": 1,
                    "length": 1,
                    "channel": "CL",
                    "retransmission": "false",
                }
            ]
        )


def test_repository_trace_schema_interoperability() -> None:
    from intent_leakage.schemas import (
        Channel as SchemaChannel,
        ChannelTrace,
        Direction,
        FusedTrace,
        TraceEvent,
    )

    llm_trace = ChannelTrace(
        channel=SchemaChannel.LLM,
        events=(
            TraceEvent(0.0, SchemaChannel.LLM, Direction.AWAY_FROM_AGENT, 100),
            TraceEvent(0.1, SchemaChannel.LLM, Direction.TOWARD_AGENT, 200),
        ),
    )
    tool_trace = ChannelTrace(
        channel=SchemaChannel.TOOL,
        events=(TraceEvent(0.05, SchemaChannel.TOOL, Direction.AWAY_FROM_AGENT, 50),),
    )
    features = extract_fused_features(FusedTrace((llm_trace, tool_trace)))
    assert features["cl__event_count"] == 2
    assert features["ct__event_count"] == 1
    assert features["fused__total_bytes"] == 350
