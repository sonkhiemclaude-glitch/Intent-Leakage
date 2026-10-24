"""Attacker-visible traffic features from paper Sections 3.2 and 5.6.

Only encrypted-traffic metadata is read.  The extractor intentionally ignores
plaintext, run identifiers, intent labels, phase markers, model response IDs,
and every other field not explicitly named below.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


PAPER_CHANNELS = ("CL", "CT", "CA")


@dataclass(frozen=True)
class NormalizedEvent:
    """Minimal attacker-visible event used internally by feature extraction."""

    timestamp_s: float
    length_bytes: int
    direction: int
    channel: str
    retransmission: bool = False


@dataclass(frozen=True)
class SequenceRepresentation:
    """Variable-length signed-size and inter-arrival sequences."""

    signed_sizes: np.ndarray
    inter_arrivals_s: np.ndarray

    def as_matrix(self) -> np.ndarray:
        return np.column_stack((self.signed_sizes, self.inter_arrivals_s))


@dataclass(frozen=True)
class BurstRepresentation:
    """Consecutive same-direction bursts for one ordered trace."""

    signed_bytes: np.ndarray
    durations_s: np.ndarray
    event_counts: np.ndarray


def _read_field(event: Any, names: Sequence[str], *, required: bool = True) -> Any:
    if isinstance(event, Mapping):
        for name in names:
            if name in event:
                return event[name]
    else:
        for name in names:
            if hasattr(event, name):
                return getattr(event, name)
    if required:
        raise ValueError(f"event is missing required field; expected one of {tuple(names)!r}")
    return None


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def normalize_channel(value: Any) -> str:
    """Normalize schema enum/string channel names to ``CL``, ``CT`` or ``CA``."""

    raw = str(_enum_value(value)).strip().upper().replace("-", "_")
    aliases = {
        "CL": "CL",
        "C_L": "CL",
        "L": "CL",
        "CT": "CT",
        "C_T": "CT",
        "T": "CT",
        "CA": "CA",
        "C_A": "CA",
        "A": "CA",
    }
    if raw not in aliases:
        raise ValueError(f"unknown protected-channel identifier: {value!r}")
    return aliases[raw]


def _unwrap_events(container: Any) -> list[Any]:
    if hasattr(container, "events"):
        return list(container.events)
    if hasattr(container, "traces"):
        traces = container.traces
        iterable = traces.values() if isinstance(traces, Mapping) else traces
        events: list[Any] = []
        for trace in iterable:
            events.extend(_unwrap_events(trace))
        return events
    if isinstance(container, Mapping) and "events" in container:
        return list(container["events"])
    if isinstance(container, Mapping) and "traces" in container:
        traces = container["traces"]
        iterable = traces.values() if isinstance(traces, Mapping) else traces
        events = []
        for trace in iterable:
            events.extend(_unwrap_events(trace))
        return events
    return list(container)


def normalize_events(events: Iterable[Any] | Any) -> list[NormalizedEvent]:
    """Validate and normalize trace events without consuming forbidden fields.

    Supported schema names include the repository contract
    ``timestamp_ns``, ``delta_time_s``, ``direction``, ``length_bytes`` and
    ``channel``.  Mapping inputs may use the documented aliases
    ``timestamp_s``/``timestamp`` and ``size_bytes``/``size``/``length``.
    """

    raw_events = _unwrap_events(events)
    normalized: list[NormalizedEvent] = []
    cumulative_time = 0.0
    for event in raw_events:
        timestamp_ns = _read_field(event, ("timestamp_ns",), required=False)
        timestamp_s = _read_field(event, ("timestamp_s", "timestamp"), required=False)
        delta_time = _read_field(event, ("delta_time_s", "delta_t"), required=False)
        if timestamp_ns is not None:
            timestamp = float(timestamp_ns) / 1_000_000_000.0
        elif timestamp_s is not None:
            timestamp = float(timestamp_s)
        elif delta_time is not None:
            delta = float(delta_time)
            if not np.isfinite(delta) or delta < 0.0:
                raise ValueError("delta time must be finite and nonnegative")
            cumulative_time += delta
            timestamp = cumulative_time
        else:
            raise ValueError("event requires timestamp_ns, timestamp_s, timestamp, or delta_time_s")
        raw_length = _read_field(
            event, ("length_bytes", "size_bytes", "length", "size")
        )
        if isinstance(raw_length, bool):
            raise ValueError("event lengths must be positive integers")
        try:
            numeric_length = float(raw_length)
        except (TypeError, ValueError) as error:
            raise ValueError("event lengths must be positive integers") from error
        if not np.isfinite(numeric_length) or not numeric_length.is_integer():
            raise ValueError("event lengths must be positive integers")
        length = int(numeric_length)
        direction = int(_enum_value(_read_field(event, ("direction",))))
        channel = normalize_channel(_read_field(event, ("channel",)))
        raw_retransmission = _read_field(
            event, ("retransmission",), required=False
        )
        if raw_retransmission is None:
            retransmission = False
        elif type(raw_retransmission) is bool:
            retransmission = raw_retransmission
        else:
            raise ValueError("retransmission must be a literal boolean")
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamps must be finite and nonnegative")
        if length <= 0:
            raise ValueError("event lengths must be positive integers")
        if direction not in (-1, 1):
            raise ValueError("event direction must be exactly -1 or +1")
        normalized.append(
            NormalizedEvent(
                timestamp_s=timestamp,
                length_bytes=length,
                direction=direction,
                channel=channel,
                retransmission=retransmission,
            )
        )
    normalized.sort(key=lambda item: item.timestamp_s)
    return normalized


def sequence_representation(
    events: Iterable[Any] | Any,
    *,
    channel: str | None = None,
) -> SequenceRepresentation:
    """Return the signed-size and inter-arrival sequences from Section 5.6."""

    normalized = normalize_events(events)
    if channel is not None:
        selected_channel = normalize_channel(channel)
        normalized = [event for event in normalized if event.channel == selected_channel]
    if not normalized:
        return SequenceRepresentation(
            signed_sizes=np.asarray([], dtype=float),
            inter_arrivals_s=np.asarray([], dtype=float),
        )
    times = np.asarray([event.timestamp_s for event in normalized], dtype=float)
    signed_sizes = np.asarray(
        [event.direction * event.length_bytes for event in normalized], dtype=float
    )
    inter_arrivals = np.diff(times, prepend=times[0])
    if np.any(inter_arrivals < -1e-12):
        raise ValueError("events must be ordered by nondecreasing timestamp")
    inter_arrivals[0] = 0.0
    return SequenceRepresentation(signed_sizes, np.maximum(inter_arrivals, 0.0))


def burst_representation(
    events: Iterable[Any] | Any,
    *,
    channel: str | None = None,
    burst_gap_s: float | None = None,
) -> BurstRepresentation:
    """Group consecutive same-direction records into deterministic bursts.

    A new burst starts on a direction change.  When ``burst_gap_s`` is set, a
    same-direction event after a larger idle gap also starts a new burst.  The
    gap rule is configuration, not a paper-derived constant.
    """

    if burst_gap_s is not None and (not np.isfinite(burst_gap_s) or burst_gap_s < 0.0):
        raise ValueError("burst_gap_s must be finite and nonnegative")
    normalized = normalize_events(events)
    if channel is not None:
        selected_channel = normalize_channel(channel)
        normalized = [event for event in normalized if event.channel == selected_channel]
    if not normalized:
        empty = np.asarray([], dtype=float)
        return BurstRepresentation(empty, empty.copy(), np.asarray([], dtype=int))

    signed_bytes: list[float] = []
    durations: list[float] = []
    counts: list[int] = []
    start = 0
    for index in range(1, len(normalized) + 1):
        boundary = index == len(normalized)
        if not boundary:
            previous = normalized[index - 1]
            current = normalized[index]
            boundary = current.direction != previous.direction
            if burst_gap_s is not None:
                boundary = boundary or current.timestamp_s - previous.timestamp_s > burst_gap_s
        if boundary:
            segment = normalized[start:index]
            direction = segment[0].direction
            signed_bytes.append(float(direction * sum(item.length_bytes for item in segment)))
            durations.append(float(segment[-1].timestamp_s - segment[0].timestamp_s))
            counts.append(len(segment))
            start = index
    return BurstRepresentation(
        signed_bytes=np.asarray(signed_bytes, dtype=float),
        durations_s=np.asarray(durations, dtype=float),
        event_counts=np.asarray(counts, dtype=int),
    )


def _summary(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(values)), float(np.std(values)), float(np.max(values))


def extract_channel_features(
    events: Iterable[Any] | Any,
    *,
    channel: str | None = None,
    prefix: str = "",
    burst_gap_s: float | None = None,
) -> dict[str, float]:
    """Extract fixed-dimensional aggregate and burst features for one view."""

    normalized = normalize_events(events)
    if channel is not None:
        selected_channel = normalize_channel(channel)
        normalized = [event for event in normalized if event.channel == selected_channel]
    sequence = sequence_representation(normalized)
    bursts = burst_representation(normalized, burst_gap_s=burst_gap_s)
    sizes = np.abs(sequence.signed_sizes)
    interarrival_mean, interarrival_std, interarrival_max = _summary(
        sequence.inter_arrivals_s[1:]
    )
    burst_size_mean, burst_size_std, burst_size_max = _summary(
        np.abs(bursts.signed_bytes)
    )
    burst_duration_mean, burst_duration_std, burst_duration_max = _summary(
        bursts.durations_s
    )
    if normalized:
        elapsed = normalized[-1].timestamp_s - normalized[0].timestamp_s
    else:
        elapsed = 0.0
    positive_bytes = sum(
        event.length_bytes for event in normalized if event.direction == 1
    )
    negative_bytes = sum(
        event.length_bytes for event in normalized if event.direction == -1
    )
    directions = np.asarray([event.direction for event in normalized], dtype=int)
    transitions = int(np.sum(np.diff(directions) != 0)) if directions.size > 1 else 0
    retransmissions = sum(event.retransmission for event in normalized)
    count = len(normalized)
    values = {
        "event_count": float(count),
        "elapsed_time_s": float(elapsed),
        "bytes_direction_pos": float(positive_bytes),
        "bytes_direction_neg": float(negative_bytes),
        "total_bytes": float(np.sum(sizes)),
        "direction_changes": float(transitions),
        "turn_count": float(transitions),
        "interarrival_mean_s": interarrival_mean,
        "interarrival_std_s": interarrival_std,
        "interarrival_max_s": interarrival_max,
        "burst_count": float(bursts.signed_bytes.size),
        "burst_abs_bytes_mean": burst_size_mean,
        "burst_abs_bytes_std": burst_size_std,
        "burst_abs_bytes_max": burst_size_max,
        "burst_duration_mean_s": burst_duration_mean,
        "burst_duration_std_s": burst_duration_std,
        "burst_duration_max_s": burst_duration_max,
        "retransmission_count": float(retransmissions),
        "retransmission_fraction": float(retransmissions / count) if count else 0.0,
    }
    return {f"{prefix}{name}": value for name, value in values.items()}


def extract_fused_features(
    events: Iterable[Any] | Any,
    *,
    channels: Sequence[str] = PAPER_CHANNELS,
    burst_gap_s: float | None = None,
) -> dict[str, float]:
    """Concatenate per-channel summaries and a common-clock fused summary."""

    normalized = normalize_events(events)
    result: dict[str, float] = {}
    for channel in channels:
        normalized_channel = normalize_channel(channel)
        result.update(
            extract_channel_features(
                normalized,
                channel=normalized_channel,
                prefix=f"{normalized_channel.lower()}__",
                burst_gap_s=burst_gap_s,
            )
        )
    result.update(
        extract_channel_features(
            normalized,
            prefix="fused__",
            burst_gap_s=burst_gap_s,
        )
    )
    return result


def padded_sequence(
    events: Iterable[Any] | Any,
    *,
    max_length: int,
    channel: str | None = None,
    truncate: str = "tail",
) -> np.ndarray:
    """Return a fixed ``(max_length, 2)`` signed-size/inter-arrival matrix."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    sequence = sequence_representation(events, channel=channel).as_matrix()
    if sequence.shape[0] > max_length:
        if truncate == "tail":
            sequence = sequence[:max_length]
        elif truncate == "head":
            sequence = sequence[-max_length:]
        else:
            raise ValueError("truncate must be 'tail' (keep prefix) or 'head' (keep suffix)")
    elif truncate not in {"tail", "head"}:
        raise ValueError("truncate must be 'tail' (keep prefix) or 'head' (keep suffix)")
    result = np.zeros((max_length, 2), dtype=float)
    result[: sequence.shape[0], :] = sequence
    return result


def fused_sequence_tensor(
    events: Iterable[Any] | Any,
    *,
    max_length: int,
    channels: Sequence[str] = PAPER_CHANNELS,
    truncate: str = "tail",
) -> np.ndarray:
    """Align channel sequences on the common event clock.

    The returned matrix has two columns per channel.  For each globally ordered
    event, its channel receives signed size and global inter-arrival time while
    the other channel columns remain zero.  This preserves channel identity and
    cross-channel ordering without using internal phase markers.
    """

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    normalized_channels = tuple(normalize_channel(channel) for channel in channels)
    if len(set(normalized_channels)) != len(normalized_channels):
        raise ValueError("channels must be unique")
    normalized = [
        event for event in normalize_events(events) if event.channel in normalized_channels
    ]
    if len(normalized) > max_length:
        if truncate == "tail":
            normalized = normalized[:max_length]
        elif truncate == "head":
            normalized = normalized[-max_length:]
        else:
            raise ValueError("truncate must be 'tail' or 'head'")
    elif truncate not in {"tail", "head"}:
        raise ValueError("truncate must be 'tail' or 'head'")
    result = np.zeros((max_length, 2 * len(normalized_channels)), dtype=float)
    previous_time = normalized[0].timestamp_s if normalized else 0.0
    channel_index = {channel: index for index, channel in enumerate(normalized_channels)}
    for row, event in enumerate(normalized):
        offset = 2 * channel_index[event.channel]
        result[row, offset] = event.direction * event.length_bytes
        result[row, offset + 1] = max(0.0, event.timestamp_s - previous_time)
        previous_time = event.timestamp_s
    return result


class TraceFeatureExtractor:
    """Sklearn-style fixed aggregate and sequence feature transformer."""

    def __init__(
        self,
        *,
        channels: Sequence[str] = PAPER_CHANNELS,
        max_sequence_length: int = 128,
        fused: bool = True,
        burst_gap_s: float | None = None,
    ) -> None:
        self.channels = tuple(normalize_channel(channel) for channel in channels)
        self.max_sequence_length = int(max_sequence_length)
        self.fused = bool(fused)
        self.burst_gap_s = burst_gap_s
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be unique")

    def extract_one(self, trace: Any) -> dict[str, float]:
        if self.fused:
            return extract_fused_features(
                trace, channels=self.channels, burst_gap_s=self.burst_gap_s
            )
        return extract_channel_features(trace, burst_gap_s=self.burst_gap_s)

    def transform(self, traces: Sequence[Any]) -> np.ndarray:
        """Transform traces into a deterministic aggregate feature matrix."""

        if not traces:
            raise ValueError("at least one trace is required")
        rows = [self.extract_one(trace) for trace in traces]
        names = tuple(rows[0])
        if any(tuple(row) != names for row in rows[1:]):
            raise RuntimeError("feature schema changed between traces")
        self.feature_names_in_order_ = names
        return np.asarray([[row[name] for name in names] for row in rows], dtype=float)

    def transform_sequences(self, traces: Sequence[Any]) -> np.ndarray:
        """Transform traces into fixed sequence tensors for k-NN/CNN attacks."""

        if not traces:
            raise ValueError("at least one trace is required")
        if self.fused:
            arrays = [
                fused_sequence_tensor(
                    trace,
                    max_length=self.max_sequence_length,
                    channels=self.channels,
                )
                for trace in traces
            ]
        else:
            arrays = [
                padded_sequence(trace, max_length=self.max_sequence_length)
                for trace in traces
            ]
        return np.stack(arrays)

    def get_feature_names_out(self) -> np.ndarray:
        if not hasattr(self, "feature_names_in_order_"):
            raise RuntimeError("transform must be called before get_feature_names_out")
        return np.asarray(self.feature_names_in_order_, dtype=object)
