"""Offline adapters for manually provisioned, pre-extracted trace JSONL.

The adapter performs no download and no packet parsing.  It accepts the
documented event schema only after a separately validated PCAP/TCP/TLS
reassembly tool has produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from intent_leakage.schemas import Channel, ChannelTrace, Direction, FusedTrace, TraceEvent


FORBIDDEN_ATTACKER_EVENT_FIELDS = frozenset(
    {
        "class_id",
        "scope",
        "scope_bucket",
        "targets",
        "natural_language_intent",
        "plaintext",
        "run_id",
        "phase_marker",
        "model_response_id",
    }
)


@dataclass(frozen=True, slots=True)
class LoadedTraceDataset:
    """Attacker-visible traces plus non-feature segmentation provenance."""

    traces: Mapping[str, FusedTrace]
    segmentation_modes: Mapping[str, str]
    source_path: Path


def _channel(value: Any) -> Channel:
    normalized = str(value).strip().upper().replace("_", "")
    aliases = {"CL": Channel.LLM, "CT": Channel.TOOL, "CA": Channel.ASSURANCE}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown channel {value!r}; expected CL, CT, or CA") from exc


def _direction(value: Any) -> Direction:
    try:
        return Direction(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"direction must be -1 or 1, got {value!r}") from exc


def _timestamp_seconds(row: Mapping[str, Any]) -> float:
    has_seconds = "timestamp_s" in row
    has_nanoseconds = "timestamp_ns" in row
    if has_seconds == has_nanoseconds:
        raise ValueError("each event must contain exactly one of timestamp_s or timestamp_ns")
    return (
        float(row["timestamp_s"])
        if has_seconds
        else float(row["timestamp_ns"]) / 1_000_000_000.0
    )


def _jsonl_rows(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"trace JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, start=1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, row


def load_trace_events_jsonl(
    path: str | Path,
    *,
    reject_ground_truth_fields: bool = True,
) -> LoadedTraceDataset:
    """Load the documented attacker-visible event JSONL format.

    Ground-truth fields are rejected by default so a capture conversion cannot
    silently place plaintext labels or internal markers into an attacker's
    observation.  Segmentation mode is retained as provenance but is not copied
    into ``TraceEvent.metadata``.
    """

    source = Path(path)
    grouped: dict[str, dict[Channel, list[TraceEvent]]] = {}
    modes: dict[str, str] = {}
    required = {"trace_id", "channel", "direction", "length_bytes"}
    allowed_modes = {"passive", "endpoint_separable", "combined", "oracle_segmented"}

    for line_number, row in _jsonl_rows(source):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{source}:{line_number} missing fields: {sorted(missing)}")
        forbidden = FORBIDDEN_ATTACKER_EVENT_FIELDS.intersection(row)
        if reject_ground_truth_fields and forbidden:
            raise ValueError(
                f"{source}:{line_number} contains forbidden attacker fields: {sorted(forbidden)}"
            )
        trace_id = str(row["trace_id"]).strip()
        if not trace_id:
            raise ValueError(f"{source}:{line_number} has an empty trace_id")
        channel = _channel(row["channel"])
        mode = str(row.get("segmentation_mode", "combined")).strip()
        if mode not in allowed_modes:
            raise ValueError(
                f"{source}:{line_number} has invalid segmentation_mode {mode!r}"
            )
        prior_mode = modes.setdefault(trace_id, mode)
        if prior_mode != mode:
            raise ValueError(f"trace {trace_id!r} mixes segmentation modes")
        if mode == "combined" and channel in {Channel.TOOL, Channel.ASSURANCE}:
            # A non-separable protected capture must not expose an artificial
            # CT-versus-CA label to feature extraction.  Collapse both endpoint
            # labels into one protected stream; the provenance mode preserves
            # the fact that phase-specific reporting is unavailable.
            channel = Channel.TOOL
        known = required | {
            "timestamp_s",
            "timestamp_ns",
            "flow_id",
            "retransmission",
            "transport_unit",
            "segmentation_mode",
        }
        metadata = {key: value for key, value in row.items() if key not in known}
        retransmission = row.get("retransmission", False)
        if not isinstance(retransmission, bool):
            raise ValueError(
                f"{source}:{line_number} retransmission must be a JSON boolean"
            )
        length_bytes = row["length_bytes"]
        if not isinstance(length_bytes, int) or isinstance(length_bytes, bool):
            raise ValueError(f"{source}:{line_number} length_bytes must be an integer")
        event = TraceEvent(
            timestamp_s=_timestamp_seconds(row),
            channel=channel,
            direction=_direction(row["direction"]),
            length_bytes=length_bytes,
            flow_id=str(row.get("flow_id", "default")),
            retransmission=retransmission,
            transport_unit=str(row.get("transport_unit", "tls_record")),
            metadata=metadata,
        )
        grouped.setdefault(trace_id, {}).setdefault(channel, []).append(event)

    if not grouped:
        raise ValueError(f"trace JSONL contains no events: {source}")
    traces: dict[str, FusedTrace] = {}
    for trace_id, by_channel in grouped.items():
        for channel, events in by_channel.items():
            timestamps = [event.timestamp_s for event in events]
            if timestamps != sorted(timestamps):
                raise ValueError(
                    f"trace {trace_id!r} channel {channel.value} is not chronological"
                )
        channels = tuple(
            ChannelTrace(
                channel=channel,
                events=tuple(events),
                oracle_segmented=modes[trace_id] == "oracle_segmented",
            )
            for channel, events in sorted(by_channel.items(), key=lambda item: item[0].value)
        )
        traces[trace_id] = FusedTrace(channels)
    return LoadedTraceDataset(traces=traces, segmentation_modes=modes, source_path=source)


def load_dataset_root(
    root: str | Path,
    *,
    events_relative_path: str | Path = "traces/events.jsonl",
) -> LoadedTraceDataset:
    """Load a configured event JSONL beneath ``DATA_ROOT`` without networking."""

    from ..utils.validation import ensure_relative_to

    data_root = Path(root).expanduser().resolve()
    source = ensure_relative_to(data_root / events_relative_path, data_root)
    return load_trace_events_jsonl(source)
