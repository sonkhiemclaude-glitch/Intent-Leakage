"""Fixed-size per-message padding baseline from Section 5.5."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intent_leakage.schemas import (
    ApplicationMessage,
    Channel,
    ChannelTrace,
    FusedTrace,
    TraceEvent,
)


@dataclass(frozen=True, slots=True)
class MessagePaddingResult:
    """Padded messages and explicit IDs that exceeded the fixed bucket."""

    messages: tuple[ApplicationMessage, ...]
    oversize_message_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TracePaddingResult:
    """Padded trace and event positions that exceeded the fixed bucket."""

    trace: FusedTrace
    oversize_events: tuple[tuple[Channel, int], ...]


class FixedSizePaddingBaseline:
    """Pad each protected message independently without transaction shaping.

    Message count, direction, timing, cadence, and termination are preserved.
    A message larger than the configured training-derived bucket cannot be
    shrunk; it is left at its original size and recorded explicitly as
    oversize.  No decoy is ever inserted.
    """

    def __init__(
        self,
        bucket_bytes: int,
        protected_channels: frozenset[Channel] = frozenset(
            {Channel.TOOL, Channel.ASSURANCE}
        ),
    ) -> None:
        if bucket_bytes <= 0:
            raise ValueError("bucket_bytes must be positive")
        if not protected_channels:
            raise ValueError("protected_channels must not be empty")
        self.bucket_bytes = bucket_bytes
        self.protected_channels = protected_channels

    def apply_messages(
        self,
        messages: tuple[ApplicationMessage, ...],
    ) -> MessagePaddingResult:
        """Apply fixed padding while retaining all non-size fields."""

        padded: list[ApplicationMessage] = []
        oversize: list[str] = []
        for message in messages:
            if message.channel not in self.protected_channels:
                padded.append(message)
            elif message.size_bytes <= self.bucket_bytes:
                padded.append(replace(message, size_bytes=self.bucket_bytes))
            else:
                padded.append(message)
                oversize.append(message.message_id)
        return MessagePaddingResult(tuple(padded), tuple(oversize))

    def apply_channel_trace(
        self,
        trace: ChannelTrace,
    ) -> tuple[ChannelTrace, tuple[int, ...]]:
        """Pad one trace and return zero-based oversize event indices."""

        if trace.channel not in self.protected_channels:
            return trace, ()
        events: list[TraceEvent] = []
        oversize: list[int] = []
        for index, event in enumerate(trace.events):
            if event.length_bytes <= self.bucket_bytes:
                events.append(replace(event, length_bytes=self.bucket_bytes))
            else:
                events.append(event)
                oversize.append(index)
        return (
            ChannelTrace(
                channel=trace.channel,
                events=tuple(events),
                oracle_segmented=trace.oracle_segmented,
            ),
            tuple(oversize),
        )

    def apply_fused_trace(self, trace: FusedTrace) -> TracePaddingResult:
        """Apply padding independently to every protected channel."""

        padded_traces: list[ChannelTrace] = []
        oversize: list[tuple[Channel, int]] = []
        for channel_trace in trace.traces:
            padded_trace, indices = self.apply_channel_trace(channel_trace)
            padded_traces.append(padded_trace)
            oversize.extend((channel_trace.channel, index) for index in indices)
        return TracePaddingResult(FusedTrace(tuple(padded_traces)), tuple(oversize))
