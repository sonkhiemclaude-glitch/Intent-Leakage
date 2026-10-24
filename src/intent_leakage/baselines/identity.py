"""No-shaping baseline from Section 5.5."""

from __future__ import annotations

from intent_leakage.schemas import ApplicationMessage, ChannelTrace, FusedTrace


class NoShapingBaseline:
    """Return inputs unchanged, modeling shims in bypass mode."""

    @staticmethod
    def apply_messages(
        messages: tuple[ApplicationMessage, ...],
    ) -> tuple[ApplicationMessage, ...]:
        return messages

    @staticmethod
    def apply_channel_trace(trace: ChannelTrace) -> ChannelTrace:
        return trace

    @staticmethod
    def apply_fused_trace(trace: FusedTrace) -> FusedTrace:
        return trace
