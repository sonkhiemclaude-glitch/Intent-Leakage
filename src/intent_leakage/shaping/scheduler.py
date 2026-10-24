"""Deterministic fixed-cadence scheduling with explicit overflow records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intent_leakage.schemas import ApplicationMessage, Channel, Direction, OverflowPolicy
from intent_leakage.shaping.envelope import EnvelopeProfile, EnvelopeSlot


class OverflowReason(str, Enum):
    """Observable reasons that a real message did not fit its envelope."""

    UNPROTECTED_CHANNEL = "unprotected_channel"
    NO_COMPATIBLE_SLOT = "no_compatible_slot"
    BUCKET_TOO_SMALL = "bucket_too_small"
    MISSED_DEADLINE = "missed_deadline"
    EXTRA_SLOT_REQUIRED = "extra_slot_required"
    SCOPE_NOT_ADMITTED = "scope_not_admitted"


@dataclass(frozen=True, slots=True)
class ScheduledTransmission:
    """One transmission emitted at a public envelope slot."""

    slot_id: str
    scheduled_at_s: float
    channel: Channel
    direction: Direction
    wire_size_bytes: int
    actual_size_bytes: int
    semantic_kind: str
    is_decoy: bool
    source_message_id: str | None
    decoy_id: str | None
    side_effecting: bool


@dataclass(frozen=True, slots=True)
class OverflowEvent:
    """A machine-readable record of an unfit real message."""

    message_id: str
    reason: OverflowReason
    detail: str


@dataclass(frozen=True, slots=True)
class ShapingResult:
    """A complete fixed envelope plus any explicit overflow records."""

    profile_id: str
    overflow_policy: OverflowPolicy
    transmissions: tuple[ScheduledTransmission, ...]
    overflow_events: tuple[OverflowEvent, ...]
    unscheduled_message_ids: tuple[str, ...]
    termination_at_s: float

    @property
    def fitted(self) -> bool:
        """Return E=1 exactly when no real message overflowed."""

        return not self.overflow_events and not self.unscheduled_message_ids


class FixedCadenceScheduler:
    """Assign ready real messages to a frozen public envelope.

    Input ``ready_at_s`` values are relative to the same public transaction
    start as the profile.  The scheduler never adds an unscheduled round or
    enlarges a bucket.  Unused slots always carry declared safe decoys, and
    any message that does not fit is returned as explicit overflow data.
    """

    def schedule(
        self,
        profile: EnvelopeProfile,
        messages: tuple[ApplicationMessage, ...],
    ) -> ShapingResult:
        """Schedule ``messages`` without test-dependent profile changes."""

        message_ids = [message.message_id for message in messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("application message IDs must be unique")

        remaining = list(enumerate(messages))
        transmissions: list[ScheduledTransmission] = []

        for slot in profile.slots:
            relative_time = profile.relative_slot_time(slot)
            candidate_position = self._find_candidate(
                slot=slot,
                relative_time=relative_time,
                remaining=remaining,
            )
            if candidate_position is None:
                transmissions.append(self._decoy_transmission(profile, slot))
                continue

            _, message = remaining.pop(candidate_position)
            transmissions.append(
                ScheduledTransmission(
                    slot_id=slot.slot_id,
                    scheduled_at_s=profile.absolute_slot_time(slot),
                    channel=slot.channel,
                    direction=slot.direction,
                    wire_size_bytes=slot.bucket_bytes,
                    actual_size_bytes=message.size_bytes,
                    semantic_kind=message.semantic_kind,
                    is_decoy=False,
                    source_message_id=message.message_id,
                    decoy_id=None,
                    side_effecting=message.side_effecting,
                )
            )

        overflow_events = tuple(
            self._classify_overflow(profile, message) for _, message in remaining
        )
        unscheduled = tuple(message.message_id for _, message in remaining)
        return ShapingResult(
            profile_id=profile.profile_id,
            overflow_policy=profile.overflow_policy,
            transmissions=tuple(transmissions),
            overflow_events=overflow_events,
            unscheduled_message_ids=unscheduled,
            termination_at_s=(
                profile.public_start_epoch_s + profile.common_termination_s
            ),
        )

    @staticmethod
    def _find_candidate(
        *,
        slot: EnvelopeSlot,
        relative_time: float,
        remaining: list[tuple[int, ApplicationMessage]],
    ) -> int | None:
        """Find the first compatible message while preserving write order."""

        for candidate_position, (original_index, message) in enumerate(remaining):
            if message.ready_at_s > relative_time:
                continue
            if message.channel is not slot.channel or message.direction is not slot.direction:
                continue
            if not slot.accepts_kind(message.semantic_kind):
                continue
            if message.size_bytes > slot.bucket_bytes:
                continue
            if message.correlation_id is not None and any(
                earlier_message.correlation_id == message.correlation_id
                and earlier_index < original_index
                for earlier_index, earlier_message in remaining
            ):
                # Do not emit a response or dependent operation before the
                # earlier message in the same transaction has been released.
                continue
            if message.side_effecting and any(
                earlier_message.side_effecting and earlier_index < original_index
                for earlier_index, earlier_message in remaining
            ):
                continue
            return candidate_position
        return None

    @staticmethod
    def _decoy_transmission(
        profile: EnvelopeProfile,
        slot: EnvelopeSlot,
    ) -> ScheduledTransmission:
        return ScheduledTransmission(
            slot_id=slot.slot_id,
            scheduled_at_s=profile.absolute_slot_time(slot),
            channel=slot.channel,
            direction=slot.direction,
            wire_size_bytes=slot.bucket_bytes,
            actual_size_bytes=slot.decoy.nominal_size_bytes,
            semantic_kind="safe_decoy_read",
            is_decoy=True,
            source_message_id=None,
            decoy_id=slot.decoy.decoy_id,
            side_effecting=False,
        )

    @staticmethod
    def _classify_overflow(
        profile: EnvelopeProfile,
        message: ApplicationMessage,
    ) -> OverflowEvent:
        if message.channel not in profile.protected_channels:
            return OverflowEvent(
                message.message_id,
                OverflowReason.UNPROTECTED_CHANNEL,
                f"{message.channel.value} is not included in profile.protected_channels",
            )

        compatible = [
            slot
            for slot in profile.slots
            if slot.channel is message.channel
            and slot.direction is message.direction
            and slot.accepts_kind(message.semantic_kind)
        ]
        if not compatible:
            return OverflowEvent(
                message.message_id,
                OverflowReason.NO_COMPATIBLE_SLOT,
                "no slot matches channel, direction, and semantic kind",
            )
        if message.size_bytes > max(slot.bucket_bytes for slot in compatible):
            return OverflowEvent(
                message.message_id,
                OverflowReason.BUCKET_TOO_SMALL,
                f"message size {message.size_bytes} exceeds every compatible bucket",
            )
        last_compatible_time = max(profile.relative_slot_time(slot) for slot in compatible)
        if message.ready_at_s > last_compatible_time:
            return OverflowEvent(
                message.message_id,
                OverflowReason.MISSED_DEADLINE,
                f"ready_at_s={message.ready_at_s} is after the last compatible slot",
            )
        return OverflowEvent(
            message.message_id,
            OverflowReason.EXTRA_SLOT_REQUIRED,
            "all compatible slots were occupied or blocked by message ordering",
        )
