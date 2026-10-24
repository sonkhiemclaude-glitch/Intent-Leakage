"""Fixed-size, fixed-cadence envelope definitions from Section 4."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from intent_leakage.exceptions import EnvelopeDefinitionError, UnsafeDecoyError
from intent_leakage.schemas import Channel, Direction, OverflowPolicy, ScopeBucket


@dataclass(frozen=True, slots=True)
class SafeDecoy:
    """A declared side-effect-free operation used in an unused slot."""

    decoy_id: str
    operation: str
    target: str
    nominal_size_bytes: int
    authenticated: bool = True
    authorized: bool = True
    read_only: bool = True
    side_effect_free: bool = True
    excluded_from_truth_criterion: bool = True

    def __post_init__(self) -> None:
        if not self.decoy_id.strip() or not self.operation.strip() or not self.target.strip():
            raise UnsafeDecoyError("decoy ID, operation, and target must not be empty")
        if self.nominal_size_bytes <= 0:
            raise UnsafeDecoyError("decoy nominal_size_bytes must be positive")
        safety_flags = (
            self.authenticated,
            self.authorized,
            self.read_only,
            self.side_effect_free,
            self.excluded_from_truth_criterion,
        )
        if any(type(flag) is not bool for flag in safety_flags) or not all(safety_flags):
            raise UnsafeDecoyError(
                "decoy safety flags must be literal booleans and must declare the "
                "operation authenticated, authorized, read-only, side-effect free, "
                "and excluded from the truth criterion"
            )


@dataclass(frozen=True, slots=True)
class EnvelopeSlot:
    """One public request or response slot in a group envelope."""

    slot_id: str
    round_index: int
    slot_index: int
    offset_in_round_s: float
    channel: Channel
    direction: Direction
    bucket_bytes: int
    allowed_semantic_kinds: frozenset[str]
    decoy: SafeDecoy

    def __post_init__(self) -> None:
        if not self.slot_id.strip():
            raise EnvelopeDefinitionError("slot_id must not be empty")
        if self.round_index < 0 or self.slot_index < 0:
            raise EnvelopeDefinitionError("round_index and slot_index must be nonnegative")
        if not isfinite(self.offset_in_round_s) or self.offset_in_round_s < 0:
            raise EnvelopeDefinitionError(
                "offset_in_round_s must be finite and nonnegative"
            )
        if self.bucket_bytes <= 0:
            raise EnvelopeDefinitionError("bucket_bytes must be positive")
        if self.decoy.nominal_size_bytes > self.bucket_bytes:
            raise EnvelopeDefinitionError("decoy does not fit its slot bucket")
        if any(not kind.strip() for kind in self.allowed_semantic_kinds):
            raise EnvelopeDefinitionError("semantic-kind names must not be empty")

    def accepts_kind(self, semantic_kind: str) -> bool:
        """Return whether this slot accepts a real message's semantic kind."""

        return not self.allowed_semantic_kinds or semantic_kind in self.allowed_semantic_kinds


@dataclass(frozen=True, slots=True)
class EnvelopeProfile:
    """The complete public profile for one shaping group."""

    profile_id: str
    group_id: str
    admitted_scopes: frozenset[ScopeBucket]
    round_count: int
    period_s: float
    slots: tuple[EnvelopeSlot, ...]
    common_termination_s: float
    protected_channels: frozenset[Channel]
    overflow_policy: OverflowPolicy = OverflowPolicy.FAIL_CLOSED
    public_start_epoch_s: float = 0.0
    start_rule: str = "fixed public experiment epoch"

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.group_id.strip():
            raise EnvelopeDefinitionError("profile_id and group_id must not be empty")
        if not self.start_rule.strip():
            raise EnvelopeDefinitionError("start_rule must not be empty")
        if not self.admitted_scopes:
            raise EnvelopeDefinitionError("admitted_scopes must not be empty")
        if self.round_count <= 0:
            raise EnvelopeDefinitionError("round_count must be positive")
        if not isfinite(self.period_s) or self.period_s <= 0:
            raise EnvelopeDefinitionError("period_s must be finite and positive")
        if not isfinite(self.public_start_epoch_s) or self.public_start_epoch_s < 0:
            raise EnvelopeDefinitionError(
                "public_start_epoch_s must be finite and nonnegative"
            )
        if not isfinite(self.common_termination_s) or self.common_termination_s <= 0:
            raise EnvelopeDefinitionError(
                "common_termination_s must be finite and positive"
            )
        if not self.protected_channels:
            raise EnvelopeDefinitionError("protected_channels must not be empty")
        if not self.slots:
            raise EnvelopeDefinitionError("an envelope must contain at least one slot")

        keys = [(slot.round_index, slot.slot_index) for slot in self.slots]
        if len(keys) != len(set(keys)):
            raise EnvelopeDefinitionError("round/slot indices must be unique")
        slot_ids = [slot.slot_id for slot in self.slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise EnvelopeDefinitionError("slot IDs must be unique")
        for slot in self.slots:
            if slot.round_index >= self.round_count:
                raise EnvelopeDefinitionError("slot round exceeds round_count")
            if slot.offset_in_round_s >= self.period_s:
                raise EnvelopeDefinitionError("slot offset must be less than period_s")
            if slot.channel not in self.protected_channels:
                raise EnvelopeDefinitionError(
                    f"slot channel {slot.channel.value} is not protected by the profile"
                )

        chronological = sorted(
            self.slots,
            key=lambda slot: (
                slot.round_index,
                slot.offset_in_round_s,
                slot.slot_index,
            ),
        )
        if list(self.slots) != chronological:
            raise EnvelopeDefinitionError("slots must be declared in chronological order")
        if set(slot.round_index for slot in self.slots) != set(range(self.round_count)):
            raise EnvelopeDefinitionError("every declared round must contain at least one slot")
        last_slot_s = max(self.relative_slot_time(slot) for slot in self.slots)
        if self.common_termination_s < last_slot_s:
            raise EnvelopeDefinitionError("common termination precedes the final slot")

    def relative_slot_time(self, slot: EnvelopeSlot) -> float:
        """Return a slot time relative to the public start epoch."""

        return slot.round_index * self.period_s + slot.offset_in_round_s

    def absolute_slot_time(self, slot: EnvelopeSlot) -> float:
        """Return the public common-clock emission time for ``slot``."""

        return self.public_start_epoch_s + self.relative_slot_time(slot)


def make_uniform_envelope(
    *,
    profile_id: str,
    group_id: str,
    round_count: int,
    period_s: float,
    bucket_bytes: int,
    admitted_scopes: frozenset[ScopeBucket] = frozenset(ScopeBucket),
    include_tool: bool = True,
    include_assurance: bool = True,
    public_start_epoch_s: float = 0.0,
) -> EnvelopeProfile:
    """Construct a deterministic demonstration profile.

    This helper does not claim that its parameters were fitted to real traces.
    It exists for tests and synthetic smoke runs.  Production profiles must be
    fitted from training captures, assigned a declared safety margin, and
    frozen before test collection.
    """

    if not include_tool and not include_assurance:
        raise EnvelopeDefinitionError("at least one channel must be included")
    slot_specs: list[tuple[float, Channel, Direction, str]] = []
    if include_tool:
        slot_specs.extend(
            (
                (0.10, Channel.TOOL, Direction.AWAY_FROM_AGENT, "tool_request"),
                (0.30, Channel.TOOL, Direction.TOWARD_AGENT, "tool_response"),
            )
        )
    if include_assurance:
        slot_specs.extend(
            (
                (0.60, Channel.ASSURANCE, Direction.AWAY_FROM_AGENT, "assurance_request"),
                (0.80, Channel.ASSURANCE, Direction.TOWARD_AGENT, "assurance_response"),
            )
        )
    slots: list[EnvelopeSlot] = []
    for round_index in range(round_count):
        for slot_index, (fraction, channel, direction, semantic_kind) in enumerate(
            sorted(slot_specs, key=lambda item: item[0])
        ):
            slot_id = f"r{round_index:03d}-s{slot_index:02d}"
            slots.append(
                EnvelopeSlot(
                    slot_id=slot_id,
                    round_index=round_index,
                    slot_index=slot_index,
                    offset_in_round_s=fraction * period_s,
                    channel=channel,
                    direction=direction,
                    bucket_bytes=bucket_bytes,
                    allowed_semantic_kinds=frozenset({semantic_kind}),
                    decoy=SafeDecoy(
                        decoy_id=f"decoy-{slot_id}",
                        operation="read_predeclared_safe_object",
                        target="synthetic-safe-object",
                        nominal_size_bytes=min(64, bucket_bytes),
                    ),
                )
            )
    protected_channels = frozenset(slot.channel for slot in slots)
    return EnvelopeProfile(
        profile_id=profile_id,
        group_id=group_id,
        admitted_scopes=admitted_scopes,
        round_count=round_count,
        period_s=period_s,
        slots=tuple(slots),
        common_termination_s=round_count * period_s,
        protected_channels=protected_channels,
        public_start_epoch_s=public_start_epoch_s,
    )
