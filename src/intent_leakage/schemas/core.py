"""Validated data objects for the intent-leakage experiments.

These schemas encode Sections 3--5 of the paper.  They intentionally keep
plaintext intent data separate from attacker-visible trace events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from math import isfinite
from typing import Any, Mapping

from intent_leakage.exceptions import SchemaValidationError


SYNTHETIC_OUTPUT_NOTICE = "synthetic smoke-test output, not a paper result"


class IntentClass(str, Enum):
    """The predeclared intent catalogue from Table 3."""

    I01 = "I01"
    I02 = "I02"
    I03 = "I03"
    I04 = "I04"
    I05 = "I05"
    I06 = "I06"
    I07 = "I07"
    I08 = "I08"
    I09 = "I09"
    I10 = "I10"
    I11 = "I11"
    I12 = "I12"
    I13 = "I13"
    I14 = "I14"
    I15 = "I15"


class ScopeBucket(str, Enum):
    """The ordered scope buckets from Section 5.2."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def q(self) -> int:
        """Return the number of primary targets represented by this bucket."""

        return {
            ScopeBucket.LOW: 1,
            ScopeBucket.MEDIUM: 2,
            ScopeBucket.HIGH: 4,
        }[self]

    @classmethod
    def from_q(cls, q: int) -> "ScopeBucket":
        """Construct a bucket from the paper's fixed q in {1, 2, 4}."""

        try:
            return {1: cls.LOW, 2: cls.MEDIUM, 4: cls.HIGH}[q]
        except KeyError as exc:
            raise SchemaValidationError("q must be one of 1, 2, or 4") from exc


class Split(str, Enum):
    """Phrase-family split fixed before collection."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Channel(str, Enum):
    """Observable channels from Section 3.2."""

    LLM = "CL"
    TOOL = "CT"
    ASSURANCE = "CA"

    @property
    def paper_symbol(self) -> str:
        """Return the typeset-equivalent notation used by the paper."""

        return {
            Channel.LLM: "C_L",
            Channel.TOOL: "C_T",
            Channel.ASSURANCE: "C_A",
        }[self]


class Direction(IntEnum):
    """Direction relative to the agent, as used in Equation (1)."""

    TOWARD_AGENT = -1
    AWAY_FROM_AGENT = 1


class DefenseCondition(str, Enum):
    """Defense conditions in Section 5.5 and Table 4."""

    NO_SHAPING = "no_shaping"
    FIXED_SIZE_PADDING = "fixed_size_padding"
    ALL_CLASS_ENVELOPE = "all_class_envelope"
    BUDGETED_GROUPS = "budgeted_groups"


class OverflowPolicy(str, Enum):
    """Predeclared policies permitted by Section 4.2."""

    FAIL_CLOSED = "fail_closed"
    LARGER_PUBLIC_PROFILE = "larger_public_profile"
    UNPROTECTED_EMERGENCY = "unprotected_emergency"


@dataclass(frozen=True, slots=True)
class IntentDefinition:
    """One row of the predeclared intent catalogue (Table 3)."""

    class_id: IntentClass
    name: str
    primary_target_type: str
    assurance_assertion: str
    implementation_note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, IntentClass):
            raise SchemaValidationError("class_id must be an IntentClass")
        for field_name in ("name", "primary_target_type", "assurance_assertion"):
            if not getattr(self, field_name).strip():
                raise SchemaValidationError(f"{field_name} must not be empty")


@dataclass(frozen=True, slots=True)
class IntentInstance:
    """A declarative intent instance used as experiment ground truth.

    The target count is required to agree with both ``q`` and ``scope``.  This
    prevents a generated or loaded manifest from silently changing the scope
    definition after observing traces.
    """

    instance_id: str
    class_id: IntentClass
    scope: ScopeBucket
    q: int
    targets: tuple[str, ...]
    topology_id: str
    topology_seed: int
    phrase_family_id: str
    split: Split
    natural_language_intent: str
    expected_postcondition: str
    inverse_cleanup_operation: str
    compliance_state: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, IntentClass):
            raise SchemaValidationError("class_id must be an IntentClass")
        if not isinstance(self.scope, ScopeBucket):
            raise SchemaValidationError("scope must be a ScopeBucket")
        if not isinstance(self.split, Split):
            raise SchemaValidationError("split must be a Split")
        if self.q not in {1, 2, 4}:
            raise SchemaValidationError("q must be one of 1, 2, or 4")
        if self.scope.q != self.q:
            raise SchemaValidationError(
                f"scope {self.scope.value!r} requires q={self.scope.q}, got {self.q}"
            )
        if len(self.targets) != self.q:
            raise SchemaValidationError(
                f"targets must contain exactly q={self.q} primary targets"
            )
        if len(set(self.targets)) != len(self.targets):
            raise SchemaValidationError("primary targets must be unique")
        for field_name in (
            "instance_id",
            "topology_id",
            "phrase_family_id",
            "natural_language_intent",
            "expected_postcondition",
            "inverse_cleanup_operation",
        ):
            if not getattr(self, field_name).strip():
                raise SchemaValidationError(f"{field_name} must not be empty")
        if self.class_id is not IntentClass.I15 and self.compliance_state is not None:
            raise SchemaValidationError(
                "compliance_state is reserved for I15 audit instances"
            )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One attacker-visible event in Equation (1)."""

    timestamp_s: float
    channel: Channel
    direction: Direction
    length_bytes: int
    flow_id: str = "default"
    retransmission: bool = False
    transport_unit: str = "tls_record"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.channel, Channel):
            raise SchemaValidationError("channel must be a Channel")
        if not isinstance(self.direction, Direction):
            raise SchemaValidationError("direction must be a Direction")
        if isinstance(self.timestamp_s, bool) or not isinstance(
            self.timestamp_s, (int, float)
        ):
            raise SchemaValidationError("timestamp_s must be a finite number")
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise SchemaValidationError("timestamp_s must be finite and nonnegative")
        if (
            not isinstance(self.length_bytes, int)
            or isinstance(self.length_bytes, bool)
            or self.length_bytes <= 0
        ):
            raise SchemaValidationError("length_bytes must be a positive integer")
        if type(self.retransmission) is not bool:
            raise SchemaValidationError("retransmission must be a literal boolean")
        if not self.flow_id.strip():
            raise SchemaValidationError("flow_id must not be empty")
        if self.transport_unit not in {"tls_record", "packet", "burst"}:
            raise SchemaValidationError(
                "transport_unit must be tls_record, packet, or burst"
            )


@dataclass(frozen=True, slots=True)
class ChannelTrace:
    """Chronologically ordered observation for one logical channel."""

    channel: Channel
    events: tuple[TraceEvent, ...]
    oracle_segmented: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.channel, Channel):
            raise SchemaValidationError("channel must be a Channel")
        if any(event.channel is not self.channel for event in self.events):
            raise SchemaValidationError("all trace events must match trace.channel")
        timestamps = [event.timestamp_s for event in self.events]
        if timestamps != sorted(timestamps):
            raise SchemaValidationError("trace events must be chronological")

    @property
    def signed_sizes(self) -> tuple[int, ...]:
        """Return the signed-size sequence used by the attack models."""

        return tuple(int(event.direction) * event.length_bytes for event in self.events)

    @property
    def interarrival_s(self) -> tuple[float, ...]:
        """Return nonnegative inter-arrival times, with zero at trace start."""

        if not self.events:
            return ()
        times = [event.timestamp_s for event in self.events]
        return (0.0, *(right - left for left, right in zip(times, times[1:])))


@dataclass(frozen=True, slots=True)
class FusedTrace:
    """Per-channel traces aligned on a common experiment clock."""

    traces: tuple[ChannelTrace, ...]

    def __post_init__(self) -> None:
        channels = [trace.channel for trace in self.traces]
        if len(set(channels)) != len(channels):
            raise SchemaValidationError("a fused trace may contain each channel once")

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        """Return all events sorted by common-clock timestamp."""

        return tuple(
            sorted(
                (event for trace in self.traces for event in trace.events),
                key=lambda event: (event.timestamp_s, event.channel.value),
            )
        )


@dataclass(frozen=True, slots=True)
class ApplicationMessage:
    """A plaintext-side message presented to a shaper or padding baseline."""

    message_id: str
    channel: Channel
    direction: Direction
    size_bytes: int
    ready_at_s: float
    semantic_kind: str
    side_effecting: bool = False
    correlation_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.channel, Channel):
            raise SchemaValidationError("channel must be a Channel")
        if not isinstance(self.direction, Direction):
            raise SchemaValidationError("direction must be a Direction")
        if not self.message_id.strip():
            raise SchemaValidationError("message_id must not be empty")
        if self.size_bytes <= 0:
            raise SchemaValidationError("size_bytes must be positive")
        if not isfinite(self.ready_at_s) or self.ready_at_s < 0:
            raise SchemaValidationError("ready_at_s must be finite and nonnegative")
        if not self.semantic_kind.strip():
            raise SchemaValidationError("semantic_kind must not be empty")


@dataclass(frozen=True, slots=True)
class RunSpecification:
    """The experimental-unit tuple declared in Section 5.4."""

    run_id: str
    intent: IntentInstance
    model_id: str
    defense_condition: DefenseCondition
    repetition_seed: int

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.model_id.strip():
            raise SchemaValidationError("run_id and model_id must not be empty")
