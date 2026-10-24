"""Intent-aware shaping primitives."""

from intent_leakage.shaping.compliance import (
    ConformanceIssue,
    EnvelopeConformanceReport,
    verify_envelope_structure,
)
from intent_leakage.shaping.envelope import (
    EnvelopeProfile,
    EnvelopeSlot,
    SafeDecoy,
    make_uniform_envelope,
)
from intent_leakage.shaping.groups import (
    GroupPartition,
    ShapingGroup,
    all_class_partition,
    explicit_partition,
)
from intent_leakage.shaping.intent_aware import IntentAwareShaper
from intent_leakage.shaping.scheduler import (
    FixedCadenceScheduler,
    OverflowEvent,
    OverflowReason,
    ScheduledTransmission,
    ShapingResult,
)

__all__ = [
    "ConformanceIssue",
    "EnvelopeConformanceReport",
    "EnvelopeProfile",
    "EnvelopeSlot",
    "FixedCadenceScheduler",
    "GroupPartition",
    "IntentAwareShaper",
    "OverflowEvent",
    "OverflowReason",
    "SafeDecoy",
    "ScheduledTransmission",
    "ShapingGroup",
    "ShapingResult",
    "all_class_partition",
    "explicit_partition",
    "make_uniform_envelope",
    "verify_envelope_structure",
]
