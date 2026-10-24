"""Encrypted-traffic feature extraction."""

from .extraction import (
    PAPER_CHANNELS,
    BurstRepresentation,
    NormalizedEvent,
    SequenceRepresentation,
    TraceFeatureExtractor,
    burst_representation,
    extract_channel_features,
    extract_fused_features,
    fused_sequence_tensor,
    normalize_channel,
    normalize_events,
    padded_sequence,
    sequence_representation,
)

__all__ = [
    "BurstRepresentation",
    "NormalizedEvent",
    "PAPER_CHANNELS",
    "SequenceRepresentation",
    "TraceFeatureExtractor",
    "burst_representation",
    "extract_channel_features",
    "extract_fused_features",
    "fused_sequence_tensor",
    "normalize_channel",
    "normalize_events",
    "padded_sequence",
    "sequence_representation",
]
