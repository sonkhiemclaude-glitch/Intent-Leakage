"""Data generation and loading interfaces."""

from intent_leakage.data.synthetic import (
    SyntheticDataset,
    SyntheticRun,
    generate_application_messages,
    generate_intent_instance,
    generate_synthetic_dataset,
    generate_synthetic_run,
    messages_to_fused_trace,
)
from intent_leakage.data.io import (
    FORBIDDEN_ATTACKER_EVENT_FIELDS,
    LoadedTraceDataset,
    load_dataset_root,
    load_trace_events_jsonl,
)

__all__ = [
    "SyntheticDataset",
    "SyntheticRun",
    "generate_application_messages",
    "generate_intent_instance",
    "generate_synthetic_dataset",
    "generate_synthetic_run",
    "messages_to_fused_trace",
    "FORBIDDEN_ATTACKER_EVENT_FIELDS",
    "LoadedTraceDataset",
    "load_dataset_root",
    "load_trace_events_jsonl",
]
