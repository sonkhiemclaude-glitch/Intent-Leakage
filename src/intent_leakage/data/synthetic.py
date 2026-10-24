"""Deterministic synthetic inputs for unit and smoke testing only.

Nothing produced here is a paper result or a substitute for Mac Studio PCAPs.
The generator creates structurally plausible messages and observations solely
to exercise schemas, shaping, features, and experiment plumbing offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from intent_leakage.catalog import get_intent_definition
from intent_leakage.schemas import (
    SYNTHETIC_OUTPUT_NOTICE,
    ApplicationMessage,
    Channel,
    ChannelTrace,
    DefenseCondition,
    Direction,
    FusedTrace,
    IntentClass,
    IntentInstance,
    RunSpecification,
    ScopeBucket,
    Split,
    TraceEvent,
)


@dataclass(frozen=True, slots=True)
class SyntheticRun:
    """One labeled synthetic run and its generated messages/traces."""

    specification: RunSpecification
    messages: tuple[ApplicationMessage, ...]
    trace: FusedTrace
    notice: str = SYNTHETIC_OUTPUT_NOTICE


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """A deterministic collection of synthetic runs."""

    runs: tuple[SyntheticRun, ...]
    seed: int
    notice: str = SYNTHETIC_OUTPUT_NOTICE


def _class_number(class_id: IntentClass) -> int:
    return int(class_id.value[1:])


def generate_intent_instance(
    class_id: IntentClass,
    scope: ScopeBucket,
    split: Split,
    *,
    topology_id: str = "synthetic-T_A",
    topology_seed: int = 0,
) -> IntentInstance:
    """Generate one manifest-valid synthetic intent instance."""

    definition = get_intent_definition(class_id)
    target_stem = definition.primary_target_type.rstrip("s") or "target"
    targets = tuple(
        f"{target_stem}-{topology_seed:03d}-{index:02d}"
        for index in range(1, scope.q + 1)
    )
    target_text = ", ".join(targets)
    compliance_state = (
        bool((topology_seed + scope.q) % 2) if class_id is IntentClass.I15 else None
    )
    phrasing = {
        Split.TRAIN: f"{definition.name} for {target_text}.",
        Split.VALIDATION: f"Please achieve this objective: {definition.name.lower()} "
        f"for {target_text}.",
        Split.TEST: f"Operational goal for {target_text}: {definition.name.lower()}.",
    }[split]
    return IntentInstance(
        instance_id=(
            f"synthetic-{topology_id}-{class_id.value}-{scope.value}-{split.value}-"
            f"{topology_seed}"
        ),
        class_id=class_id,
        scope=scope,
        q=scope.q,
        targets=targets,
        topology_id=topology_id,
        topology_seed=topology_seed,
        phrase_family_id=f"synthetic-{class_id.value}-{scope.value}-{split.value}",
        split=split,
        natural_language_intent=phrasing,
        expected_postcondition=definition.assurance_assertion,
        inverse_cleanup_operation=f"Restore synthetic base state for {target_text}.",
        compliance_state=compliance_state,
        metadata={"notice": SYNTHETIC_OUTPUT_NOTICE},
    )


def generate_application_messages(
    intent: IntentInstance,
    *,
    seed: int,
) -> tuple[ApplicationMessage, ...]:
    """Generate a deterministic plaintext-side message sequence.

    The values are deliberately synthetic and have no empirical meaning.  I15
    emits no real tool call, matching the catalogue note, while later shaping
    may still populate tool slots with no-ops.
    """

    combined_seed = (
        seed * 1_000_003
        + _class_number(intent.class_id) * 10_007
        + intent.q * 101
        + intent.topology_seed
        + {
            Split.TRAIN: 17,
            Split.VALIDATION: 31,
            Split.TEST: 47,
        }[intent.split]
    )
    rng = Random(combined_seed)
    messages: list[ApplicationMessage] = []
    clock = 0.0

    def add_pair(
        channel: Channel,
        semantic_prefix: str,
        ordinal: int,
        *,
        side_effecting: bool,
    ) -> None:
        nonlocal clock
        request_id = f"{intent.instance_id}-{semantic_prefix}-{ordinal}-req"
        response_id = f"{intent.instance_id}-{semantic_prefix}-{ordinal}-resp"
        request_size = 96 + 7 * _class_number(intent.class_id) + 13 * intent.q
        request_size += rng.randint(0, 15)
        response_size = 128 + 11 * _class_number(intent.class_id) + 17 * intent.q
        response_size += rng.randint(0, 31)
        clock += 0.05 + rng.random() * 0.01
        messages.append(
            ApplicationMessage(
                message_id=request_id,
                channel=channel,
                direction=Direction.AWAY_FROM_AGENT,
                size_bytes=request_size,
                ready_at_s=round(clock, 6),
                semantic_kind=f"{semantic_prefix}_request",
                side_effecting=side_effecting,
                correlation_id=f"{intent.instance_id}-{semantic_prefix}-{ordinal}",
                payload={"notice": SYNTHETIC_OUTPUT_NOTICE},
            )
        )
        clock += 0.08 + rng.random() * 0.02
        messages.append(
            ApplicationMessage(
                message_id=response_id,
                channel=channel,
                direction=Direction.TOWARD_AGENT,
                size_bytes=response_size,
                ready_at_s=round(clock, 6),
                semantic_kind=f"{semantic_prefix}_response",
                correlation_id=f"{intent.instance_id}-{semantic_prefix}-{ordinal}",
                payload={"notice": SYNTHETIC_OUTPUT_NOTICE},
            )
        )

    add_pair(Channel.LLM, "llm", 0, side_effecting=False)

    if intent.class_id is not IntentClass.I15:
        tool_turns = 1 + ((_class_number(intent.class_id) + intent.q) % 2)
        for ordinal in range(tool_turns):
            add_pair(Channel.TOOL, "tool", ordinal, side_effecting=True)

    for ordinal in range(intent.q):
        add_pair(Channel.ASSURANCE, "assurance", ordinal, side_effecting=False)

    return tuple(messages)


def messages_to_fused_trace(
    messages: tuple[ApplicationMessage, ...],
    *,
    seed: int,
) -> FusedTrace:
    """Convert synthetic messages into deterministic attacker-visible events."""

    rng = Random(seed)
    by_channel: dict[Channel, list[TraceEvent]] = {channel: [] for channel in Channel}
    for message in messages:
        transport_overhead = 29 + rng.randint(0, 7)
        by_channel[message.channel].append(
            TraceEvent(
                timestamp_s=message.ready_at_s,
                channel=message.channel,
                direction=message.direction,
                length_bytes=message.size_bytes + transport_overhead,
                flow_id=f"synthetic-{message.channel.value}",
                metadata={"notice": SYNTHETIC_OUTPUT_NOTICE},
            )
        )
    traces = tuple(
        ChannelTrace(channel=channel, events=tuple(by_channel[channel]))
        for channel in Channel
    )
    return FusedTrace(traces=traces)


def generate_synthetic_run(
    class_id: IntentClass,
    scope: ScopeBucket,
    split: Split,
    *,
    seed: int,
    topology_id: str = "synthetic-T_A",
) -> SyntheticRun:
    """Generate one complete deterministic synthetic run."""

    intent = generate_intent_instance(
        class_id,
        scope,
        split,
        topology_id=topology_id,
        topology_seed=seed,
    )
    messages = generate_application_messages(intent, seed=seed)
    specification = RunSpecification(
        run_id=f"synthetic-run-{class_id.value}-{scope.value}-{split.value}-{seed}",
        intent=intent,
        model_id="synthetic-model",
        defense_condition=DefenseCondition.NO_SHAPING,
        repetition_seed=seed,
    )
    return SyntheticRun(
        specification=specification,
        messages=messages,
        trace=messages_to_fused_trace(messages, seed=seed),
    )


def generate_synthetic_dataset(
    *,
    seed: int = 0,
    repetitions: int = 1,
    splits: tuple[Split, ...] = (Split.TRAIN, Split.VALIDATION, Split.TEST),
) -> SyntheticDataset:
    """Generate every class-scope-split cell for lightweight smoke tests."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    runs: list[SyntheticRun] = []
    for class_id in IntentClass:
        for scope in ScopeBucket:
            for split in splits:
                for repetition in range(repetitions):
                    run_seed = seed + repetition
                    runs.append(
                        generate_synthetic_run(
                            class_id,
                            scope,
                            split,
                            seed=run_seed,
                        )
                    )
    return SyntheticDataset(runs=tuple(runs), seed=seed)
