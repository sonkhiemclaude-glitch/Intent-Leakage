from __future__ import annotations

from intent_leakage.data import (
    generate_application_messages,
    generate_intent_instance,
    generate_synthetic_dataset,
)
from intent_leakage.schemas import (
    SYNTHETIC_OUTPUT_NOTICE,
    Channel,
    IntentClass,
    ScopeBucket,
    Split,
)


def test_generator_covers_every_class_scope_and_split() -> None:
    dataset = generate_synthetic_dataset(seed=11)
    assert len(dataset.runs) == 15 * 3 * 3
    cells = {
        (
            run.specification.intent.class_id,
            run.specification.intent.scope,
            run.specification.intent.split,
        )
        for run in dataset.runs
    }
    assert cells == {
        (class_id, scope, split)
        for class_id in IntentClass
        for scope in ScopeBucket
        for split in Split
    }
    assert dataset.notice == SYNTHETIC_OUTPUT_NOTICE
    assert all(run.notice == SYNTHETIC_OUTPUT_NOTICE for run in dataset.runs)


def test_synthetic_generation_is_deterministic_under_fixed_seed() -> None:
    first = generate_synthetic_dataset(seed=2026, splits=(Split.TEST,))
    second = generate_synthetic_dataset(seed=2026, splits=(Split.TEST,))
    third = generate_synthetic_dataset(seed=2027, splits=(Split.TEST,))
    assert first == second
    assert first != third


def test_i15_has_assurance_messages_but_no_real_tool_messages() -> None:
    intent = generate_intent_instance(
        IntentClass.I15,
        ScopeBucket.HIGH,
        Split.TEST,
        topology_seed=3,
    )
    messages = generate_application_messages(intent, seed=7)
    assert not [message for message in messages if message.channel is Channel.TOOL]
    assert [message for message in messages if message.channel is Channel.ASSURANCE]
