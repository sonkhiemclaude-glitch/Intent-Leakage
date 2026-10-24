from __future__ import annotations

import pytest

from intent_leakage.catalog import (
    INTENT_CATALOG,
    get_intent_definition,
    list_intent_definitions,
)
from intent_leakage.exceptions import PartitionError
from intent_leakage.schemas import IntentClass
from intent_leakage.shaping import all_class_partition, explicit_partition


def test_catalog_contains_all_and_only_fifteen_predeclared_classes() -> None:
    assert len(INTENT_CATALOG) == 15
    assert tuple(item.class_id for item in list_intent_definitions()) == tuple(IntentClass)
    assert set(INTENT_CATALOG) == set(IntentClass)


def test_i14_and_i15_encode_deadline_feasibility_corrections() -> None:
    i14 = get_intent_definition("I14")
    i15 = get_intent_definition(IntentClass.I15)
    assert "not a service-function chain" in i14.implementation_note
    assert "assurance-only" in i15.implementation_note
    assert "no persistent monitor" in i15.assurance_assertion


def test_group_partition_is_explicit_complete_and_disjoint() -> None:
    partition = all_class_partition()
    assert len(partition.groups) == 1
    assert partition.group_for(IntentClass.I12).group_id == "all-classes"

    with pytest.raises(PartitionError, match="cover exactly"):
        explicit_partition({"partial": [IntentClass.I01, IntentClass.I02]})

    with pytest.raises(PartitionError, match="multiple groups"):
        explicit_partition(
            {
                "a": tuple(IntentClass),
                "b": [IntentClass.I01],
            }
        )
