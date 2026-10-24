from __future__ import annotations

import pytest

from intent_leakage.splits.grouped import (
    GroupLeakageError,
    assign_grouped_splits,
    split_assignment_digest,
    validate_grouped_split,
)
from intent_leakage.splits.transfer import FrozenTransferProtocol, TransferRefitError


def test_grouped_split_rejects_phrase_family_leakage() -> None:
    rows = [
        {"phrase_family_id": "family-1", "split": "train"},
        {"phrase_family_id": "family-1", "split": "test"},
    ]
    with pytest.raises(GroupLeakageError):
        validate_grouped_split(rows, group_fields=("phrase_family_id",))


def test_group_assignment_is_deterministic_and_disjoint() -> None:
    rows = [
        {"phrase_family_id": f"family-{index}", "value": repetition}
        for index in range(12)
        for repetition in range(2)
    ]
    first = assign_grouped_splits(rows, seed=17)
    second = assign_grouped_splits(rows, seed=17)
    assert first == second
    validate_grouped_split(first, group_fields=("phrase_family_id",))
    assert split_assignment_digest(first, identity_fields=("phrase_family_id",)) == (
        split_assignment_digest(second, identity_fields=("phrase_family_id",))
    )


def test_transfer_protocol_forbids_refit_and_two_factor_change() -> None:
    protocol = FrozenTransferProtocol(core_topology="TA", core_model="A")
    with pytest.raises(TransferRefitError):
        protocol.assert_fit_allowed(topology="TA", model="B", split="test")
    protocol.validate_transfer(topology="TA", model="B", split="test")
    protocol.validate_transfer(topology="TB", model="A", split="test")
    with pytest.raises(ValueError):
        protocol.validate_transfer(topology="TB", model="B", split="test")
