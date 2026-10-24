from __future__ import annotations

from collections import Counter

import pytest

from intent_leakage.orchestration.grid import (
    INTENT_CLASSES,
    SCOPE_COUNTS,
    build_minimum_grid,
    expected_run_count,
)


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_two_topology_grid_has_exact_630r_runs(repetitions: int) -> None:
    grid = build_minimum_grid(repetitions)
    assert len(grid) == 630 * repetitions
    assert len({row.run_key for row in grid}) == len(grid)
    assert {row.class_id for row in grid} == set(INTENT_CLASSES)
    assert {row.scope_q for row in grid} == set(SCOPE_COUNTS)


def test_one_topology_fallback_has_exact_585r_runs() -> None:
    grid = build_minimum_grid(1, topology_b=None)
    assert len(grid) == 585
    assert expected_run_count(1, include_topology_transfer=False) == 585
    assert not any(row.block == "topology_transfer" for row in grid)


def test_grid_is_nonfactorial_and_transfer_is_test_only() -> None:
    grid = build_minimum_grid(1)
    counts = Counter(row.block for row in grid)
    assert counts == {
        "core": 135,
        "defense": 405,
        "model_transfer": 45,
        "topology_transfer": 45,
    }
    transfer = [row for row in grid if row.block.endswith("transfer")]
    assert all(row.partition == "test" and not row.fit_allowed for row in transfer)
    assert not any(
        row.topology_id == "TB" and row.model_id == "B" for row in grid
    )
