"""Exact minimum execution grid from manuscript Table 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

INTENT_CLASSES = tuple(f"I{index:02d}" for index in range(1, 16))
SCOPE_COUNTS = (1, 2, 4)
PARTITIONS = ("train", "validation", "test")
DEFENSE_CONDITIONS = (
    "no_shaping",
    "fixed_size_padding",
    "all_class_envelope",
    "budgeted_groups",
)


@dataclass(frozen=True)
class GridRun:
    """One unique confirmatory capture in the non-factorial minimum grid."""

    run_key: str
    block: str
    topology_id: str
    model_id: str
    condition: str
    partition: str
    class_id: str
    scope_q: int
    phrase_family_id: str
    repetition: int
    fit_allowed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cell_runs(
    *,
    block: str,
    topology_id: str,
    model_id: str,
    condition: str,
    partitions: Iterable[str],
    repetitions: int,
    phrase_family_names: Mapping[str, str],
    fit_allowed: bool,
) -> list[GridRun]:
    rows: list[GridRun] = []
    for partition in partitions:
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        family_name = phrase_family_names[partition]
        for class_id in INTENT_CLASSES:
            for scope_q in SCOPE_COUNTS:
                phrase_family_id = f"{class_id}-q{scope_q}-{family_name}"
                for repetition in range(1, repetitions + 1):
                    run_key = ":".join(
                        (
                            topology_id,
                            model_id,
                            condition,
                            partition,
                            class_id,
                            f"q{scope_q}",
                            f"r{repetition}",
                        )
                    )
                    rows.append(
                        GridRun(
                            run_key=run_key,
                            block=block,
                            topology_id=topology_id,
                            model_id=model_id,
                            condition=condition,
                            partition=partition,
                            class_id=class_id,
                            scope_q=scope_q,
                            phrase_family_id=phrase_family_id,
                            repetition=repetition,
                            fit_allowed=fit_allowed and partition != "test",
                        )
                    )
    return rows


def expected_run_count(repetitions: int, *, include_topology_transfer: bool = True) -> int:
    """Return ``630R`` or the one-topology fallback ``585R``."""

    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    return repetitions * (630 if include_topology_transfer else 585)


def build_minimum_grid(
    repetitions: int,
    *,
    topology_a: str = "TA",
    topology_b: str | None = "TB",
    model_a: str = "A",
    model_b: str = "B",
    phrase_family_names: Mapping[str, str] | None = None,
) -> list[GridRun]:
    """Build Table 4 without a model-topology-defense cross-product.

    The TA/model-A/no-shaping core capture is generated once and is reused in the
    defense comparison. Model and topology transfer blocks contain test data only.
    """

    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not topology_a or not model_a or not model_b:
        raise ValueError("TA, model A, and model B identifiers are required")
    names = dict(
        phrase_family_names
        or {"train": "train", "validation": "validation", "test": "test"}
    )
    if set(names) != set(PARTITIONS) or any(not str(value) for value in names.values()):
        raise ValueError("phrase_family_names must define train, validation, and test")

    grid: list[GridRun] = []
    for condition in DEFENSE_CONDITIONS:
        grid.extend(
            _cell_runs(
                block="core" if condition == "no_shaping" else "defense",
                topology_id=topology_a,
                model_id=model_a,
                condition=condition,
                partitions=PARTITIONS,
                repetitions=repetitions,
                phrase_family_names=names,
                fit_allowed=True,
            )
        )
    grid.extend(
        _cell_runs(
            block="model_transfer",
            topology_id=topology_a,
            model_id=model_b,
            condition="no_shaping",
            partitions=("test",),
            repetitions=repetitions,
            phrase_family_names=names,
            fit_allowed=False,
        )
    )
    if topology_b is not None:
        if not topology_b:
            raise ValueError("topology_b must be a non-empty identifier or None")
        grid.extend(
            _cell_runs(
                block="topology_transfer",
                topology_id=topology_b,
                model_id=model_a,
                condition="no_shaping",
                partitions=("test",),
                repetitions=repetitions,
                phrase_family_names=names,
                fit_allowed=False,
            )
        )

    expected = expected_run_count(
        repetitions, include_topology_transfer=topology_b is not None
    )
    if len(grid) != expected:
        raise AssertionError(f"Internal grid error: produced {len(grid)}, expected {expected}")
    keys = [row.run_key for row in grid]
    if len(keys) != len(set(keys)):
        raise AssertionError("Minimum grid contains duplicate capture keys")
    return grid
