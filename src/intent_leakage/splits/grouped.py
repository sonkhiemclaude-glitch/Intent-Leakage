"""Grouped split construction and validation from Section 5.6."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..utils.hashing import stable_digest


class GroupLeakageError(ValueError):
    """Raised when one intent instance or phrase family crosses split boundaries."""


def _as_rows(records: Any) -> list[dict[str, Any]]:
    if hasattr(records, "to_dict"):
        return [dict(row) for row in records.to_dict(orient="records")]
    return [dict(row) for row in records]


def validate_grouped_split(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    split_field: str = "split",
    group_fields: Sequence[str] = ("intent_instance_id", "phrase_family_id"),
) -> None:
    """Require each value of every grouping field to occur in only one split.

    The check treats instance ID and phrase-family ID independently, matching the
    paper's rule that neither repetitions of an instance nor a phrase family may
    appear on both sides of a split.
    """

    rows = _as_rows(records)
    if not rows:
        raise ValueError("Cannot validate an empty split")
    for field in (split_field, *group_fields):
        missing = [index for index, row in enumerate(rows) if field not in row]
        if missing:
            raise KeyError(f"Missing field {field!r} in rows {missing[:5]}")
    problems: list[str] = []
    for field in group_fields:
        observed: dict[Any, set[str]] = defaultdict(set)
        for row in rows:
            observed[row[field]].add(str(row[split_field]))
        for group, splits in observed.items():
            if len(splits) > 1:
                problems.append(f"{field}={group!r} crosses {sorted(splits)}")
    if problems:
        raise GroupLeakageError("; ".join(problems))


def assign_grouped_splits(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    group_field: str = "phrase_family_id",
    ratios: Mapping[str, float] | None = None,
    seed: int = 0,
    split_field: str = "split",
) -> list[dict[str, Any]]:
    """Assign whole groups to deterministic randomized splits.

    This helper is for synthetic examples and dataset preparation. Confirmatory
    phrase families should use the author-frozen assignment from the manifest.
    """

    rows = _as_rows(records)
    if not rows:
        raise ValueError("Cannot split an empty collection")
    if ratios is None:
        ratios = {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}
    if not ratios or any(weight < 0 for weight in ratios.values()):
        raise ValueError("Split ratios must be non-negative")
    total = float(sum(ratios.values()))
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    if any(group_field not in row for row in rows):
        raise KeyError(f"All records require {group_field!r}")

    groups = sorted({str(row[group_field]) for row in rows})
    random.Random(seed).shuffle(groups)
    split_names = list(ratios)
    normalized = [ratios[name] / total for name in split_names]

    # Allocate each group to the split with the largest remaining quota. This is
    # deterministic after the seeded shuffle and avoids splitting a small group.
    targets = {name: normalized[i] * len(groups) for i, name in enumerate(split_names)}
    counts = {name: 0 for name in split_names}
    assignment: dict[str, str] = {}
    for group in groups:
        selected = max(
            split_names,
            key=lambda name: (targets[name] - counts[name], -split_names.index(name)),
        )
        assignment[group] = selected
        counts[selected] += 1

    assigned = []
    for row in rows:
        copy = dict(row)
        copy[split_field] = assignment[str(row[group_field])]
        assigned.append(copy)
    validate_grouped_split(
        assigned, split_field=split_field, group_fields=(group_field,)
    )
    return assigned


def split_assignment_digest(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    split_field: str = "split",
    identity_fields: Sequence[str] = (
        "intent_instance_id",
        "phrase_family_id",
        "topology_seed",
    ),
) -> str:
    """Hash the frozen split assignment for provenance."""

    rows = _as_rows(records)
    projection = [
        {field: row.get(field) for field in (*identity_fields, split_field)}
        for row in rows
    ]
    projection.sort(key=lambda item: tuple(str(item.get(k)) for k in item))
    return stable_digest(projection)
