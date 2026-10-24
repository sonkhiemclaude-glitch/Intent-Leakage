"""Explicit intent-class partitions for shaping anonymity groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from intent_leakage.exceptions import PartitionError
from intent_leakage.schemas import IntentClass


@dataclass(frozen=True, slots=True)
class ShapingGroup:
    """One nonempty anonymity set in Equation (4)."""

    group_id: str
    classes: frozenset[IntentClass]

    def __post_init__(self) -> None:
        if not self.group_id.strip():
            raise PartitionError("group_id must not be empty")
        if not self.classes:
            raise PartitionError("a shaping group must contain at least one class")


@dataclass(frozen=True, slots=True)
class GroupPartition:
    """A complete, disjoint, explicit partition of all 15 intent classes."""

    groups: tuple[ShapingGroup, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise PartitionError("a partition must contain at least one group")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise PartitionError("group IDs must be unique")

        seen: set[IntentClass] = set()
        duplicates: set[IntentClass] = set()
        for group in self.groups:
            duplicates.update(seen.intersection(group.classes))
            seen.update(group.classes)
        if duplicates:
            values = ", ".join(sorted(item.value for item in duplicates))
            raise PartitionError(f"classes appear in multiple groups: {values}")

        expected = set(IntentClass)
        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            missing_text = ", ".join(sorted(item.value for item in missing)) or "none"
            extra_text = ", ".join(sorted(item.value for item in extra)) or "none"
            raise PartitionError(
                f"partition must cover exactly the catalogue; missing={missing_text}; "
                f"extra={extra_text}"
            )

    def group_for(self, class_id: IntentClass) -> ShapingGroup:
        """Return the unique group containing ``class_id``."""

        for group in self.groups:
            if class_id in group.classes:
                return group
        raise PartitionError(f"class {class_id.value} is not in the partition")

    @property
    def class_to_group(self) -> Mapping[IntentClass, str]:
        """Return the deterministic class-to-group map gamma(C)."""

        return {
            class_id: group.group_id
            for group in self.groups
            for class_id in sorted(group.classes, key=lambda value: value.value)
        }


def explicit_partition(groups: Mapping[str, Iterable[IntentClass | str]]) -> GroupPartition:
    """Build and validate an explicit partition.

    No data-dependent clustering is hidden in this helper.  The caller must
    state every group and class membership, making the privacy claim auditable.
    """

    shaped_groups = tuple(
        ShapingGroup(
            group_id=group_id,
            classes=frozenset(IntentClass(class_id) for class_id in class_ids),
        )
        for group_id, class_ids in groups.items()
    )
    return GroupPartition(groups=shaped_groups)


def all_class_partition(group_id: str = "all-classes") -> GroupPartition:
    """Return the strongest single-group configuration evaluated by the paper."""

    return GroupPartition(
        groups=(ShapingGroup(group_id=group_id, classes=frozenset(IntentClass)),)
    )
