"""Class- and scope-aware selection of frozen envelope profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..exceptions import EnvelopeDefinitionError
from ..schemas import ApplicationMessage, IntentInstance
from .envelope import EnvelopeProfile
from .groups import GroupPartition
from .scheduler import (
    FixedCadenceScheduler,
    OverflowEvent,
    OverflowReason,
    ShapingResult,
)


@dataclass(frozen=True, slots=True)
class IntentAwareShaper:
    """Select one public profile through ``G = gamma(C)`` and shape messages.

    Profiles are validated against the complete explicit partition at
    construction time. An unadmitted scope produces an explicit overflow result
    and never falls back to a test-dependent profile.
    """

    partition: GroupPartition
    profiles: Mapping[str, EnvelopeProfile]
    scheduler: FixedCadenceScheduler = field(default_factory=FixedCadenceScheduler)

    def __post_init__(self) -> None:
        expected = {group.group_id for group in self.partition.groups}
        supplied = set(self.profiles)
        if supplied != expected:
            raise EnvelopeDefinitionError(
                "profiles must cover exactly the partition groups; "
                f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
            )
        for group_id, profile in self.profiles.items():
            if profile.group_id != group_id:
                raise EnvelopeDefinitionError(
                    f"profile {profile.profile_id!r} declares group {profile.group_id!r}, "
                    f"expected {group_id!r}"
                )

    def shape(
        self,
        intent: IntentInstance,
        messages: tuple[ApplicationMessage, ...],
    ) -> ShapingResult:
        """Shape one intent using its frozen class group and admitted scope."""

        group = self.partition.group_for(intent.class_id)
        profile = self.profiles[group.group_id]
        if intent.scope not in profile.admitted_scopes:
            return ShapingResult(
                profile_id=profile.profile_id,
                overflow_policy=profile.overflow_policy,
                transmissions=(),
                overflow_events=(
                    OverflowEvent(
                        message_id="__intent_scope__",
                        reason=OverflowReason.SCOPE_NOT_ADMITTED,
                        detail=(
                            f"scope {intent.scope.value!r} is not admitted by "
                            f"profile {profile.profile_id!r}"
                        ),
                    ),
                ),
                unscheduled_message_ids=tuple(message.message_id for message in messages),
                termination_at_s=(
                    profile.public_start_epoch_s + profile.common_termination_s
                ),
            )
        return self.scheduler.schedule(profile, messages)
