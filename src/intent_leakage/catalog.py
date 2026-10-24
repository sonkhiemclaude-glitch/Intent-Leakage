"""The complete predeclared intent catalogue from paper Table 3."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from intent_leakage.schemas import IntentClass, IntentDefinition


_CATALOG = {
    IntentClass.I01: IntentDefinition(
        IntentClass.I01,
        "Quarantine compromised endpoints",
        "hosts",
        "Targets cannot exchange data-plane traffic except through the declared "
        "management exception.",
    ),
    IntentClass.I02: IntentDefinition(
        IntentClass.I02,
        "Release quarantined endpoints",
        "hosts",
        "Baseline reachability for the targets is restored.",
    ),
    IntentClass.I03: IntentDefinition(
        IntentClass.I03,
        "Block communicating endpoint pairs",
        "endpoint_pairs",
        "Traffic for each named pair is denied by installed policy.",
    ),
    IntentClass.I04: IntentDefinition(
        IntentClass.I04,
        "Permit communicating endpoint pairs",
        "endpoint_pairs",
        "Traffic for each named pair is allowed by installed policy.",
    ),
    IntentClass.I05: IntentDefinition(
        IntentClass.I05,
        "Rate-limit selected services",
        "flows",
        "The requested meter or queue and its flow binding are present.",
    ),
    IntentClass.I06: IntentDefinition(
        IntentClass.I06,
        "Raise bandwidth or priority",
        "flows",
        "The requested queue, rate, or priority policy and binding are present.",
    ),
    IntentClass.I07: IntentDefinition(
        IntentClass.I07,
        "Reroute demands away from a resource",
        "demands",
        "Each installed path avoids the named link or node.",
    ),
    IntentClass.I08: IntentDefinition(
        IntentClass.I08,
        "Restore default paths",
        "demands",
        "Each demand follows the predeclared default routing policy.",
    ),
    IntentClass.I09: IntentDefinition(
        IntentClass.I09,
        "Create isolated tenant slices",
        "slices",
        "Slice objects, endpoint membership, and isolation rules are present.",
    ),
    IntentClass.I10: IntentDefinition(
        IntentClass.I10,
        "Resize tenant slices",
        "slices",
        "Membership or resource allocation matches the requested slice state.",
    ),
    IntentClass.I11: IntentDefinition(
        IntentClass.I11,
        "Delete tenant slices",
        "slices",
        "Slice objects and their derived rules are absent.",
    ),
    IntentClass.I12: IntentDefinition(
        IntentClass.I12,
        "Onboard tenant services",
        "services",
        "Required service, attachment, and policy objects are present and reachable.",
    ),
    IntentClass.I13: IntentDefinition(
        IntentClass.I13,
        "Withdraw tenant services",
        "services",
        "Service-specific objects and forwarding rules are absent.",
    ),
    IntentClass.I14: IntentDefinition(
        IntentClass.I14,
        "Redirect selected flows",
        "flows",
        "Each selected flow reaches the declared alternate service endpoint or "
        "inspection sink, while non-selected flows retain baseline forwarding.",
        "Use an ordinary flow selector and output treatment, not a service-function "
        "chain. Unlike I07, this changes destination or output.",
    ),
    IntentClass.I15: IntentDefinition(
        IntentClass.I15,
        "Audit selected policies once",
        "policies",
        "One bounded read-only assurance cycle retrieves flow, meter, and counter "
        "state and returns a compliance verdict for every selected policy; no "
        "persistent monitor is installed.",
        "This is assurance-only. Its real C_T call count may be zero, but a shaped "
        "run still emits scheduled C_T no-op slots.",
    ),
}

INTENT_CATALOG: Mapping[IntentClass, IntentDefinition] = MappingProxyType(_CATALOG)


def get_intent_definition(class_id: IntentClass | str) -> IntentDefinition:
    """Return the canonical definition for one Table 3 class."""

    return INTENT_CATALOG[IntentClass(class_id)]


def list_intent_definitions() -> tuple[IntentDefinition, ...]:
    """Return all 15 definitions in stable class-ID order."""

    return tuple(INTENT_CATALOG[class_id] for class_id in IntentClass)
