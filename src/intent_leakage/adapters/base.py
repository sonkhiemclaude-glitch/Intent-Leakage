"""Safety gates and shared types for external adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ExternalExecutionDisabledError(RuntimeError):
    """Raised when a network, controller, capture, or VM action was not authorized."""


class AdapterNotProvisionedError(RuntimeError):
    """Raised when a paper-required external implementation is still unavailable."""


def require_external_execution(allow_external: bool) -> None:
    """Fail closed unless external execution was explicitly enabled by the caller."""

    if allow_external is not True:
        raise ExternalExecutionDisabledError(
            "External execution is disabled. Pass --allow-external only on the "
            "provisioned Mac Studio after reviewing the frozen manifest."
        )


@dataclass(frozen=True)
class AdapterMetadata:
    """Non-secret adapter metadata stored in run provenance."""

    adapter_type: str
    version: str | None = None
    endpoint_label: str | None = None
    details: Mapping[str, Any] | None = None
