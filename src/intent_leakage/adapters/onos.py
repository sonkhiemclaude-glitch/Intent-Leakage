"""Typed contract for ONOS controller operations required by Table 3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .base import AdapterNotProvisionedError, require_external_execution


class OnosControllerAdapter(ABC):
    """Abstract ONOS integration.

    Concrete REST paths and payloads depend on the frozen ONOS image and any
    application providing slice/service objects, neither of which is fixed in the
    manuscript. The base constructor therefore only provides an execution gate.
    """

    def __init__(self, *, allow_external: bool = False) -> None:
        require_external_execution(allow_external)

    @abstractmethod
    def discover_topology(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def lookup_hosts(self, host_ids: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def lookup_paths(self, demands: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def apply_operation(self, operation: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def assurance_read(self, query: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, inverse_operation: Mapping[str, Any]) -> None:
        raise NotImplementedError


class UnconfiguredOnosAdapter(OnosControllerAdapter):
    """Explicit placeholder that never performs a controller request."""

    def _missing(self) -> None:
        raise AdapterNotProvisionedError(
            "ONOS version, applications, REST schemas, TLS endpoint, and cleanup "
            "semantics must be provisioned before controller execution."
        )

    def discover_topology(self) -> Mapping[str, Any]:
        self._missing()

    def lookup_hosts(self, host_ids: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        del host_ids
        self._missing()

    def lookup_paths(self, demands: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        del demands
        self._missing()

    def apply_operation(self, operation: Mapping[str, Any]) -> Mapping[str, Any]:
        del operation
        self._missing()

    def assurance_read(self, query: Mapping[str, Any]) -> Mapping[str, Any]:
        del query
        self._missing()

    def cleanup(self, inverse_operation: Mapping[str, Any]) -> None:
        del inverse_operation
        self._missing()
