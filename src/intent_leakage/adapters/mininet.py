"""Contract for a Mininet/OVS Linux VM controlled from the Mac Studio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from .base import AdapterNotProvisionedError, require_external_execution


class MininetVmAdapter(ABC):
    def __init__(self, *, allow_external: bool = False) -> None:
        require_external_execution(allow_external)

    @abstractmethod
    def restore_snapshot(self, topology_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def verify_base_state(self, topology_id: str) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def topology_metadata(self, topology_id: str) -> Mapping[str, Any]:
        raise NotImplementedError


class UnconfiguredMininetVmAdapter(MininetVmAdapter):
    def _missing(self) -> None:
        raise AdapterNotProvisionedError(
            "The Mac Studio hypervisor, Linux VM, and versioned TA/TB topologies "
            "must be provisioned before execution."
        )

    def restore_snapshot(self, topology_id: str) -> None:
        del topology_id
        self._missing()

    def verify_base_state(self, topology_id: str) -> Mapping[str, Any]:
        del topology_id
        self._missing()

    def topology_metadata(self, topology_id: str) -> Mapping[str, Any]:
        del topology_id
        self._missing()
