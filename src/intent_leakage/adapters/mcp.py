"""MCP local-stdio adapter contract for the confirmatory grid."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from .base import AdapterNotProvisionedError, require_external_execution


class McpToolAdapter(ABC):
    """Contract for the frozen MCP tool server.

    The implementation intentionally accepts no shell command from configuration.
    A concrete integration must register an in-process or audited executable adapter.
    """

    def __init__(self, *, revision: str, allow_external: bool = False) -> None:
        require_external_execution(allow_external)
        if not revision.strip():
            raise ValueError("MCP revision must be recorded")
        self.revision = revision

    @abstractmethod
    def list_tools(self) -> Sequence[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError


class UnconfiguredMcpToolAdapter(McpToolAdapter):
    def list_tools(self) -> Sequence[Mapping[str, Any]]:
        raise AdapterNotProvisionedError("The frozen MCP server is not provisioned")

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        del name, arguments
        raise AdapterNotProvisionedError("The frozen MCP server is not provisioned")
