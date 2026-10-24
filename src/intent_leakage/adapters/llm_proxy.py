"""Conceptual adapter for CL shaping, which is outside the tool-shim prototype."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AdapterNotProvisionedError, require_external_execution


class LlmChannelProxyAdapter:
    """Explicitly unavailable trusted proxy required for a CL formal guarantee."""

    def __init__(self, *, allow_external: bool = False) -> None:
        require_external_execution(allow_external)

    def shape_exchange(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        raise AdapterNotProvisionedError(
            "No trusted Bedrock-side buffering/shaping proxy is specified in the "
            "paper. CL and fused views containing CL remain empirical only."
        )
