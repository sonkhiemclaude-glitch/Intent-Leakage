"""External-system adapter contracts.

Production adapters are deliberately not activated by import. Callers must supply
``allow_external=True`` and provision the corresponding dependency explicitly.
"""

from .base import (
    AdapterNotProvisionedError,
    ExternalExecutionDisabledError,
    require_external_execution,
)
from .bedrock import BedrockAdapter, BedrockConfig
from .llm_proxy import LlmChannelProxyAdapter
from .mcp import McpToolAdapter
from .mininet import MininetVmAdapter
from .onos import OnosControllerAdapter
from .pcap import PcapParserAdapter

__all__ = [
    "AdapterNotProvisionedError",
    "BedrockAdapter",
    "BedrockConfig",
    "ExternalExecutionDisabledError",
    "LlmChannelProxyAdapter",
    "McpToolAdapter",
    "MininetVmAdapter",
    "OnosControllerAdapter",
    "PcapParserAdapter",
    "require_external_execution",
]
