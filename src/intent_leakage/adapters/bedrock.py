"""Explicitly enabled Bedrock Converse/ConverseStream integration.

Uses the provider-neutral messages API rather than guessing native model codecs.
Credentials are resolved by the SDK, never stored in configuration or metadata.
The remaining ONOS/Mininet/capture integrations are independent requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping
import math
import re

from .base import AdapterNotProvisionedError, require_external_execution


@dataclass(frozen=True)
class BedrockConfig:
    model_id: str
    region: str
    streaming: bool
    decoding_parameters: Mapping[str, Any]
    max_response_tokens: int
    retry_policy: Mapping[str, Any]
    connection_pooling: str
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0

    def validate(self) -> None:
        if not self.model_id.strip() or any(v in self.model_id for v in ("BEDROCK-MODEL-ID", "[TO CONFIRM]")):
            raise ValueError("A concrete Bedrock model ID is required")
        if not re.fullmatch(r"[a-z]{2}(?:-[a-z]+)+-\d+", self.region):
            raise ValueError("A Bedrock region is required")
        if type(self.max_response_tokens) is not int or self.max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be positive")
        if type(self.streaming) is not bool:
            raise ValueError("streaming must be boolean")
        if self.connection_pooling not in {"reuse", "new_client_per_request", "disabled"}:
            raise ValueError("connection_pooling must be reuse or new_client_per_request")
        if set(self.decoding_parameters) - {"temperature", "topP", "stopSequences"}:
            raise ValueError("Only Converse temperature/topP/stopSequences are decoding parameters")
        for value in (self.connect_timeout_s, self.read_timeout_s):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("timeouts must be positive and finite")
        if set(self.retry_policy) - {"mode", "total_max_attempts"}:
            raise ValueError("retry_policy requires mode/total_max_attempts, not ambiguous max_attempts")
        attempts = self.retry_policy.get("total_max_attempts", 1)
        if type(attempts) is not int or attempts < 1:
            raise ValueError("total_max_attempts must be a positive integer")
        if self.retry_policy.get("mode", "standard") not in {"standard", "adaptive", "legacy"}:
            raise ValueError("Unsupported SDK retry mode")


class BedrockAdapter:
    """Converse client with frozen model/decoding and an output-token ceiling."""

    def __init__(self, config: BedrockConfig, *, allow_external: bool = False,
                 client_factory: Callable[[], Any] | None = None) -> None:
        require_external_execution(allow_external)
        config.validate()
        self.config = config
        self._client_factory = client_factory
        self._client: Any = None

    def _get_client(self) -> Any:
        if self.config.connection_pooling == "reuse" and self._client is not None:
            return self._client
        if self._client_factory is not None:
            client = self._client_factory()
        else:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise AdapterNotProvisionedError("Install the optional aws dependency") from exc
            client = boto3.client("bedrock-runtime", region_name=self.config.region,
                config=Config(connect_timeout=self.config.connect_timeout_s,
                    read_timeout=self.config.read_timeout_s, max_pool_connections=2,
                    retries={"mode": "standard", "total_max_attempts": 1, **self.config.retry_policy}))
        if self.config.connection_pooling == "reuse":
            self._client = client
        return client

    def _request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {"messages", "system", "toolConfig", "additionalModelRequestFields", "requestMetadata"}
        if set(request) - allowed:
            raise ValueError("Request cannot override frozen model/decoding or SDK connection configuration")
        messages = request.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("A nonempty Converse messages list is required")
        return {**request, "modelId": self.config.model_id,
                "inferenceConfig": {**self.config.decoding_parameters,
                                    "maxTokens": self.config.max_response_tokens}}

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def invoke(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Invoke Converse once. SDK retries follow the explicit frozen policy."""
        if self.config.streaming:
            raise ValueError("Use invoke_stream when streaming is configured")
        parameters = self._request(request)
        client = self._get_client()
        try:
            return client.converse(**parameters)
        finally:
            if self.config.connection_pooling != "reuse":
                client.close()

    def invoke_stream(self, request: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        """Yield SDK events; close the event stream even after consumer failure.

        SDK event errors are propagated, not converted into successful responses.
        Transport capture and trace parsing are handled separately.
        """
        if not self.config.streaming:
            raise ValueError("Streaming must be configured explicitly")
        parameters = self._request(request)
        client = self._get_client()
        stream = None
        try:
            response = client.converse_stream(**parameters)
            stream = response["stream"]
            for event in stream:
                if any(name.endswith("Exception") for name in event):
                    raise RuntimeError("Bedrock returned a stream error event")
                yield event
        finally:
            if stream is not None:
                stream.close()
            if self.config.connection_pooling != "reuse":
                client.close()
