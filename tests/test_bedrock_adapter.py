"""Converse contracts, token limits and stream/connection cleanup without AWS."""
from dataclasses import replace
import pytest
from intent_leakage.adapters.bedrock import BedrockAdapter, BedrockConfig
from intent_leakage.adapters.base import ExternalExecutionDisabledError


def config(**changes):
    return replace(BedrockConfig("amazon.nova-micro-v1:0", "us-east-1", False,
        {"temperature": 0.0}, 32, {"total_max_attempts": 1}, "reuse"), **changes)


class Events:
    def __init__(self, events):
        self.events = events
        self.closed = False
    def __iter__(self):
        yield from self.events
    def close(self):
        self.closed = True


class Client:
    def __init__(self):
        self.calls = []
        self.closed = 0
        self.events = Events([{"contentBlockDelta": {"delta": {"text": "READY"}}}])
        self.fail = False
    def converse(self, **request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("injected")
        return {"output": {"message": {"content": [{"text": "READY"}]}}}
    def converse_stream(self, **request):
        self.calls.append(request)
        return {"stream": self.events}
    def close(self):
        self.closed += 1


REQUEST = {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}


def test_gate_prevents_client_creation():
    with pytest.raises(ExternalExecutionDisabledError):
        BedrockAdapter(config(), client_factory=lambda: pytest.fail("client created"))


def test_reuse_and_frozen_inference_parameters():
    client = Client()
    adapter = BedrockAdapter(config(), allow_external=True, client_factory=lambda: client)
    adapter.invoke(REQUEST)
    adapter.invoke(REQUEST)
    assert client.closed == 0
    assert client.calls[0]["modelId"] == "amazon.nova-micro-v1:0"
    assert client.calls[0]["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 32}
    adapter.close()
    adapter.close()
    assert client.closed == 1


@pytest.mark.parametrize("override", [{"modelId": "other"}, {"inferenceConfig": {"maxTokens": 10000}}, {"endpoint_url": "other"}])
def test_request_cannot_override_model_token_ceiling_or_endpoint(override):
    adapter = BedrockAdapter(config(), allow_external=True, client_factory=lambda: pytest.fail("client created"))
    with pytest.raises(ValueError, match="override"):
        adapter.invoke({**REQUEST, **override})


def test_private_client_is_closed_after_api_failure():
    client = Client()
    client.fail = True
    adapter = BedrockAdapter(config(connection_pooling="new_client_per_request"), allow_external=True, client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="injected"):
        adapter.invoke(REQUEST)
    assert client.closed == 1


def test_early_stream_close_releases_resources():
    client = Client()
    adapter = BedrockAdapter(config(streaming=True, connection_pooling="disabled"), allow_external=True, client_factory=lambda: client)
    events = adapter.invoke_stream(REQUEST)
    assert "contentBlockDelta" in next(events)
    events.close()
    assert client.events.closed
    assert client.closed == 1


def test_stream_error_event_is_not_success():
    client = Client()
    client.events = Events([{"modelStreamErrorException": {"message": "injected"}}])
    adapter = BedrockAdapter(config(streaming=True, connection_pooling="disabled"), allow_external=True, client_factory=lambda: client)
    with pytest.raises(RuntimeError, match="stream error"):
        list(adapter.invoke_stream(REQUEST))
    assert client.events.closed and client.closed == 1


@pytest.mark.parametrize("changes", [{"max_response_tokens": True}, {"read_timeout_s": float("nan")}, {"retry_policy": {"total_max_attempts": 0}}, {"decoding_parameters": {"maxTokens": 9999}}, {"region": "https://example.com"}])
def test_invalid_runtime_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        BedrockAdapter(config(**changes), allow_external=True)
