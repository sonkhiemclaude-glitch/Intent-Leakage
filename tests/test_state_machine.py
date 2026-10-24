from __future__ import annotations

import pytest

from intent_leakage.orchestration.state_machine import RunState, RunStateMachine


class RecordingBackend:
    def __init__(self, fail_at: str | None = None) -> None:
        self.fail_at = fail_at
        self.calls: list[str] = []

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"failure at {name}")

    def restore_and_verify(self, run):
        self._call("restore")

    def start_capture(self, run):
        self._call("start_capture")

    def submit_and_wait(self, run):
        self._call("submit")
        return {"status": "ok"}

    def verify_postcondition(self, run):
        self._call("verify")
        return {"verified": True}

    def stop_and_store_capture(self, run):
        self._call("stop_and_store")
        return {"sha256": "synthetic-test-digest"}

    def cleanup_and_verify(self, run):
        self._call("cleanup")


def test_state_machine_success_order() -> None:
    backend = RecordingBackend()
    machine = RunStateMachine(backend)
    result = machine.execute({"run_key": "run-1"}, allow_external=True)
    assert machine.state is RunState.CLEANED
    assert result["state"] == "cleaned"
    assert backend.calls == [
        "restore",
        "start_capture",
        "submit",
        "verify",
        "stop_and_store",
        "cleanup",
    ]


def test_state_machine_stops_active_capture_before_emergency_cleanup() -> None:
    backend = RecordingBackend(fail_at="submit")
    machine = RunStateMachine(backend)
    with pytest.raises(RuntimeError, match="failure at submit"):
        machine.execute({"run_key": "run-2"}, allow_external=True)
    assert machine.state is RunState.FAILED
    assert backend.calls == [
        "restore",
        "start_capture",
        "submit",
        "stop_and_store",
        "cleanup",
    ]


def test_state_machine_emergency_stop_after_start_attempt_raises() -> None:
    backend = RecordingBackend(fail_at="start_capture")
    machine = RunStateMachine(backend)
    with pytest.raises(RuntimeError, match="failure at start_capture"):
        machine.execute({"run_key": "run-3"}, allow_external=True)
    assert machine.state is RunState.FAILED
    assert backend.calls == ["restore", "start_capture", "stop_and_store", "cleanup"]
