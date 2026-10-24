"""Fail-closed lifecycle for one real confirmatory capture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from ..adapters.base import require_external_execution


class RunState(str, Enum):
    CREATED = "created"
    RESTORED = "restored"
    CAPTURING = "capturing"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    STORED = "stored"
    CLEANED = "cleaned"
    FAILED = "failed"


class RunBackend(Protocol):
    """Audited backend implemented later on the provisioned Mac Studio."""

    def restore_and_verify(self, run: Mapping[str, Any]) -> None: ...
    def start_capture(self, run: Mapping[str, Any]) -> None: ...
    def submit_and_wait(self, run: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def verify_postcondition(self, run: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def stop_and_store_capture(self, run: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def cleanup_and_verify(self, run: Mapping[str, Any]) -> None: ...


@dataclass
class RunStateMachine:
    backend: RunBackend
    state: RunState = RunState.CREATED

    def execute(
        self,
        run: Mapping[str, Any],
        *,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        """Execute the Section 5.4 lifecycle only after explicit authorization."""

        require_external_execution(allow_external)
        result: dict[str, Any] = {"run_key": run.get("run_key")}
        capture_start_attempted = False
        capture_stored = False
        try:
            self.backend.restore_and_verify(run)
            self.state = RunState.RESTORED
            # Once start is attempted, a backend may have activated capture
            # before reporting an error. Emergency stop/store is therefore
            # required even when start_capture itself raises.
            capture_start_attempted = True
            self.backend.start_capture(run)
            self.state = RunState.CAPTURING
            result["agent"] = dict(self.backend.submit_and_wait(run))
            self.state = RunState.SUBMITTED
            result["postcondition"] = dict(self.backend.verify_postcondition(run))
            self.state = RunState.VERIFIED
            result["capture"] = dict(self.backend.stop_and_store_capture(run))
            capture_stored = True
            self.state = RunState.STORED
            self.backend.cleanup_and_verify(run)
            self.state = RunState.CLEANED
            result["state"] = self.state.value
            return result
        except BaseException:
            # Preserve partial evidence and stop an active capture before state
            # cleanup. Both emergency operations must be idempotent because the
            # triggering failure may have occurred inside either backend call.
            if capture_start_attempted and not capture_stored:
                try:
                    self.backend.stop_and_store_capture(run)
                except BaseException:
                    pass
            try:
                self.backend.cleanup_and_verify(run)
            except BaseException:
                pass
            self.state = RunState.FAILED
            raise
