"""Guards for the paper's one-factor-at-a-time transfer protocol."""

from __future__ import annotations

from dataclasses import dataclass


class TransferRefitError(RuntimeError):
    """Raised when a transfer block attempts to refit a frozen component."""


@dataclass(frozen=True)
class FrozenTransferProtocol:
    """Record the core environment and reject transfer-time fitting."""

    core_topology: str = "TA"
    core_model: str = "A"
    preprocessing_digest: str = ""
    attacker_digest: str = ""

    def assert_fit_allowed(self, *, topology: str, model: str, split: str) -> None:
        if topology != self.core_topology or model != self.core_model:
            raise TransferRefitError(
                "Model/topology transfer blocks are test-only; preprocessing and "
                "attackers must remain frozen from the core block."
            )
        if split == "test":
            raise TransferRefitError("The held-out test partition cannot be fitted")

    def validate_transfer(self, *, topology: str, model: str, split: str) -> None:
        if split != "test":
            raise ValueError("Transfer blocks must contain held-out test data only")
        changed = int(topology != self.core_topology) + int(model != self.core_model)
        if changed != 1:
            raise ValueError(
                "The minimum grid changes exactly one of topology or model at a time"
            )
