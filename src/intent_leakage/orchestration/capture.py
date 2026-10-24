"""Capture metadata schemas; actual capture is an external adapter responsibility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..utils.hashing import file_sha256


def normalize_channel_label(value: str) -> str:
    """Normalize ``CL``/``C_L`` style labels used across configs and schemas."""

    raw = str(value).strip().upper().replace("-", "_")
    aliases = {
        "CL": "CL",
        "C_L": "CL",
        "CT": "CT",
        "C_T": "CT",
        "CA": "CA",
        "C_A": "CA",
        "CT+CA": "CT+CA",
        "C_T+C_A": "CT+CA",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(f"Unknown channel label: {value!r}") from exc


@dataclass(frozen=True)
class CaptureSpec:
    interface: str
    endpoint_pairs: tuple[tuple[str, str], ...]
    capture_filter: str
    clock_source: str
    tls_session_policy: Literal["warm", "restart"]
    channel: Literal["CL", "CT", "CA", "CT+CA"]
    segmentation: Literal["passive", "oracle_segmented", "combined"]

    def validate(self) -> None:
        if not self.interface.strip():
            raise ValueError("Capture interface must be declared")
        if not self.endpoint_pairs:
            raise ValueError("At least one endpoint pair must be declared")
        channel = normalize_channel_label(self.channel)
        if channel in {"CT", "CA"} and self.segmentation == "combined":
            raise ValueError("A combined capture cannot be reported as CT or CA alone")


@dataclass(frozen=True)
class CaptureArtifact:
    path: Path
    sha256: str
    byte_count: int

    @classmethod
    def from_file(cls, path: str | Path) -> "CaptureArtifact":
        capture = Path(path)
        if not capture.is_file():
            raise FileNotFoundError(capture)
        return cls(path=capture, sha256=file_sha256(capture), byte_count=capture.stat().st_size)
