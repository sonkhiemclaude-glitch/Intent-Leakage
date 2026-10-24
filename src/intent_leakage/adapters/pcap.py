"""Packet-capture parsing contracts.

Production parsing must reassemble TCP before parsing visible TLS record headers.
This repository never pretends that packet length and TLS-record length are the same.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AdapterNotProvisionedError


class PcapParserAdapter(ABC):
    @abstractmethod
    def parse(
        self,
        capture_path: str | Path,
        *,
        flow_manifest: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """Return normalized trace-event mappings from a packet capture."""


class UnconfiguredPcapParser(PcapParserAdapter):
    def parse(
        self,
        capture_path: str | Path,
        *,
        flow_manifest: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        del flow_manifest
        path = Path(capture_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        raise AdapterNotProvisionedError(
            "Select and validate a TCP reassembly/TLS-record parser backend "
            "(for example an audited tshark invocation) on the Mac Studio."
        )
