"""Stable hashes for frozen configurations and result provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_digest(value: Any) -> str:
    """Return a SHA-256 digest of a JSON-serializable value.

    Mapping keys are sorted and whitespace is normalized so semantically identical
    configuration mappings have the same digest.
    """

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading a potentially large capture into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
