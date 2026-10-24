"""Shared validation primitives."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PLACEHOLDER = re.compile(
    r"(?:\[\s*TO\s+CONFIRM|\bTO\s*DO\b|\bTBD\b|BEDROCK-MODEL-ID|"
    r"/ABSOLUTE/PATH|MANUALLY/PROVISIONED)",
    flags=re.IGNORECASE,
)


def find_placeholders(value: Any, path: str = "$" ) -> list[str]:
    """Return configuration paths containing unresolved manuscript placeholders."""

    found: list[str] = []
    if isinstance(value, str):
        if _PLACEHOLDER.search(value):
            found.append(path)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(find_placeholders(child, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{path}[{index}]"))
    return found


def find_literal_secret_fields(value: Any, path: str = "$") -> list[str]:
    """Return paths whose key names suggest an embedded credential value.

    Environment-variable *names* such as ``password_env`` are permitted; the
    corresponding secret values must never appear in a manifest.
    """

    found: list[str] = []
    secret_key = re.compile(
        r"(?:^|[_-])(?:password|passwd|secret|token|api[_-]?key|"
        r"access[_-]?key(?:[_-]?id)?|credential|private[_-]?key|"
        r"authorization|auth(?:entication)?[_-]?header|bearer|cookie)"
        r"(?:$|[_-])",
        flags=re.IGNORECASE,
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if (
                secret_key.search(key_text)
                and not key_text.lower().endswith(("_env", "_env_var", "_environment"))
                and child not in (None, "", False)
            ):
                found.append(child_path)
            found.extend(find_literal_secret_fields(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_literal_secret_fields(child, f"{path}[{index}]"))
    return found


def ensure_relative_to(path: str | Path, root: str | Path) -> Path:
    """Resolve ``path`` and require it to remain under ``root``."""

    resolved = Path(path).expanduser().resolve()
    allowed_root = Path(root).expanduser().resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"Path {resolved} is outside allowed root {allowed_root}") from exc
    return resolved


def validate_run_kind_values(
    values: Sequence[Any],
    *,
    expected_synthetic: bool,
    context: str,
) -> None:
    """Require an explicit, uniform synthetic flag matching the caller."""

    if len(values) == 0:
        raise ValueError(f"{context} contains no run-kind values")
    parsed: list[bool] = []
    for value in values:
        if isinstance(value, bool):
            parsed.append(value)
        elif isinstance(value, int) and value in {0, 1}:
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            parsed.append(value.strip().lower() == "true")
        else:
            raise ValueError(
                f"{context} synthetic flags must be booleans, got {value!r}"
            )
    if any(value != expected_synthetic for value in parsed):
        expected = "synthetic" if expected_synthetic else "real"
        raise ValueError(f"{context} run-kind mismatch: expected only {expected} rows")
