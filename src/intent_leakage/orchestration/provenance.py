"""Result provenance that contains no credentials or plaintext intents."""

from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone
from typing import Any, Mapping

from ..utils.hashing import stable_digest

SYNTHETIC_NOTICE = "synthetic smoke-test output, not a paper result"
REAL_RESULT_NOTICE = (
    "No paper-level claim is implied; verify the frozen manifest and provenance."
)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance(
    config: Mapping[str, Any],
    *,
    synthetic: bool,
    seed: int,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without serializing secrets or credentials."""

    moment = timestamp or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "synthetic": synthetic,
        "run_kind": "synthetic_smoke_test" if synthetic else "real_offline_analysis",
        "notice": SYNTHETIC_NOTICE if synthetic else REAL_RESULT_NOTICE,
        "created_utc": moment.astimezone(timezone.utc).isoformat(),
        "seed": seed,
        "config_sha256": stable_digest(config),
        "package_version": _version("intent-leakage") or "source-checkout",
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            name: _version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "matplotlib",
                "PyYAML",
                "torch",
                "boto3",
            )
        },
    }
