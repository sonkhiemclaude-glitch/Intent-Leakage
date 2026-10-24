"""Frozen-grid construction, run manifests, and safe orchestration."""

from .grid import GridRun, build_minimum_grid, expected_run_count
from .manifest import (
    ManifestValidationError,
    freeze_manifest,
    synthetic_mode_enabled,
    validate_manifest,
)
from .provenance import SYNTHETIC_NOTICE, collect_provenance

__all__ = [
    "GridRun",
    "ManifestValidationError",
    "SYNTHETIC_NOTICE",
    "build_minimum_grid",
    "collect_provenance",
    "expected_run_count",
    "freeze_manifest",
    "synthetic_mode_enabled",
    "validate_manifest",
]
