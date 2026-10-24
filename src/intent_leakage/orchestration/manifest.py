"""Manifest validation and freezing for prospective experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from ..utils.hashing import stable_digest
from ..utils.io import atomic_write_json
from ..utils.validation import find_literal_secret_fields, find_placeholders

EXPECTED_TOP_LEVEL = {
    "seed",
    "synthetic",
    "data",
    "experiment",
    "models",
    "shaping",
    "evaluation",
    "output",
}


class ManifestValidationError(ValueError):
    """Raised when a synthetic or real-run manifest is not executable."""


def synthetic_mode_enabled(config: Mapping[str, Any]) -> bool:
    """Interpret the supported boolean or mapping-valued synthetic setting.

    A mapping without ``enabled`` is treated as enabled for existing smoke-test
    configurations. An explicit ``enabled: false`` is authoritative.
    """

    value = config.get("synthetic", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ManifestValidationError("synthetic.enabled must be boolean")
        return enabled
    raise ManifestValidationError("synthetic must be a boolean or mapping")


def _positive_int(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _model_ids(models: Any) -> tuple[str | None, str | None]:
    if not isinstance(models, Mapping):
        return None, None
    def get_one(key: str) -> str | None:
        value = models.get(key) or models.get(key.lower())
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            candidate = value.get("id") or value.get("model_id")
            return str(candidate) if candidate else None
        return None
    return get_one("A"), get_one("B")


def validate_manifest(config: Mapping[str, Any], *, real_run: bool = False) -> list[str]:
    """Validate a config and return non-fatal warnings.

    Real execution is stricter: every paper placeholder must be resolved and core
    runtime controls must be frozen. Synthetic smoke configs may omit irrelevant
    external-system fields.
    """

    if not isinstance(config, Mapping):
        raise ManifestValidationError("Manifest must be a mapping")
    warnings: list[str] = []
    secret_fields = find_literal_secret_fields(config)
    if secret_fields:
        raise ManifestValidationError(
            "Manifest contains literal credential fields; use environment-variable "
            "names or a private external file instead: " + ", ".join(secret_fields)
        )
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ManifestValidationError("seed must be a non-negative integer")

    missing_top = sorted(EXPECTED_TOP_LEVEL - set(config))
    if real_run and missing_top:
        raise ManifestValidationError(
            f"Real manifest is missing top-level sections: {missing_top}"
        )
    if not real_run and missing_top:
        warnings.append(f"Synthetic manifest omits unused sections: {missing_top}")

    experiment = config.get("experiment", {})
    if experiment and not isinstance(experiment, Mapping):
        raise ManifestValidationError("experiment must be a mapping")
    if real_run:
        if synthetic_mode_enabled(config):
            raise ManifestValidationError(
                "a real-run manifest must set synthetic to false or enabled: false"
            )
        placeholders = find_placeholders(config)
        if placeholders:
            raise ManifestValidationError(
                "Real manifest contains unresolved placeholders at: "
                + ", ".join(placeholders)
            )
        if not _positive_int(experiment, "repetitions"):
            raise ManifestValidationError("experiment.repetitions must be positive")
        timeout_key = (
            "timeout_seconds" if "timeout_seconds" in experiment else "timeout_s"
        )
        timeout = experiment.get(timeout_key)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ManifestValidationError("A positive experiment timeout is required")
        model_a, model_b = _model_ids(config.get("models"))
        if not model_a or not model_b:
            nested_models = experiment.get("models")
            if isinstance(nested_models, Mapping):
                primary = nested_models.get("primary")
                transfer = nested_models.get("transfer")
                model_a = str(primary) if primary else model_a
                model_b = str(transfer) if transfer else model_b
        if not model_a or not model_b or model_a == model_b:
            raise ManifestValidationError("Two distinct concrete Bedrock model IDs are required")
        nested_models = experiment.get("models")
        if isinstance(nested_models, Mapping):
            nested_a = nested_models.get("primary")
            nested_b = nested_models.get("transfer")
            if nested_a and str(nested_a) != model_a:
                raise ManifestValidationError(
                    "experiment.models.primary must match models.A.model_id"
                )
            if nested_b and str(nested_b) != model_b:
                raise ManifestValidationError(
                    "experiment.models.transfer must match models.B.model_id"
                )
        data = config.get("data")
        if not isinstance(data, Mapping) or not data.get("root"):
            raise ManifestValidationError("data.root is required for a real run")
        evaluation = config.get("evaluation")
        if not isinstance(evaluation, Mapping):
            raise ManifestValidationError("evaluation must be a mapping")
        if "bootstrap_resamples" not in evaluation:
            raise ManifestValidationError("evaluation.bootstrap_resamples must be frozen")
        if "confidence_level" not in evaluation and "confidence" not in evaluation:
            raise ManifestValidationError("evaluation.confidence must be frozen")
    else:
        placeholders = find_placeholders(config)
        if placeholders:
            warnings.append("Synthetic config contains ignored external placeholders")
    return warnings


def freeze_manifest(
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    real_run: bool = False,
) -> dict[str, Any]:
    """Validate, hash, and write an immutable manifest representation."""

    warnings = validate_manifest(config, real_run=real_run)
    frozen = deepcopy(dict(config))
    digest = stable_digest(frozen)
    payload = {
        "schema_version": 1,
        "manifest_sha256": digest,
        "real_run": real_run,
        "validation_warnings": warnings,
        "manifest": frozen,
    }
    atomic_write_json(output_path, payload)
    return payload
