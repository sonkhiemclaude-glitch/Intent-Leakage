from __future__ import annotations

import csv

import pytest

from intent_leakage.adapters.base import ExternalExecutionDisabledError
from intent_leakage.adapters.bedrock import BedrockAdapter, BedrockConfig
from intent_leakage.experiments.runner import run_experiment
from intent_leakage.experiments.train_attackers import (
    train_grouped_random_forest_attackers_from_csv,
)
from intent_leakage.orchestration.manifest import (
    ManifestValidationError,
    validate_manifest,
)


def _real_manifest() -> dict:
    return {
        "seed": 7,
        "synthetic": {"enabled": False},
        "data": {"root": "/already/provisioned/data"},
        "experiment": {"repetitions": 2, "timeout_seconds": 120},
        "models": {
            "A": {"model_id": "provider.model-a-v1"},
            "B": {"model_id": "provider.model-b-v1"},
        },
        "shaping": {"groups": [["I01", "I02"]]},
        "evaluation": {"bootstrap_resamples": 1000, "confidence_level": 0.95},
        "output": {"directory": "outputs/real"},
    }


def test_valid_frozen_real_manifest_passes() -> None:
    assert validate_manifest(_real_manifest(), real_run=True) == []


def test_real_manifest_rejects_placeholders() -> None:
    manifest = _real_manifest()
    manifest["models"]["A"]["model_id"] = "[BEDROCK-MODEL-ID-A]"
    with pytest.raises(ManifestValidationError, match="placeholders"):
        validate_manifest(manifest, real_run=True)


def test_manifest_rejects_literal_credentials_without_echoing_value() -> None:
    manifest = _real_manifest()
    manifest["bedrock"] = {"api_key": "do-not-print-this"}
    with pytest.raises(ManifestValidationError, match="literal credential") as error:
        validate_manifest(manifest, real_run=True)
    assert "do-not-print-this" not in str(error.value)


@pytest.mark.parametrize(
    "field",
    ("aws_access_key_id", "authorization", "auth_header", "session_token"),
)
def test_manifest_rejects_common_authorization_fields(field: str) -> None:
    manifest = _real_manifest()
    manifest["connection"] = {field: "never-serialize-this"}
    with pytest.raises(ManifestValidationError, match="literal credential") as error:
        validate_manifest(manifest, real_run=True)
    assert "never-serialize-this" not in str(error.value)


def test_external_adapter_is_disabled_by_default() -> None:
    config = BedrockConfig(
        model_id="provider.model-v1",
        region="test-region-1",
        streaming=False,
        decoding_parameters={},
        max_response_tokens=32,
        retry_policy={},
        connection_pooling="disabled",
    )
    with pytest.raises(ExternalExecutionDisabledError):
        BedrockAdapter(config)


def test_false_synthetic_mapping_never_enters_synthetic_runner(tmp_path) -> None:
    manifest = _real_manifest()
    manifest["output"] = {"root": str(tmp_path / "real")}
    validate_manifest(manifest, real_run=True)
    with pytest.raises(ValueError, match=r"data\.feature_csv"):
        run_experiment(manifest, stage="evaluate")
    assert not (tmp_path / "real" / "synthetic_features.csv").exists()


def test_real_manifest_rejects_enabled_synthetic_mode() -> None:
    manifest = _real_manifest()
    manifest["synthetic"] = {"enabled": True}
    with pytest.raises(ManifestValidationError, match="real-run manifest"):
        validate_manifest(manifest, real_run=True)


def test_real_offline_analysis_rejects_synthetic_fixture_data(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    features = data_root / "features.csv"
    with features.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "f1",
                "f2",
                "class_id",
                "split",
                "condition",
                "protocol_block",
                "topology_id",
                "model_id",
                "synthetic",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"f1": 0, "f2": 0, "class_id": "I01", "split": "train", "condition": "no_shaping", "protocol_block": "core", "topology_id": "TA", "model_id": "provider.model-a-v1", "synthetic": True},
                {"f1": 1, "f2": 1, "class_id": "I02", "split": "train", "condition": "no_shaping", "protocol_block": "core", "topology_id": "TA", "model_id": "provider.model-a-v1", "synthetic": True},
                {"f1": 0.1, "f2": 0, "class_id": "I01", "split": "test", "condition": "no_shaping", "protocol_block": "core", "topology_id": "TA", "model_id": "provider.model-a-v1", "synthetic": True},
            ]
        )

    manifest = _real_manifest()
    manifest["data"] = {
        "root": str(data_root),
        "feature_csv": features.name,
    }
    manifest["models"]["random_forest"] = {
        "feature_columns": ["f1", "f2"],
        "group_columns": ["condition"],
        "n_estimators": 5,
    }
    manifest["evaluation"]["prediction_source"] = "calibrated_model"
    manifest["output"] = {"root": str(tmp_path / "outputs")}
    with pytest.raises(ValueError, match="run-kind mismatch"):
        run_experiment(manifest, stage="calibrate")
    assert not (tmp_path / "outputs" / "synthetic_features.csv").exists()


def test_grouped_trainer_rejects_synthetic_relabeling_and_forbidden_features(
    tmp_path,
) -> None:
    path = tmp_path / "features.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["f1", "class_id", "split", "synthetic"],
        )
        writer.writeheader()
        writer.writerow(
            {"f1": 0.0, "class_id": "I01", "split": "train", "synthetic": True}
        )
    with pytest.raises(ValueError, match="run-kind mismatch"):
        train_grouped_random_forest_attackers_from_csv(
            path,
            tmp_path / "model.joblib",
            feature_columns_by_view={"fused": ["f1"]},
            group_columns=(),
            synthetic=False,
        )
    with pytest.raises(ValueError, match="forbidden"):
        train_grouped_random_forest_attackers_from_csv(
            path,
            tmp_path / "model.joblib",
            feature_columns_by_view={"fused": ["class_id"]},
            group_columns=(),
            synthetic=True,
        )
