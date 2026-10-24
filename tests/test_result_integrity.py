from __future__ import annotations

import csv
import json

import pytest

from intent_leakage.experiments.synthetic_demo import run_synthetic_demo
from intent_leakage.orchestration.manifest import ManifestValidationError
from intent_leakage.orchestration.provenance import SYNTHETIC_NOTICE


def test_demo_outputs_cannot_be_mistaken_for_paper_results(tmp_path) -> None:
    config = {
        "seed": 5,
        "synthetic": {"runs_per_cell": 1, "random_forest_trees": 5},
        "output": {},
    }
    artifacts = run_synthetic_demo(config, output_dir=tmp_path / "demo")
    metadata = json.loads(artifacts["metadata"].read_text(encoding="utf-8"))
    assert metadata["synthetic"] is True
    assert metadata["notice"] == SYNTHETIC_NOTICE
    assert metadata["paper_results_claimed"] is False
    assert metadata["package_version"]
    resolved = json.loads(artifacts["resolved_config"].read_text(encoding="utf-8"))
    assert resolved["seed"] == 5
    assert artifacts["shaping_smoke"].is_file()
    bounds = json.loads(artifacts["bounds"].read_text(encoding="utf-8"))
    assert bounds["synthetic"] is True
    assert bounds["paper_results_claimed"] is False
    assert bounds["notice"] == SYNTHETIC_NOTICE
    with artifacts["metrics"].open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["synthetic"].lower() == "true" for row in rows)
    assert all(row["notice"] == SYNTHETIC_NOTICE for row in rows)
    assert all("paper result" in row["notice"] for row in rows)
    with artifacts["features"].open("r", encoding="utf-8", newline="") as stream:
        feature = next(csv.DictReader(stream))
    assert feature["protocol_block"] == "synthetic_core"
    assert feature["topology_id"] == "TA"
    assert feature["model_id"] == "synthetic-model-A"


def test_synthetic_demo_rejects_literal_secret_before_writing(tmp_path) -> None:
    destination = tmp_path / "must-not-exist"
    config = {
        "seed": 5,
        "synthetic": True,
        "password": "never-serialize-me",
        "output": {"directory": str(destination)},
    }
    with pytest.raises(ManifestValidationError, match="literal credential") as error:
        run_synthetic_demo(config)
    assert "never-serialize-me" not in str(error.value)
    assert not destination.exists()


def test_synthetic_demo_rejects_partial_paper_catalogue(tmp_path) -> None:
    config = {
        "seed": 5,
        "synthetic": True,
        "data": {"classes": ["I01", "I02"]},
        "output": {"directory": str(tmp_path / "partial")},
    }
    with pytest.raises(ValueError, match="complete paper catalogue"):
        run_synthetic_demo(config)
