from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from intent_leakage.cli import main


def _write_config(path, output) -> None:
    path.write_text(
        json.dumps(
            {
                "seed": 13,
                "synthetic": {"runs_per_cell": 1, "random_forest_trees": 5},
                "output": {"directory": str(output)},
            }
        ),
        encoding="utf-8",
    )


def test_cli_help(capsys) -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "synthetic" in capsys.readouterr().out


def test_module_cli_help_from_subprocess() -> None:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        [sys.executable, "-m", "intent_leakage.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0
    assert "synthetic" in completed.stdout


def test_cli_runs_synthetic_demo(tmp_path) -> None:
    config = tmp_path / "config.json"
    output = tmp_path / "demo"
    _write_config(config, output)
    assert main(["synthetic", "--config", str(config)]) == 0
    assert (output / "metrics.csv").is_file()
    assert (output / "synthetic_metrics.csv").is_file()
    assert (output / "synthetic_confusion_fused.png").is_file()


def test_cli_materializes_grid_without_external_actions(tmp_path) -> None:
    config = tmp_path / "config.json"
    output = tmp_path / "plan"
    config.write_text(
        json.dumps(
            {
                "seed": 1,
                "synthetic": {},
                "experiment": {"repetitions": 1, "include_topology_transfer": False},
                "output": {"directory": str(output)},
            }
        ),
        encoding="utf-8",
    )
    assert main(["plan", "--config", str(config)]) == 0
    summary = json.loads((output / "minimum_grid_summary.json").read_text())
    assert summary["run_count"] == 585
    assert summary["captures_started"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(["plan", "--config", str(config)])


def test_synthetic_calibrate_evaluate_and_ablation_stages(tmp_path) -> None:
    import joblib

    config = tmp_path / "config.json"
    output = tmp_path / "staged"
    _write_config(config, output)
    for stage in ("calibrate", "evaluate", "ablation"):
        assert main(
            [
                "experiment",
                "--config",
                str(config),
                "--stage",
                stage,
            ]
        ) == 0
    assert (output / "models" / "grouped_random_forests.joblib").is_file()
    bundle = joblib.load(output / "models" / "grouped_random_forests.joblib")
    assert bundle["random_forest_hyperparameters"]["n_estimators"] == 5
    assert (output / "calibration_metadata.json").is_file()
    assert (output / "evaluation.csv").is_file()
    assert (output / "evaluation_metadata.json").is_file()
    assert (output / "staged_predictions.csv").is_file()
    assert (output / "ablation.csv").is_file()
    assert (output / "ablation_metadata.json").is_file()
    with (output / "metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        direct = {
            (row["condition"], row["view"]): row for row in csv.DictReader(stream)
        }
    with (output / "evaluation.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        staged = {
            (row["condition"], row["view"]): row for row in csv.DictReader(stream)
        }
    assert direct.keys() == staged.keys()
    for key in direct:
        for metric in (
            "accuracy",
            "macro_f1",
            "prior_accuracy",
            "advantage",
            "observed_mi_bits",
        ):
            assert float(direct[key][metric]) == pytest.approx(
                float(staged[key][metric])
            )
