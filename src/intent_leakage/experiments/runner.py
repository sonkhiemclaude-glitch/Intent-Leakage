"""Top-level experiment dispatcher with external execution disabled by default."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from ..adapters.base import AdapterNotProvisionedError, require_external_execution
from ..orchestration.grid import build_minimum_grid
from ..orchestration.manifest import synthetic_mode_enabled, validate_manifest
from ..orchestration.provenance import collect_provenance
from ..utils.hashing import file_sha256, stable_digest
from ..utils.io import atomic_write_json
from ..utils.validation import ensure_relative_to
from .ablation import summarize_ablation
from .evaluate import evaluate_prediction_groups
from .synthetic_demo import run_synthetic_demo
from .train_attackers import (
    predict_grouped_random_forest_attackers_from_csv,
    train_grouped_random_forest_attackers_from_csv,
)


def _core_environment(config: Mapping[str, Any]) -> dict[str, str | None]:
    experiment = config.get('experiment', {})
    topologies = experiment.get('topologies', {})
    primary = topologies.get('primary') if isinstance(topologies, Mapping) else None
    models = config.get('models', {})
    models = models if isinstance(models, Mapping) else {}
    model = models.get('A') or models.get('a')
    if isinstance(model, Mapping):
        model = model.get('model_id') or model.get('id')
    if not model:
        model = experiment.get('models', {}).get('primary')
    # The trainer checks input run kind before enforcing missing protocol metadata.
    return {'core_topology': primary if isinstance(primary, str) else None,
            'core_model': model if isinstance(model, str) else None}


def _configured_output(config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    output = config.get("output", {})
    if isinstance(output, str):
        return Path(output)
    if isinstance(output, Mapping):
        for key in ("directory", "dir", "root", "path"):
            if output.get(key):
                return Path(str(output[key]))
    return Path("outputs")


def _overwrite_enabled(config: Mapping[str, Any]) -> bool:
    output = config.get("output", {})
    if isinstance(output, str):
        return False
    if not isinstance(output, Mapping):
        raise ValueError("output must be a path string or mapping")
    value = output.get("overwrite", False)
    if not isinstance(value, bool):
        raise ValueError("output.overwrite must be boolean")
    return value


def _guard_output_targets(config: Mapping[str, Any], *paths: Path) -> None:
    if _overwrite_enabled(config):
        return
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s); set output.overwrite=true "
            f"only for an intentional rerun: {existing}"
        )


def _provisioned_data_file(config: Mapping[str, Any], key: str) -> Path:
    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    root_value = data.get("root")
    relative_value = data.get(key)
    if not root_value or not relative_value:
        raise ValueError(f"data.root and data.{key} are required for this stage")
    root = Path(str(root_value)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"manually provisioned data root not found: {root}")
    candidate = Path(str(relative_value))
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = ensure_relative_to(candidate, root)
    if not resolved.is_file():
        raise FileNotFoundError(f"manually provisioned data file not found: {resolved}")
    return resolved


def _write_offline_provenance(
    config: Mapping[str, Any],
    destination: Path,
    *,
    stage: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> Path:
    metadata = collect_provenance(
        config, synthetic=False, seed=int(config.get("seed", 0))
    )
    metadata.update(
        {
            "stage": stage,
            "external_actions_attempted": False,
            "paper_results_claimed": False,
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in inputs.items()
            },
            "outputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in outputs.items()
            },
        }
    )
    return atomic_write_json(destination / f"{stage}_provenance.json", metadata)


def _write_synthetic_stage_metadata(
    config: Mapping[str, Any],
    destination: Path,
    *,
    stage: str,
    filename: str,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    details: Mapping[str, Any] | None = None,
) -> Path:
    """Link a synthetic staged artifact to its inputs, config, and hashes."""

    metadata = collect_provenance(
        config, synthetic=True, seed=int(config.get("seed", 0))
    )
    metadata.update(
        {
            "stage": stage,
            "external_actions_attempted": False,
            "paper_results_claimed": False,
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in inputs.items()
            },
            "outputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in outputs.items()
            },
        }
    )
    if details:
        metadata["details"] = dict(details)
    return atomic_write_json(destination / filename, metadata)


def _feature_columns_by_view(rf_config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    configured = rf_config.get("feature_columns_by_view")
    if isinstance(configured, Mapping) and configured:
        result: dict[str, tuple[str, ...]] = {}
        for view, columns in configured.items():
            if not isinstance(columns, (list, tuple)) or not columns:
                raise ValueError(
                    "models.random_forest.feature_columns_by_view values must be nonempty lists"
                )
            result[str(view)] = tuple(str(value) for value in columns)
        return result
    legacy = rf_config.get("feature_columns")
    if isinstance(legacy, (list, tuple)) and legacy:
        return {str(rf_config.get("view", "fused")): tuple(str(value) for value in legacy)}
    raise ValueError(
        "models.random_forest.feature_columns_by_view or feature_columns is required"
    )


def _group_columns(rf_config: Mapping[str, Any]) -> tuple[str, ...]:
    value = rf_config.get("group_columns", ("condition",))
    if not isinstance(value, (list, tuple)):
        raise ValueError("models.random_forest.group_columns must be a list")
    return tuple(str(column) for column in value)


def run_experiment(
    config: Mapping[str, Any],
    *,
    stage: str,
    output_dir: str | Path | None = None,
    allow_external: bool = False,
) -> dict[str, Path]:
    """Dispatch a plan, synthetic smoke test, or gated real-experiment stage."""

    stage = stage.lower()
    destination = _configured_output(config, output_dir)
    if stage == "synthetic":
        return run_synthetic_demo(config, output_dir=destination)
    if stage == "plan":
        plan_is_synthetic = synthetic_mode_enabled(config)
        validate_manifest(config, real_run=not plan_is_synthetic)
        experiment = config.get("experiment", {})
        experiment = experiment if isinstance(experiment, Mapping) else {}
        repetitions = int(experiment.get("repetitions", config.get("repetitions", 1)))
        topologies = experiment.get("topologies", {})
        topologies = topologies if isinstance(topologies, Mapping) else {}
        topology_a = str(topologies.get("primary", experiment.get("topology_a", "TA")))
        topology_b = topologies.get("transfer", experiment.get("topology_b", "TB"))
        if experiment.get("include_topology_transfer", True) is False:
            topology_b = None
        models = experiment.get("models", {})
        models = models if isinstance(models, Mapping) else {}
        model_a = str(models.get("primary", experiment.get("model_a", "A")))
        model_b = str(models.get("transfer", experiment.get("model_b", "B")))
        phrase_names = config.get("phrase_families")
        phrase_names = phrase_names if isinstance(phrase_names, Mapping) else None
        grid = build_minimum_grid(
            repetitions,
            topology_a=topology_a,
            topology_b=str(topology_b) if topology_b is not None else None,
            model_a=model_a,
            model_b=model_b,
            phrase_family_names=phrase_names,
        )
        destination.mkdir(parents=True, exist_ok=True)
        grid_path = destination / "minimum_grid.csv"
        summary_path = destination / "minimum_grid_summary.json"
        provenance_path = destination / "plan_provenance.json"
        _guard_output_targets(config, grid_path, summary_path, provenance_path)
        with grid_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(grid[0].to_dict()))
            writer.writeheader()
            writer.writerows(row.to_dict() for row in grid)
        summary = {
            "run_count": len(grid),
            "repetitions": repetitions,
            "topology_transfer": topology_b is not None,
            "captures_started": False,
            "paper_results_claimed": False,
            "synthetic": plan_is_synthetic,
            "config_sha256": stable_digest(config),
            "grid_sha256": file_sha256(grid_path),
        }
        atomic_write_json(summary_path, summary)
        provenance = collect_provenance(
            config,
            synthetic=plan_is_synthetic,
            seed=int(config.get("seed", 0)),
        )
        provenance.update(
            {
                "stage": "plan",
                "captures_started": False,
                "paper_results_claimed": False,
                "outputs": {
                    "grid": {"path": str(grid_path), "sha256": file_sha256(grid_path)},
                    "summary": {
                        "path": str(summary_path),
                        "sha256": file_sha256(summary_path),
                    },
                },
            }
        )
        atomic_write_json(provenance_path, provenance)
        return {
            "grid": grid_path,
            "summary": summary_path,
            "provenance": provenance_path,
        }

    if stage not in {"calibrate", "evaluate", "ablation", "capture"}:
        raise ValueError(
            "stage must be one of synthetic, plan, calibrate, evaluate, ablation, capture"
        )
    synthetic_mode = synthetic_mode_enabled(config)
    if synthetic_mode and stage != "capture":
        # These local stages exercise the same persisted-artifact interfaces as a
        # later real run. They are always conspicuously marked and never invoke an
        # adapter. Generate the deterministic fixture once if it is not present.
        destination.mkdir(parents=True, exist_ok=True)
        feature_path = destination / "synthetic_features.csv"
        prediction_path = destination / "synthetic_predictions.csv"
        metrics_path = destination / "metrics.csv"
        metadata_path = destination / "run_metadata.json"
        cached_metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                cached_metadata = (
                    loaded_metadata if isinstance(loaded_metadata, dict) else {}
                )
            except (json.JSONDecodeError, OSError, AttributeError):
                cached_metadata = {}
        from .synthetic_demo import _SYNTHETIC_GENERATOR_VERSION

        required_artifacts = {
            "features": feature_path,
            "predictions": prediction_path,
            "metrics": metrics_path,
        }
        artifact_metadata = cached_metadata.get("artifacts", {})
        cache_valid = (
            cached_metadata.get("synthetic") is True
            and cached_metadata.get("paper_results_claimed") is False
            and cached_metadata.get("config_sha256") == stable_digest(config)
            and cached_metadata.get("generator_version")
            == _SYNTHETIC_GENERATOR_VERSION
            and isinstance(artifact_metadata, Mapping)
        )
        if cache_valid:
            for name, path in required_artifacts.items():
                record = artifact_metadata.get(name)
                if (
                    not path.is_file()
                    or not isinstance(record, Mapping)
                    or record.get("path") != path.name
                    or record.get("sha256") != file_sha256(path)
                ):
                    cache_valid = False
                    break
        if not cache_valid:
            run_synthetic_demo(config, output_dir=destination)

        if stage == "calibrate":
            from .synthetic_demo import _VIEW_COLUMNS

            model_path = destination / "models" / "grouped_random_forests.joblib"
            model_config = config.get("models", {})
            model_config = model_config if isinstance(model_config, Mapping) else {}
            rf_config = model_config.get("random_forest", {})
            rf_config = rf_config if isinstance(rf_config, Mapping) else {}
            synthetic_config = config.get("synthetic", {})
            synthetic_config = (
                synthetic_config if isinstance(synthetic_config, Mapping) else {}
            )
            experiment_config = config.get("experiment", {})
            experiment_config = (
                experiment_config
                if isinstance(experiment_config, Mapping)
                else {}
            )
            selected_views = tuple(
                str(value)
                for value in experiment_config.get("views", tuple(_VIEW_COLUMNS))
            )
            calibration_metadata = destination / "calibration_metadata.json"
            _guard_output_targets(config, model_path, calibration_metadata)
            train_grouped_random_forest_attackers_from_csv(
                feature_path,
                model_path,
                feature_columns_by_view={
                    view: _VIEW_COLUMNS[view] for view in selected_views
                },
                group_columns=("condition",),
                split_column="partition",
                seed=int(rf_config.get("random_state", config.get("seed", 0))),
                n_estimators=int(
                    synthetic_config.get(
                        "random_forest_trees", rf_config.get("n_estimators", 40)
                    )
                ),
                max_depth=(
                    None
                    if rf_config.get("max_depth") is None
                    else int(rf_config["max_depth"])
                ),
                min_samples_leaf=int(rf_config.get("min_samples_leaf", 1)),
                class_weight=rf_config.get("class_weight"),
                scope_class_weight=rf_config.get("scope_class_weight"),
                synthetic=True,
                config_sha256=stable_digest(config),
            )
            _write_synthetic_stage_metadata(
                config,
                destination,
                stage="calibrate",
                filename=calibration_metadata.name,
                inputs={"features": feature_path},
                outputs={"model": model_path},
                details={
                    "training_split": "train",
                    "views": list(selected_views),
                    "random_forest_hyperparameters": {
                        "n_estimators": int(
                            synthetic_config.get(
                                "random_forest_trees",
                                rf_config.get("n_estimators", 40),
                            )
                        ),
                        "max_depth": rf_config.get("max_depth"),
                        "min_samples_leaf": int(
                            rf_config.get("min_samples_leaf", 1)
                        ),
                        "class_weight": rf_config.get("class_weight"),
                        "scope_class_weight": rf_config.get(
                            "scope_class_weight"
                        ),
                        "random_state": int(
                            rf_config.get(
                                "random_state", config.get("seed", 0)
                            )
                        ),
                    },
                },
            )
            return {"model": model_path, "metadata": calibration_metadata}

        if stage == "evaluate":
            model_path = destination / "models" / "grouped_random_forests.joblib"
            if not model_path.is_file():
                raise FileNotFoundError(
                    "Run the calibrate stage before evaluate; grouped model is missing"
                )
            staged_predictions = destination / "staged_predictions.csv"
            evaluation_path = destination / "evaluation.csv"
            evaluation_metadata = destination / "evaluation_metadata.json"
            _guard_output_targets(
                config, staged_predictions, evaluation_path, evaluation_metadata
            )
            predict_grouped_random_forest_attackers_from_csv(
                feature_path,
                model_path,
                staged_predictions,
                evaluation_splits=("test",),
                synthetic=True,
                expected_config_sha256=stable_digest(config),
            )
            evaluate_prediction_groups(
                staged_predictions, evaluation_path, synthetic=True
            )
            _write_synthetic_stage_metadata(
                config,
                destination,
                stage="evaluate",
                filename=evaluation_metadata.name,
                inputs={"features": feature_path, "model": model_path},
                outputs={
                    "predictions": staged_predictions,
                    "evaluation": evaluation_path,
                },
                details={"evaluation_splits": ["test"]},
            )
            return {
                "predictions": staged_predictions,
                "evaluation": evaluation_path,
                "metadata": evaluation_metadata,
            }

        ablation_path = destination / "ablation.csv"
        ablation_metadata = destination / "ablation_metadata.json"
        _guard_output_targets(config, ablation_path, ablation_metadata)
        summarize_ablation(metrics_path, ablation_path, synthetic=True)
        _write_synthetic_stage_metadata(
            config,
            destination,
            stage="ablation",
            filename=ablation_metadata.name,
            inputs={"metrics": metrics_path},
            outputs={"ablation": ablation_path},
            details={
                "limitation": (
                    "descriptive re-summary only; this stage does not execute "
                    "the paper's paired ablation factors"
                )
            },
        )
        return {"ablation": ablation_path, "metadata": ablation_metadata}

    validate_manifest(config, real_run=True)
    if stage == "capture":
        require_external_execution(allow_external)
        raise AdapterNotProvisionedError(
            "Real capture is gated correctly but cannot run until concrete Mac "
            "Studio Bedrock ONOS Mininet MCP and capture adapters are provisioned."
        )

    # Calibration, evaluation, and ablation are offline operations over files
    # already provisioned beneath data.root. They never need external authority.
    destination.mkdir(parents=True, exist_ok=True)
    models = config.get("models")
    models = models if isinstance(models, Mapping) else {}
    rf_config = models.get("random_forest", {})
    rf_config = rf_config if isinstance(rf_config, Mapping) else {}
    if stage == "calibrate":
        feature_path = _provisioned_data_file(config, "feature_csv")
        model_path = destination / "models" / "grouped_random_forests.joblib"
        provenance_path = destination / "calibrate_provenance.json"
        _guard_output_targets(config, model_path, provenance_path)
        train_grouped_random_forest_attackers_from_csv(
            feature_path,
            model_path,
            feature_columns_by_view=_feature_columns_by_view(rf_config),
            group_columns=_group_columns(rf_config),
            target_column=str(rf_config.get("target_column", "class_id")),
            scope_column=str(rf_config.get("scope_column", "scope_q")),
            split_column=str(rf_config.get("split_column", "split")),
            training_split=str(rf_config.get("training_split", "train")),
            seed=int(rf_config.get("random_state", config.get("seed", 0))),
            n_estimators=int(rf_config.get("n_estimators", 200)),
            max_depth=(
                None
                if rf_config.get("max_depth") is None
                else int(rf_config["max_depth"])
            ),
            min_samples_leaf=int(rf_config.get("min_samples_leaf", 1)),
            class_weight=rf_config.get("class_weight"),
            scope_class_weight=rf_config.get("scope_class_weight"),
            synthetic=False,
            **_core_environment(config),
            config_sha256=stable_digest(config),
        )
        provenance = _write_offline_provenance(
            config,
            destination,
            stage=stage,
            inputs={"feature_csv": feature_path},
            outputs={"model": model_path},
        )
        return {"model": model_path, "provenance": provenance}

    if stage == "evaluate":
        evaluation = config.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, Mapping) else {}
        prediction_source = str(
            evaluation.get("prediction_source", "calibrated_model")
        )
        raw_splits = evaluation.get("evaluation_splits", ("test",))
        if not isinstance(raw_splits, (list, tuple)) or not raw_splits:
            raise ValueError("evaluation.evaluation_splits must be a nonempty list")
        evaluation_splits = tuple(str(value) for value in raw_splits)
        inputs: dict[str, Path]
        if prediction_source == "calibrated_model":
            feature_path = _provisioned_data_file(config, "feature_csv")
            model_path = destination / "models" / "grouped_random_forests.joblib"
            if not model_path.is_file():
                raise FileNotFoundError(
                    "Run the calibrate stage before evaluate; grouped model is missing"
                )
            prediction_path = destination / "predictions.csv"
            evaluation_path = destination / "evaluation.csv"
            provenance_path = destination / "evaluate_provenance.json"
            _guard_output_targets(
                config, prediction_path, evaluation_path, provenance_path
            )
            predict_grouped_random_forest_attackers_from_csv(
                feature_path,
                model_path,
                prediction_path,
                evaluation_splits=evaluation_splits,
                synthetic=False,
                expected_config_sha256=stable_digest(config),
            )
            inputs = {"feature_csv": feature_path, "model": model_path}
        elif prediction_source == "provisioned_csv":
            prediction_path = _provisioned_data_file(config, "prediction_csv")
            inputs = {"prediction_csv": prediction_path}
            evaluation_path = destination / "evaluation.csv"
            provenance_path = destination / "evaluate_provenance.json"
            _guard_output_targets(config, evaluation_path, provenance_path)
        else:
            raise ValueError(
                "evaluation.prediction_source must be calibrated_model or provisioned_csv"
            )
        evaluate_prediction_groups(
            prediction_path,
            evaluation_path,
            required_group_columns=(
                "protocol_block",
                "topology_id",
                "model_id",
                "view",
                "condition",
                "attacker",
            ),
            evaluation_splits=evaluation_splits,
            synthetic=False,
        )
        provenance = _write_offline_provenance(
            config,
            destination,
            stage=stage,
            inputs=inputs,
            outputs=(
                {"predictions": prediction_path, "evaluation": evaluation_path}
                if prediction_source == "calibrated_model"
                else {"evaluation": evaluation_path}
            ),
        )
        return {
            "predictions": prediction_path,
            "evaluation": evaluation_path,
            "provenance": provenance,
        }

    evaluation = config.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, Mapping) else {}
    ablation_source = str(evaluation.get("ablation_source", "evaluation_output"))
    if ablation_source == "evaluation_output":
        metrics_path = destination / "evaluation.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(
                "Run the evaluate stage before ablation; evaluation.csv is missing"
            )
    elif ablation_source == "provisioned_metrics":
        metrics_path = _provisioned_data_file(config, "metrics_csv")
    else:
        raise ValueError(
            "evaluation.ablation_source must be evaluation_output or provisioned_metrics"
        )
    ablation_path = destination / "ablation.csv"
    provenance_path = destination / "ablation_provenance.json"
    _guard_output_targets(config, ablation_path, provenance_path)
    summarize_ablation(
        metrics_path,
        ablation_path,
        required_group_columns=(
            "protocol_block",
            "topology_id",
            "model_id",
            "view",
            "condition",
            "attacker",
        ),
        synthetic=False,
    )
    provenance = _write_offline_provenance(
        config,
        destination,
        stage=stage,
        inputs={"metrics_csv": metrics_path},
        outputs={"ablation": ablation_path},
    )
    return {"ablation": ablation_path, "provenance": provenance}
