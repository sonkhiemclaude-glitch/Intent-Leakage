"""Offline random-forest fitting on an already prepared feature CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..orchestration.provenance import SYNTHETIC_NOTICE
from ..schemas import IntentClass
from ..utils.hashing import file_sha256
from ..utils.validation import validate_run_kind_values


FORBIDDEN_MODEL_FEATURE_COLUMNS = frozenset(
    {
        "class_id",
        "scope",
        "scope_q",
        "scope_bucket",
        "split",
        "partition",
        "condition",
        "view",
        "attacker",
        "trace_id",
        "run_id",
        "sample_id",
        "workload_cell_id",
        "phrase_family_id",
        "intent_instance_id",
        "repetition",
        "repetition_seed",
        "capture_seed",
        "target_seed",
        "topology_id",
        "topology_seed",
        "model_id",
        "targets",
        "natural_language_intent",
        "plaintext",
        "phase_marker",
        "model_response_id",
        "synthetic",
        "notice",
        "paper_results_claimed",
    }
)


def train_random_forest_from_csv(
    input_path: str | Path,
    output_model: str | Path,
    *,
    feature_columns: Sequence[str],
    target_column: str = "class_id",
    split_column: str = "split",
    seed: int = 0,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | dict[Any, float] | None = None,
    synthetic: bool = False,
    core_topology: str | None = None,
    core_model: str | None = None,
) -> Path:
    """Fit only on rows explicitly marked ``train`` and serialize with joblib."""

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    frame = pd.read_csv(input_path)
    if "synthetic" not in frame:
        raise KeyError("Feature CSV must contain an explicit synthetic column")
    validate_run_kind_values(
        frame["synthetic"].tolist(),
        expected_synthetic=synthetic,
        context="feature CSV",
    )
    required = set(feature_columns) | {target_column, split_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Feature CSV is missing columns: {sorted(missing)}")
    if set(feature_columns) & (FORBIDDEN_MODEL_FEATURE_COLUMNS | {target_column,split_column}):
        raise ValueError('Forbidden ground-truth/provenance model feature')
    if not synthetic:
        from .protocol_guards import validate_real_attacker_protocol
        validate_real_attacker_protocol(frame,feature_columns_by_view={'fused':feature_columns},
            split_column=split_column,core_topology=core_topology,core_model=core_model)
    train = frame[frame[split_column] == "train"]
    if train.empty:
        raise ValueError("No training rows were supplied")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[list(feature_columns)], train[target_column])
    destination = Path(output_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(feature_columns),
            "target_column": target_column,
            "seed": seed,
            "random_forest_hyperparameters": {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "class_weight": class_weight,
            },
            "synthetic": synthetic,
            "paper_results_claimed": False,
            "notice": (
                "synthetic smoke-test output, not a paper result"
                if synthetic
                else "real-data model; inspect capture and run provenance"
            ),
        },
        destination,
    )
    return destination


def train_grouped_random_forest_attackers_from_csv(
    input_path: str | Path,
    output_model: str | Path,
    *,
    feature_columns_by_view: Mapping[str, Sequence[str]],
    group_columns: Sequence[str] = ("condition",),
    target_column: str = "class_id",
    scope_column: str = "scope_q",
    split_column: str = "partition",
    training_split: str = "train",
    seed: int = 0,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight: str | dict[Any, float] | None = None,
    scope_class_weight: str | dict[Any, float] | None = None,
    synthetic: bool = False,
    config_sha256: str | None = None,
    core_topology: str | None = None,
    core_model: str | None = None,
) -> Path:
    """Fit one class-and-scope attacker per view and declared result group.

    This is the persisted calibration stage for paper Section 5.6.  Scope is
    fitted with :class:`TwoStagePredictedClassScopeModel`, so held-out scope
    inference uses predicted class and can never consume the true test class.
    The most-probable *training* class probability is stored with every model
    for the Section 5.5 prior-only baseline.
    """

    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from ..attackers.scope import TwoStagePredictedClassScopeModel

    if not feature_columns_by_view:
        raise ValueError("feature_columns_by_view must not be empty")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive or None")
    if min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive")
    frame = pd.read_csv(input_path)
    if "synthetic" not in frame:
        raise KeyError("Feature CSV must contain an explicit synthetic column")
    validate_run_kind_values(
        frame["synthetic"].tolist(),
        expected_synthetic=synthetic,
        context="feature CSV",
    )
    feature_columns = {
        str(view): tuple(str(column) for column in columns)
        for view, columns in feature_columns_by_view.items()
    }
    if any(not columns for columns in feature_columns.values()):
        raise ValueError("every view must declare at least one feature column")
    forbidden = (
        FORBIDDEN_MODEL_FEATURE_COLUMNS
        | {target_column, scope_column, split_column}
        | set(group_columns)
    )
    leaked = sorted(
        {
            column
            for columns in feature_columns.values()
            for column in columns
            if column in forbidden
        }
    )
    if leaked:
        raise ValueError(
            f"Configured attacker features include forbidden ground-truth/provenance columns: {leaked}"
        )
    required = {
        target_column,
        split_column,
        *group_columns,
        *(column for columns in feature_columns.values() for column in columns),
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Feature CSV is missing columns: {sorted(missing)}")
    protocol_guard = None
    if not synthetic:
        if training_split != 'train':raise ValueError('Real attackers fit only the train partition')
        from .protocol_guards import validate_real_attacker_protocol
        protocol_guard=validate_real_attacker_protocol(frame,feature_columns_by_view=feature_columns,
            split_column=split_column,core_topology=core_topology,core_model=core_model)
    train = frame[frame[split_column].astype(str) == str(training_split)]
    if train.empty:
        raise ValueError(f"No rows were supplied for training split {training_split!r}")
    valid_classes = {value.value for value in IntentClass}
    invalid_classes = sorted(
        set(train[target_column].astype(str).tolist()) - valid_classes
    )
    if invalid_classes:
        raise ValueError(
            f"Training target contains class IDs outside I01-I15: {invalid_classes}"
        )
    if scope_column in frame.columns:
        numeric_scope = pd.to_numeric(train[scope_column], errors="coerce")
        if numeric_scope.isna().any() or not numeric_scope.isin((1, 2, 4)).all():
            raise ValueError("Training scope values must be exactly q in {1, 2, 4}")

    if group_columns:
        group_records = train[list(group_columns)].drop_duplicates().to_dict("records")
    else:
        group_records = [{}]
    records: list[dict[str, Any]] = []
    for group_index, group_values in enumerate(group_records):
        group = train
        for column, value in group_values.items():
            group = group[group[column] == value]
        if group.empty:  # pragma: no cover - defensive after drop_duplicates
            continue
        counts = group[target_column].astype(str).value_counts(normalize=True)
        modal_probability = float(counts.max())
        modal_class = sorted(
            str(label) for label, probability in counts.items() if probability == modal_probability
        )[0]
        for view_index, (view, columns) in enumerate(feature_columns.items()):
            random_state = int(seed + 1000 * group_index + view_index)
            class_estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=1,
            )
            features = group[list(columns)]
            if scope_column in frame.columns:
                scope_estimator = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=scope_class_weight,
                    random_state=random_state + 100,
                    n_jobs=1,
                )
                estimator: Any = TwoStagePredictedClassScopeModel(
                    class_estimator,
                    scope_estimator,
                    task="classification",
                    random_state=random_state,
                ).fit(
                    features,
                    group[target_column],
                    pd.to_numeric(group[scope_column], errors="raise").astype(int),
                )
                has_scope = True
            else:
                estimator = class_estimator.fit(features, group[target_column])
                has_scope = False
            records.append(
                {
                    "view": view,
                    "group_values": dict(group_values),
                    "feature_columns": list(columns),
                    "training_modal_class": modal_class,
                    "training_modal_probability": modal_probability,
                    "training_class_counts": {
                        str(label): int(count)
                        for label, count in group[target_column].astype(str).value_counts().items()
                    },
                    "has_scope": has_scope,
                    "estimator": estimator,
                }
            )
    if not records:
        raise ValueError("No grouped attacker could be fitted")
    destination = Path(output_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "grouped_random_forest_attackers",
            "target_column": target_column,
            "scope_column": scope_column,
            "split_column": split_column,
            "training_split": training_split,
            "group_columns": list(group_columns),
            "seed": int(seed),
            "random_forest_hyperparameters": {
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "class_weight": class_weight,
                "scope_class_weight": scope_class_weight,
            },
            "config_sha256": config_sha256,
            "protocol_guard": protocol_guard,
            "input_sha256": file_sha256(input_path),
            "synthetic": synthetic,
            "paper_results_claimed": False,
            "notice": (
                SYNTHETIC_NOTICE
                if synthetic
                else "real-data model; inspect capture and run provenance"
            ),
            "records": records,
        },
        destination,
    )
    return destination


def predict_grouped_random_forest_attackers_from_csv(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_splits: Sequence[str] = ("test",),
    synthetic: bool = False,
    expected_config_sha256: str | None = None,
) -> Path:
    """Load calibrated attackers and write held-out predictions only."""

    import joblib
    import pandas as pd

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("kind") != "grouped_random_forest_attackers":
        raise ValueError("Unsupported or invalid grouped attacker bundle")
    if bool(bundle.get("synthetic")) != bool(synthetic):
        raise ValueError("Model/data run kind mismatch")
    if expected_config_sha256 is not None and bundle.get("config_sha256") != expected_config_sha256:
        raise ValueError("Calibrated model configuration hash does not match this run")
    if bundle.get("input_sha256") != file_sha256(input_path):
        raise ValueError("Feature CSV changed after attacker calibration")
    frame = pd.read_csv(input_path)
    if "synthetic" not in frame:
        raise KeyError("Feature CSV must contain an explicit synthetic column")
    validate_run_kind_values(
        frame["synthetic"].tolist(),
        expected_synthetic=synthetic,
        context="feature CSV",
    )
    target_column = str(bundle["target_column"])
    scope_column = str(bundle["scope_column"])
    split_column = str(bundle["split_column"])
    if not evaluation_splits:
        raise ValueError("evaluation_splits must not be empty")
    required = {target_column, split_column}
    if any(bool(record.get("has_scope")) for record in bundle["records"]):
        required.add(scope_column)
    for record in bundle["records"]:
        required.update(record["feature_columns"])
        required.update(record["group_values"])
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Feature CSV is missing columns: {sorted(missing)}")
    if not synthetic:
        from .protocol_guards import validate_real_attacker_protocol
        frozen=bundle.get('protocol_guard')
        if not frozen:raise ValueError('Real model bundle lacks frozen protocol guards')
        checked=validate_real_attacker_protocol(frame,
            feature_columns_by_view={r['view']:r['feature_columns'] for r in bundle['records']},
            split_column=split_column,core_topology=frozen['core_topology'],core_model=frozen['core_model'])
        if checked != frozen:raise ValueError('Frozen real protocol metadata changed')
    allowed_splits = {str(value) for value in evaluation_splits}
    passthrough = (
        "sample_id",
        "workload_cell_id",
        "trace_id",
        "run_id",
        "phrase_family_id",
        "intent_instance_id",
        "repetition",
        "repetition_seed",
        "capture_seed",
        "target_seed",
        "topology_id",
        "topology_seed",
        "model_id",
            "protocol_block",
            "held_out_condition",
        "split",
        "partition",
    )
    rows: list[dict[str, Any]] = []
    for record in bundle["records"]:
        group = frame[frame[split_column].astype(str).isin(allowed_splits)]
        for column, value in record["group_values"].items():
            group = group[group[column] == value]
        if group.empty:
            continue
        features = group[list(record["feature_columns"])]
        estimator = record["estimator"]
        if record["has_scope"]:
            predicted_class, predicted_scope = estimator.predict_with_classes(features)
        else:
            predicted_class = estimator.predict(features)
            predicted_scope = None
        for row_index, (_, source) in enumerate(group.iterrows()):
            output: dict[str, Any] = {
                target_column: str(source[target_column]),
                "predicted_class": str(predicted_class[row_index]),
                "view": str(record["view"]),
                "attacker": "random_forest",
                "prior_predicted_class": str(record["training_modal_class"]),
                "training_modal_probability": float(
                    record["training_modal_probability"]
                ),
                "synthetic": synthetic,
                "run_kind": (
                    "synthetic_smoke_test" if synthetic else "real_offline_analysis"
                ),
                "notice": (
                    SYNTHETIC_NOTICE
                    if synthetic
                    else "real-data prediction; inspect capture and model provenance"
                ),
                "paper_results_claimed": False,
            }
            output.update(record["group_values"])
            for column in passthrough:
                if column in source.index:
                    output[column] = source[column]
            if predicted_scope is not None:
                output[scope_column] = source[scope_column]
                output["predicted_scope_q"] = predicted_scope[row_index]
            rows.append(output)
    if not rows:
        raise ValueError(
            f"No rows matched evaluation splits {sorted(allowed_splits)} and model groups"
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination
