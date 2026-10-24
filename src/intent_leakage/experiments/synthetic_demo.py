"""Lightweight, deterministic synthetic smoke test.

The generated records exercise the data, attack, metric, provenance, and plotting
paths. They are conspicuously labelled and are never paper results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

from ..orchestration.grid import INTENT_CLASSES, SCOPE_COUNTS
from ..orchestration.manifest import validate_manifest
from ..orchestration.provenance import SYNTHETIC_NOTICE, collect_provenance
from ..plotting.confusion import plot_confusion_from_csv
from ..utils.hashing import file_sha256, stable_digest
from ..utils.io import atomic_write_json
from ..utils.random import seed_everything

_VIEW_COLUMNS = {
    "CL": ("cl_bytes", "cl_records", "cl_duration"),
    "CT": ("ct_bytes", "ct_records", "ct_duration"),
    "CA": ("ca_bytes", "ca_records", "ca_duration"),
    "fused": (
        "cl_bytes",
        "cl_records",
        "cl_duration",
        "ct_bytes",
        "ct_records",
        "ct_duration",
        "ca_bytes",
        "ca_records",
        "ca_duration",
    ),
}

_SUPPORTED_CONDITIONS = (
    "no_shaping",
    "fixed_size_padding",
    "all_class_envelope",
    "budgeted_groups",
)
_SYNTHETIC_GENERATOR_VERSION = 2


def _output_directory(config: Mapping[str, Any], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override)
    output = config.get("output", {})
    if isinstance(output, str):
        return Path(output)
    if isinstance(output, Mapping):
        for key in ("directory", "dir", "root", "path"):
            if output.get(key):
                return Path(str(output[key]))
    return Path("outputs/synthetic_demo")


def _allow_overwrite(config: Mapping[str, Any]) -> bool:
    output = config.get("output", {})
    if isinstance(output, str):
        return False
    if not isinstance(output, Mapping):
        raise ValueError("output must be a path string or mapping")
    value = output.get("overwrite", False)
    if not isinstance(value, bool):
        raise ValueError("output.overwrite must be boolean")
    return value


def _generate_features(config: Mapping[str, Any]) -> "Any":
    import numpy as np
    import pandas as pd

    seed = int(config.get("seed", 0))
    config_sha256 = stable_digest(config)
    synthetic_value = config.get("synthetic", {})
    synthetic = synthetic_value if isinstance(synthetic_value, Mapping) else {}
    if synthetic_value not in ({}, True, False) and not isinstance(
        synthetic_value, Mapping
    ):
        raise ValueError("synthetic must be a boolean or mapping")
    data = config.get("data", {})
    data = data if isinstance(data, Mapping) else {}
    configured_classes = tuple(
        str(value) for value in data.get("classes", INTENT_CLASSES)
    )
    if configured_classes != tuple(INTENT_CLASSES):
        raise ValueError(
            "synthetic data.classes must list the complete paper catalogue I01-I15 "
            "in canonical order"
        )
    configured_scopes = tuple(
        int(value) for value in data.get("scope_counts", SCOPE_COUNTS)
    )
    if configured_scopes != tuple(SCOPE_COUNTS):
        raise ValueError("synthetic data.scope_counts must be exactly [1, 2, 4]")
    raw_families = data.get("phrase_families", ("train", "validation", "test"))
    if not isinstance(raw_families, (list, tuple)) or len(raw_families) != 3:
        raise ValueError("synthetic data.phrase_families must contain exactly three names")
    phrase_families = tuple(str(value).strip() for value in raw_families)
    if any(not value for value in phrase_families) or len(set(phrase_families)) != 3:
        raise ValueError("synthetic phrase-family names must be nonempty and unique")
    topology_ids = data.get("topology_ids", ("TA",))
    model_ids = data.get("model_ids", ("synthetic-model-A",))
    if not isinstance(topology_ids, (list, tuple)) or len(topology_ids) != 1:
        raise ValueError("the smoke demo supports exactly one synthetic topology_id")
    if not isinstance(model_ids, (list, tuple)) or len(model_ids) != 1:
        raise ValueError("the smoke demo supports exactly one synthetic model_id")
    topology_id = str(topology_ids[0]).strip()
    model_id = str(model_ids[0]).strip()
    if not topology_id or not model_id:
        raise ValueError("synthetic topology_id and model_id must not be empty")
    repeats = int(
        synthetic.get(
            "runs_per_cell",
            synthetic.get("samples_per_cell", data.get("samples_per_cell", 2)),
        )
    )
    if repeats <= 0 or repeats > 100:
        raise ValueError("synthetic.runs_per_cell must be between 1 and 100")
    noise = float(synthetic.get("noise_scale", 0.10))
    if noise < 0:
        raise ValueError("synthetic.noise_scale cannot be negative")

    experiment = config.get("experiment", {})
    experiment = experiment if isinstance(experiment, Mapping) else {}
    raw_views = experiment.get("views", tuple(_VIEW_COLUMNS))
    if not isinstance(raw_views, (list, tuple)) or not raw_views:
        raise ValueError("experiment.views must be a nonempty list")
    views = tuple(str(value) for value in raw_views)
    if len(set(views)) != len(views) or set(views) - set(_VIEW_COLUMNS):
        raise ValueError(
            f"experiment.views must be unique members of {sorted(_VIEW_COLUMNS)}"
        )
    if "fused" not in views:
        raise ValueError("experiment.views must include fused for the smoke confusion plot")
    raw_conditions = experiment.get("conditions", ["no_shaping"])
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise ValueError("experiment.conditions must be a nonempty list")
    conditions = tuple(
        "fixed_size_padding" if str(value) == "fixed_size" else str(value)
        for value in raw_conditions
    )
    unknown_conditions = set(conditions) - set(_SUPPORTED_CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"unknown synthetic conditions: {sorted(unknown_conditions)}")
    fixed_padding_bytes = int(experiment.get("fixed_padding_bytes", 1024))
    shaping = config.get("shaping", {})
    shaping = shaping if isinstance(shaping, Mapping) else {}
    protected_channels = tuple(
        str(value) for value in shaping.get("protected_channels", ("CT", "CA"))
    )
    if set(protected_channels) != {"CT", "CA"} or len(protected_channels) != 2:
        raise ValueError("the implemented tool-side shaper protects exactly CT and CA")
    if str(shaping.get("overflow_policy", "fail_closed")) != "fail_closed":
        raise ValueError("the smoke scheduler supports overflow_policy=fail_closed only")
    bucket_bytes = int(shaping.get("bucket_bytes", 1024))
    round_count = int(shaping.get("round_count", 4))
    period_s = float(shaping.get("period_s", 0.1))
    budgeted_groups = shaping.get("budgeted_groups", {})
    budgeted_groups = budgeted_groups if isinstance(budgeted_groups, Mapping) else {}
    class_to_group: dict[str, int] = {}
    for group_index, members in enumerate(budgeted_groups.values(), start=1):
        for member in members:
            class_to_group[str(member)] = group_index

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(("train", "validation", "test")):
        for class_index, class_id in enumerate(configured_classes, start=1):
            for scope_q in configured_scopes:
                family = (
                    f"{class_id}-q{scope_q}-{phrase_families[partition_index]}"
                )
                for repetition in range(1, repeats + 1):
                    # These transparent signals are only for integration testing.
                    # They are not calibrated to, or asserted about, real traffic.
                    jitter = rng.normal(0.0, noise, size=9)
                    class_wave = 1.0 + 0.18 * math.sin(class_index)
                    workload_cell_id = (
                        f"synthetic-{partition}-{class_id}-q{scope_q}-r{repetition}"
                    )
                    base_row = {
                        "workload_cell_id": workload_cell_id,
                        "class_id": class_id,
                        "scope_q": scope_q,
                        "scope_bucket": {1: "low", 2: "medium", 4: "high"}[scope_q],
                        "phrase_family_id": family,
                        "protocol_block": "synthetic_core",
                        "topology_id": topology_id,
                        "topology_seed": repetition,
                        "model_id": model_id,
                        "partition": partition,
                        "synthetic": True,
                        "run_kind": "synthetic_smoke_test",
                        "paper_results_claimed": False,
                        "seed": seed,
                        "config_sha256": config_sha256,
                        "generator_version": _SYNTHETIC_GENERATOR_VERSION,
                        "notice": SYNTHETIC_NOTICE,
                        "cl_bytes": 900 + 48 * class_index + 22 * scope_q + 90 * jitter[0],
                        "cl_records": 7 + class_index % 5 + scope_q + jitter[1],
                        "cl_duration": 0.7 + 0.025 * class_index + 0.04 * scope_q + 0.08 * jitter[2],
                        "ct_bytes": 620 + 75 * class_index * class_wave + 45 * scope_q + 110 * jitter[3],
                        "ct_records": 4 + class_index % 7 + 1.5 * scope_q + jitter[4],
                        "ct_duration": 0.35 + 0.04 * class_index + 0.08 * scope_q + 0.08 * jitter[5],
                        "ca_bytes": 430 + 31 * class_index + 62 * scope_q + 85 * jitter[6],
                        "ca_records": 3 + class_index % 4 + scope_q + jitter[7],
                        "ca_duration": 0.25 + 0.02 * class_index + 0.1 * scope_q + 0.08 * jitter[8],
                    }
                    for condition in conditions:
                        row = dict(base_row)
                        row["sample_id"] = f"{workload_cell_id}-{condition}"
                        row["condition"] = condition
                        if condition == "fixed_size_padding":
                            row["ct_bytes"] = fixed_padding_bytes * max(
                                1, int(round(float(row["ct_records"])))
                            )
                            row["ca_bytes"] = fixed_padding_bytes * max(
                                1, int(round(float(row["ca_records"])))
                            )
                        elif condition == "all_class_envelope":
                            # Synthetic ideal-envelope features are intentionally
                            # identical across class and scope on CT/CA. CL stays
                            # unchanged because the paper's tool shims do not shape it.
                            row.update(
                                {
                                    "ct_bytes": 2 * round_count * bucket_bytes,
                                    "ct_records": 2 * round_count,
                                    "ct_duration": round_count * period_s,
                                    "ca_bytes": 2 * round_count * bucket_bytes,
                                    "ca_records": 2 * round_count,
                                    "ca_duration": round_count * period_s,
                                }
                            )
                        elif condition == "budgeted_groups":
                            group_index = class_to_group.get(class_id)
                            if group_index is None:
                                raise ValueError(
                                    f"budgeted_groups does not cover synthetic class {class_id}"
                                )
                            # Group identity remains visible while members and scope
                            # share a common protected synthetic representation.
                            row.update(
                                {
                                    "ct_bytes": 2 * round_count * bucket_bytes
                                    + 100 * group_index,
                                    "ct_records": 2 * round_count + group_index,
                                    "ct_duration": round_count * period_s
                                    + 0.01 * group_index,
                                    "ca_bytes": 2 * round_count * bucket_bytes
                                    + 80 * group_index,
                                    "ca_records": 2 * round_count + group_index,
                                    "ca_duration": round_count * period_s
                                    + 0.01 * group_index,
                                }
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def _observed_mi(y_true: list[str], y_pred: list[str]) -> float:
    import numpy as np

    labels_true = {label: index for index, label in enumerate(sorted(set(y_true)))}
    labels_pred = {label: index for index, label in enumerate(sorted(set(y_pred)))}
    counts = np.zeros((len(labels_true), len(labels_pred)), dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("true and predicted labels must have equal lengths")
    for true, predicted in zip(y_true, y_pred):
        counts[labels_true[true], labels_pred[predicted]] += 1.0
    probabilities = counts / counts.sum()
    row_mass = probabilities.sum(axis=1, keepdims=True)
    column_mass = probabilities.sum(axis=0, keepdims=True)
    expected = row_mass @ column_mass
    mask = probabilities > 0
    return float(np.sum(probabilities[mask] * np.log2(probabilities[mask] / expected[mask])))


def _run_actual_shaping_smoke(
    config: Mapping[str, Any], destination: Path
) -> tuple[Path, Path]:
    """Exercise the real scheduler and analytic bounds on synthetic messages."""

    from ..data.synthetic import generate_synthetic_run
    from ..metrics.bounds import (
        conservative_unconditional_mi_upper_bound,
        fitted_accuracy_upper_bound,
        fitted_class_prior,
        overflow_indicator_information,
        unconditional_accuracy_upper_bound,
    )
    from ..schemas import Channel, IntentClass, ScopeBucket, Split
    from ..shaping.envelope import make_uniform_envelope
    from ..shaping import IntentAwareShaper, all_class_partition

    shaping = config.get("shaping", {})
    shaping = shaping if isinstance(shaping, Mapping) else {}
    profile = make_uniform_envelope(
        profile_id="synthetic-all-class",
        group_id="all",
        round_count=int(shaping.get("round_count", 4)),
        period_s=float(shaping.get("period_s", 1.0)),
        bucket_bytes=int(shaping.get("bucket_bytes", 1024)),
    )
    seed = int(config.get("seed", 0))
    config_sha256 = stable_digest(config)
    shaper = IntentAwareShaper(
        all_class_partition(group_id="all"), {"all": profile}
    )
    rows: list[dict[str, Any]] = []
    class_attempts = {class_id.value: 0 for class_id in IntentClass}
    class_overflows = {class_id.value: 0 for class_id in IntentClass}
    for class_index, class_id in enumerate(IntentClass, start=1):
        for scope in ScopeBucket:
            synthetic_run = generate_synthetic_run(
                class_id,
                scope,
                Split.TEST,
                seed=seed + 100 * class_index + scope.q,
            )
            protected_messages = tuple(
                message
                for message in synthetic_run.messages
                if message.channel in {Channel.TOOL, Channel.ASSURANCE}
            )
            result = shaper.shape(
                synthetic_run.specification.intent, protected_messages
            )
            class_attempts[class_id.value] += 1
            class_overflows[class_id.value] += int(not result.fitted)
            rows.append(
                {
                    "class_id": class_id.value,
                    "scope": scope.value,
                    "fitted": result.fitted,
                    "overflow_count": len(result.overflow_events),
                    "overflow_reasons": "|".join(
                        event.reason.value for event in result.overflow_events
                    ),
                    "slot_count": len(result.transmissions),
                    "decoy_count": sum(item.is_decoy for item in result.transmissions),
                    "real_transmission_count": sum(
                        not item.is_decoy for item in result.transmissions
                    ),
                    "wire_bytes": sum(
                        item.wire_size_bytes for item in result.transmissions
                    ),
                    "termination_s": result.termination_at_s,
                    "synthetic": True,
                    "run_kind": "synthetic_smoke_test",
                    "paper_results_claimed": False,
                    "seed": seed,
                    "config_sha256": config_sha256,
                    "notice": SYNTHETIC_NOTICE,
                }
            )
    shaping_path = destination / "synthetic_shaping_smoke.csv"
    with shaping_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    prior = {class_id: 1.0 / len(INTENT_CLASSES) for class_id in INTENT_CLASSES}
    overflow_rates = {
        class_id: class_overflows[class_id] / class_attempts[class_id]
        for class_id in INTENT_CLASSES
    }
    all_groups = {"all": list(INTENT_CLASSES)}
    configured_budgeted = shaping.get("budgeted_groups", {})
    budgeted_groups = (
        {str(group): [str(member) for member in members] for group, members in configured_budgeted.items()}
        if isinstance(configured_budgeted, Mapping) and configured_budgeted
        else all_groups
    )
    bounds: dict[str, Any] = {
        "synthetic": True,
        "run_kind": "synthetic_smoke_test",
        "paper_results_claimed": False,
        "seed": seed,
        "config_sha256": config_sha256,
        "notice": SYNTHETIC_NOTICE,
        "class_prior": prior,
        "class_overflow_rates": overflow_rates,
        "overflow_indicator_information_bits_eq24": overflow_indicator_information(
            prior, overflow_rates
        ),
        "profiles": {},
    }
    for name, groups in (("all_class", all_groups), ("budgeted_groups", budgeted_groups)):
        profile_bounds: dict[str, Any] = {
            "unconditional_accuracy_upper_eq19": unconditional_accuracy_upper_bound(
                prior, overflow_rates, groups
            ),
            "conservative_unconditional_mi_upper_bits_eq21": (
                conservative_unconditional_mi_upper_bound(prior, overflow_rates, groups)
            ),
        }
        if any(rate < 1.0 for rate in overflow_rates.values()):
            fitted_prior = fitted_class_prior(prior, overflow_rates)
            profile_bounds["fitted_accuracy_upper_eq7"] = fitted_accuracy_upper_bound(
                fitted_prior, groups
            )
        else:
            profile_bounds["fitted_accuracy_upper_eq7"] = None
        bounds["profiles"][name] = profile_bounds
    bounds_path = destination / "synthetic_bounds.json"
    atomic_write_json(bounds_path, bounds)
    return shaping_path, bounds_path


def run_synthetic_demo(
    config: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run the synthetic integration demo and return created artifact paths."""

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    from ..attackers.scope import TwoStagePredictedClassScopeModel

    # Validate before creating a directory or serializing the resolved config.
    # In particular, literal credential-like fields must never reach an output
    # artifact even in synthetic mode.
    validate_manifest(config, real_run=False)
    seed = int(config.get("seed", 0))
    config_sha256 = stable_digest(config)
    seed_everything(seed)
    destination = _output_directory(config, output_dir)
    expected_outputs = (
        "synthetic_features.csv",
        "metrics.csv",
        "synthetic_metrics.csv",
        "synthetic_predictions.csv",
        "synthetic_shaping_smoke.csv",
        "synthetic_bounds.json",
        "resolved_config.json",
        "run_metadata.json",
        "synthetic_confusion_fused.png",
    )
    existing = [
        str(destination / name)
        for name in expected_outputs
        if (destination / name).exists()
    ]
    if existing and not _allow_overwrite(config):
        raise FileExistsError(
            "Refusing to overwrite existing synthetic output(s); set "
            f"output.overwrite=true for an intentional rerun: {existing}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    frame = _generate_features(config)
    feature_path = destination / "synthetic_features.csv"
    frame.to_csv(feature_path, index=False)

    synthetic_value = config.get("synthetic", {})
    synthetic_config = synthetic_value if isinstance(synthetic_value, Mapping) else {}
    models = config.get("models", {})
    models = models if isinstance(models, Mapping) else {}
    random_forest = models.get("random_forest", {})
    random_forest = random_forest if isinstance(random_forest, Mapping) else {}
    n_estimators = int(
        synthetic_config.get(
            "random_forest_trees", random_forest.get("n_estimators", 40)
        )
    )
    if n_estimators <= 0:
        raise ValueError("synthetic.random_forest_trees must be positive")
    max_depth_value = random_forest.get("max_depth")
    max_depth = None if max_depth_value is None else int(max_depth_value)
    if max_depth is not None and max_depth <= 0:
        raise ValueError("models.random_forest.max_depth must be positive or null")
    random_forest_seed = int(random_forest.get("random_state", seed))
    min_samples_leaf = int(random_forest.get("min_samples_leaf", 1))
    if min_samples_leaf <= 0:
        raise ValueError("models.random_forest.min_samples_leaf must be positive")
    class_weight = random_forest.get("class_weight")
    scope_class_weight = random_forest.get("scope_class_weight")
    experiment = config.get("experiment", {})
    experiment = experiment if isinstance(experiment, Mapping) else {}
    configured_views = tuple(
        str(value) for value in experiment.get("views", tuple(_VIEW_COLUMNS))
    )
    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for condition_index, condition in enumerate(frame["condition"].drop_duplicates()):
        condition_frame = frame[frame["condition"] == condition]
        train = condition_frame[condition_frame["partition"] == "train"]
        test = condition_frame[condition_frame["partition"] == "test"]
        training_counts = train["class_id"].astype(str).value_counts(normalize=True)
        modal_probability = float(training_counts.max())
        prior_class = sorted(
            str(label)
            for label, probability in training_counts.items()
            if probability == modal_probability
        )[0]
        prior_accuracy = float((test["class_id"].astype(str) == prior_class).mean())
        test_strata: dict[str, str] = {}
        for stratum in ("partition", "protocol_block", "topology_id", "model_id"):
            values = test[stratum].astype(str).drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(f"synthetic test rows require one {stratum} value")
            test_strata[stratum] = values[0]
        for view_index, view in enumerate(configured_views):
            columns = _VIEW_COLUMNS[view]
            class_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=random_forest_seed + 1000 * condition_index + view_index,
                n_jobs=1,
            )
            scope_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=scope_class_weight,
                random_state=random_forest_seed + 1000 * condition_index + 100 + view_index,
                n_jobs=1,
            )
            joint_model = TwoStagePredictedClassScopeModel(
                class_model,
                scope_model,
                task="classification",
                random_state=random_forest_seed + 1000 * condition_index + view_index,
            ).fit(train[list(columns)], train["class_id"], train["scope_q"])
            predicted_class, predicted_scope = joint_model.predict_with_classes(
                test[list(columns)]
            )
            predicted_scope = predicted_scope.astype(int)
            true_class = test["class_id"].astype(str).tolist()
            predicted_class_list = [str(value) for value in predicted_class]
            result_rows.append(
                {
                    **test_strata,
                    "view": view,
                    "accuracy": float(
                        accuracy_score(true_class, predicted_class_list)
                    ),
                    "macro_f1": float(
                        f1_score(
                            true_class,
                            predicted_class_list,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "prior_accuracy": prior_accuracy,
                    "advantage": float(
                        accuracy_score(true_class, predicted_class_list)
                        - prior_accuracy
                    ),
                    "scope_macro_f1": float(
                        f1_score(
                            test["scope_q"],
                            predicted_scope,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "scope_mae": float(
                        mean_absolute_error(test["scope_q"], predicted_scope)
                    ),
                    "observed_mi_bits": _observed_mi(
                        true_class, predicted_class_list
                    ),
                    "condition": condition,
                    "attacker": "random_forest",
                    "prior_predicted_class": prior_class,
                    "training_modal_probability": modal_probability,
                    "synthetic": True,
                    "run_kind": "synthetic_smoke_test",
                    "paper_results_claimed": False,
                    "seed": seed,
                    "config_sha256": config_sha256,
                    "notice": SYNTHETIC_NOTICE,
                }
            )
            for (
                sample_id,
                workload_cell_id,
                phrase_family_id,
                partition,
                protocol_block,
                topology_id,
                model_id,
                actual,
                predicted,
                actual_q,
                predicted_q,
            ) in zip(
                test["sample_id"],
                test["workload_cell_id"],
                test["phrase_family_id"],
                test["partition"],
                test["protocol_block"],
                test["topology_id"],
                test["model_id"],
                true_class,
                predicted_class_list,
                test["scope_q"],
                predicted_scope,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "sample_id": sample_id,
                        "workload_cell_id": workload_cell_id,
                        "phrase_family_id": phrase_family_id,
                        "partition": partition,
                        "protocol_block": protocol_block,
                        "topology_id": topology_id,
                        "model_id": model_id,
                        "view": view,
                        "class_id": actual,
                        "predicted_class": predicted,
                        "scope_q": int(actual_q),
                        "predicted_scope_q": int(predicted_q),
                        "condition": condition,
                        "attacker": "random_forest",
                        "prior_predicted_class": prior_class,
                        "training_modal_probability": modal_probability,
                        "synthetic": True,
                        "run_kind": "synthetic_smoke_test",
                        "paper_results_claimed": False,
                        "seed": seed,
                        "config_sha256": config_sha256,
                        "notice": SYNTHETIC_NOTICE,
                    }
                )

    results_path = destination / "metrics.csv"
    result_frame = pd.DataFrame(result_rows)
    result_frame.to_csv(results_path, index=False)
    # Compatibility alias whose name itself makes the provenance conspicuous.
    synthetic_results_path = destination / "synthetic_metrics.csv"
    result_frame.to_csv(synthetic_results_path, index=False)
    predictions_path = destination / "synthetic_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    shaping_path, bounds_path = _run_actual_shaping_smoke(config, destination)
    resolved_config_path = destination / "resolved_config.json"
    atomic_write_json(resolved_config_path, dict(config))
    plot_path = destination / "synthetic_confusion_fused.png"
    plot_confusion_from_csv(
        predictions_path,
        plot_path,
        view="fused",
        condition="no_shaping",
        synthetic=True,
    )
    metadata_path = destination / "run_metadata.json"
    artifact_paths = {
        "features": feature_path,
        "metrics": results_path,
        "synthetic_metrics_alias": synthetic_results_path,
        "predictions": predictions_path,
        "shaping_smoke": shaping_path,
        "bounds": bounds_path,
        "resolved_config": resolved_config_path,
        "confusion_plot": plot_path,
    }
    provenance = collect_provenance(config, synthetic=True, seed=seed)
    provenance.update(
        {
            "generator_version": _SYNTHETIC_GENERATOR_VERSION,
            "smoke_component_scope": {
                "executed": [
                    "generated aggregate feature records",
                    "prior-only baseline",
                    "random-forest class and two-stage scope attack",
                    "intent-aware scheduler smoke",
                    "analytic leakage bounds",
                    "confusion plotting",
                ],
                "not_executed_by_this_command": [
                    "models.sequence_knn",
                    "models.cnn",
                    "evaluation.bootstrap_resamples",
                    "evaluation.confidence",
                    "evaluation.bootstrap_seed",
                ],
            },
            "artifacts": {
                name: {"path": path.name, "sha256": file_sha256(path)}
                for name, path in artifact_paths.items()
            },
            "paper_results_claimed": False,
        }
    )
    atomic_write_json(metadata_path, provenance)
    return {
        "features": feature_path,
        "metrics": results_path,
        "synthetic_metrics": synthetic_results_path,
        "predictions": predictions_path,
        "shaping_smoke": shaping_path,
        "bounds": bounds_path,
        "resolved_config": resolved_config_path,
        "metadata": metadata_path,
        "confusion_plot": plot_path,
    }
