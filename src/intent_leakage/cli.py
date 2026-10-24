"""Command-line interface for planning, smoke testing, and offline analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .experiments.runner import run_experiment
from .experiments.synthetic_demo import run_synthetic_demo
from .orchestration.manifest import validate_manifest
from .plotting import (
    plot_confusion_from_csv,
    plot_generalization_from_csv,
    plot_metric_summary_from_csv,
    plot_tradeoff_from_csv,
)
from .utils.io import load_config
from .utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intent-leakage",
        description=(
            "Reproducible implementation scaffold for encrypted-traffic intent "
            "inference and intent-aware shaping."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "synthetic", help="run the lightweight synthetic smoke test"
    )
    synthetic.add_argument("--config", required=True, type=Path)
    synthetic.add_argument("--output", type=Path)

    plan = subparsers.add_parser("plan", help="materialize the minimum run grid")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--output", type=Path)

    validate = subparsers.add_parser(
        "validate-manifest", help="validate a synthetic or frozen real manifest"
    )
    validate.add_argument("--config", required=True, type=Path)
    validate.add_argument("--real", action="store_true")

    experiment = subparsers.add_parser(
        "experiment", help="run a gated experiment stage"
    )
    experiment.add_argument("--config", required=True, type=Path)
    experiment.add_argument(
        "--stage", required=True, choices=("calibrate", "evaluate", "ablation", "capture")
    )
    experiment.add_argument("--output", type=Path)
    experiment.add_argument(
        "--allow-external",
        action="store_true",
        help="enable provisioned external adapters; never implied by configuration",
    )

    plot = subparsers.add_parser("plot", help="generate a plot from a result file")
    plot.add_argument("--input", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    plot.add_argument(
        "--kind",
        choices=("auto", "summary", "confusion", "generalization", "tradeoff"),
        default="auto",
    )
    plot.add_argument("--view", default="fused")
    plot.add_argument("--synthetic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    if args.command == "synthetic":
        artifacts = run_synthetic_demo(
            load_config(args.config), output_dir=args.output
        )
        LOGGER.info("Created synthetic smoke-test artifacts in %s", next(iter(artifacts.values())).parent)
        return 0
    if args.command == "plan":
        artifacts = run_experiment(
            load_config(args.config), stage="plan", output_dir=args.output
        )
        LOGGER.info("Created plan: %s", artifacts["grid"])
        return 0
    if args.command == "validate-manifest":
        warnings = validate_manifest(load_config(args.config), real_run=args.real)
        if warnings:
            LOGGER.warning("; ".join(warnings))
        LOGGER.info("Manifest validation passed")
        return 0
    if args.command == "experiment":
        run_experiment(
            load_config(args.config),
            stage=args.stage,
            output_dir=args.output,
            allow_external=args.allow_external,
        )
        return 0
    if args.command == "plot":
        import csv

        kind = args.kind
        if kind == "auto":
            with args.input.open("r", encoding="utf-8", newline="") as stream:
                fields = set(next(csv.reader(stream)))
            if {"class_id", "predicted_class"} <= fields:
                kind = "confusion"
            elif {"profile", "observed_mi", "added_latency", "bandwidth_overhead"} <= fields:
                kind = "tradeoff"
            elif {"held_out_condition", "accuracy"} <= fields:
                kind = "generalization"
            elif {"view", "accuracy"} <= fields:
                kind = "summary"
            else:
                raise ValueError("Could not infer plot kind from CSV columns")
        output = args.output
        if output.suffix.lower() not in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
            output.mkdir(parents=True, exist_ok=True)
            output = output / f"{kind}.png"
        if kind == "confusion":
            plot_confusion_from_csv(
                args.input,
                output,
                view=args.view,
                synthetic=args.synthetic,
            )
        elif kind == "generalization":
            plot_generalization_from_csv(
                args.input, output, synthetic=args.synthetic
            )
        elif kind == "tradeoff":
            plot_tradeoff_from_csv(args.input, output, synthetic=args.synthetic)
        else:
            plot_metric_summary_from_csv(
                args.input, output, synthetic=args.synthetic
            )
        LOGGER.info("Created plot: %s", output)
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
