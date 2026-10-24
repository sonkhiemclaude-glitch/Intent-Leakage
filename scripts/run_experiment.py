#!/usr/bin/env python3
"""Plan or run a gated experiment stage from a source checkout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intent_leakage.cli import main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "calibrate", "evaluate", "ablation", "capture"),
    )
    parser.add_argument("--output")
    parser.add_argument("--allow-external", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage == "plan":
        command = ["plan", "--config", arguments.config]
    else:
        command = [
            "experiment",
            "--config",
            arguments.config,
            "--stage",
            arguments.stage,
        ]
    if arguments.output:
        command.extend(("--output", arguments.output))
    if arguments.allow_external and arguments.stage != "plan":
        command.append("--allow-external")
    raise SystemExit(main(command))
