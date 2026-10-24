#!/usr/bin/env python3
"""Generate manuscript-compatible plots from result CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intent_leakage.cli import main  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--kind",
        choices=("auto", "summary", "confusion", "generalization", "tradeoff"),
        default="auto",
    )
    parser.add_argument("--view", default="fused")
    parser.add_argument("--synthetic", action="store_true")
    arguments = parser.parse_args()
    command = [
        "plot",
        "--input",
        arguments.input,
        "--output",
        arguments.output,
        "--kind",
        arguments.kind,
        "--view",
        arguments.view,
    ]
    if arguments.synthetic:
        command.append("--synthetic")
    raise SystemExit(main(command))
