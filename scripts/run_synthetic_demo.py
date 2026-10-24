#!/usr/bin/env python3
"""Run the labelled synthetic smoke test from a source checkout."""

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
    parser.add_argument("--output")
    arguments = parser.parse_args()
    command = ["synthetic", "--config", arguments.config]
    if arguments.output:
        command.extend(("--output", arguments.output))
    raise SystemExit(main(command))
