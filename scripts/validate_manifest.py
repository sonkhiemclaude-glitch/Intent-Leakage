#!/usr/bin/env python3
"""Validate a manifest without contacting any external system."""

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
    parser.add_argument("--real", action="store_true")
    arguments = parser.parse_args()
    command = ["validate-manifest", "--config", arguments.config]
    if arguments.real:
        command.append("--real")
    raise SystemExit(main(command))
