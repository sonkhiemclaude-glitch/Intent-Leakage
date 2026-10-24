#!/usr/bin/env python3
"""Offline readiness evidence; never loads credentials or starts real captures.

Run with the locked project interpreter after ``uv sync --frozen --extra dev``.
Every invocation gets a fresh output directory. A successful smoke test does not
mean the real testbed is ready. See readiness.json for the separate decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent_leakage.orchestration.grid import build_minimum_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    destination = (args.output or ROOT / "outputs" / (
        "readiness_" + now.strftime("%Y%m%dT%H%M%S_%fZ")
    )).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label: str, arguments: list[str], expected: int = 0) -> bool:
        command = [sys.executable, *arguments]
        print(f"Running {label}", flush=True)
        with (destination / f"{label}.log").open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        commands.append({"label": label, "argv": command,
                         "returncode": result.returncode,
                         "expected_returncode": expected,
                         "log": f"{label}.log"})
        return result.returncode == expected

    checks = []
    if args.run_tests:
        checks.append(run("pytest", ["-m", "pytest", "-q"]))
    smoke = destination / "synthetic"
    checks.append(run("synthetic", ["-m", "intent_leakage.cli", "synthetic",
        "--config", "configs/synthetic_demo.yaml", "--output", str(smoke)]))
    if checks[-1]:
        for stage in ("calibrate", "evaluate", "ablation"):
            checks.append(run(stage, ["-m", "intent_leakage.cli", "experiment",
                "--config", "configs/synthetic_demo.yaml", "--stage", stage,
                "--output", str(smoke)]))
            if not checks[-1]:
                break
    # This deliberately tests the unresolved template's fail-closed behavior.
    checks.append(run("template_preflight", ["scripts/mac_studio_preflight.py",
        "--config", "configs/experiment_template.yaml"], expected=1))
    grids = {}
    for label, topology in (("two_topologies", "TB"), ("one_topology", None)):
        rows = build_minimum_grid(1, topology_b=topology)
        grids[label] = {"design_only": True, "R": 1, "count": len(rows),
                        "blocks": dict(Counter(row.block for row in rows))}
    checks.append(grids["two_topologies"]["count"] == 630)
    checks.append(grids["one_topology"]["count"] == 585)
    env = ROOT / ".env"
    source_hash = hashlib.sha256()
    source_files = sorted(set(ROOT.glob("src/**/*.py")) | set(ROOT.glob("scripts/*.py"))
                          | set(ROOT.glob("tests/*.py")) | set(ROOT.glob("configs/*.yaml"))
                          | {ROOT / "pyproject.toml", ROOT / "uv.lock"})
    for path in source_files:
        source_hash.update(str(path.relative_to(ROOT)).encode() + b"\0")
        source_hash.update(path.read_bytes())
    report = {
        "recorded_at_utc": now.isoformat(),
        "host_is_verified_mac_studio": False,
        "host": {"system": platform.system(), "machine": platform.machine(),
                 "macos": platform.mac_ver()[0], "python": platform.python_version()},
        "env_file": {"exists": env.exists(),
                     "empty": env.stat().st_size == 0 if env.exists() else None,
                     "values_read": False},
        "commands_available": {name: shutil.which(name) is not None
                               for name in ("ssh", "uv", "tcpdump", "tshark")},
        "modules_available": {name: importlib.util.find_spec(name) is not None
                              for name in ("boto3", "torch", "numpy", "pandas", "sklearn")},
        "code_config_lock_sha256": source_hash.hexdigest(),
        "hashed_file_count": len(source_files),
        "grid_designs": grids, "commands": commands,
        "offline_checks_passed": all(checks), "tests_executed": args.run_tests,
        "real_capture_ready": False, "remote_connection_attempted": False,
        "paper_results_claimed": False,
        "blockers": ["Concrete external adapters and live shaping shims are absent",
                     "Real manifest still contains unresolved author decisions",
                     "Real captures and the trace-to-feature join are absent",
                     "Split/transfer guards and full statistical reporting need integration"],
    }
    (destination / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Readiness report: {destination / 'readiness.json'}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
