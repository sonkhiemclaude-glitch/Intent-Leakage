from __future__ import annotations

import importlib


def test_owned_modules_import_without_optional_external_dependencies() -> None:
    modules = (
        "intent_leakage",
        "intent_leakage.cli",
        "intent_leakage.adapters",
        "intent_leakage.splits",
        "intent_leakage.orchestration",
        "intent_leakage.experiments",
        "intent_leakage.plotting",
        "intent_leakage.utils",
    )
    for module in modules:
        importlib.import_module(module)
