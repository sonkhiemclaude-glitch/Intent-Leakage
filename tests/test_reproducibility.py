from __future__ import annotations

from intent_leakage.experiments.synthetic_demo import run_synthetic_demo


def test_synthetic_demo_is_reproducible_under_fixed_seed(tmp_path) -> None:
    config = {
        "seed": 991,
        "synthetic": {"runs_per_cell": 1, "random_forest_trees": 5},
        "output": {},
    }
    first = run_synthetic_demo(config, output_dir=tmp_path / "first")
    second = run_synthetic_demo(config, output_dir=tmp_path / "second")
    for artifact in ("features", "metrics", "predictions"):
        assert first[artifact].read_bytes() == second[artifact].read_bytes()
