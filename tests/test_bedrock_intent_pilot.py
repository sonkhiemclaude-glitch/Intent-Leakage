import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("intent_pilot", Path(__file__).resolve().parents[1] / "scripts/run_bedrock_intent_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_generated_pilot_covers_each_class_scope_once_without_class_ids_in_prompt():
    cases = pilot.build_cases()
    assert len(cases) == len({r["case_id"] for r in cases}) == 45
    assert {(r["class_id"], r["scope_q"]) for r in cases} == {
        (f"I{i:02d}", q) for i in range(1, 16) for q in (1, 2, 4)}
    assert all(len(r["targets"]) == r["scope_q"] and r["class_id"] not in r["prompt"] for r in cases)


def test_scoring_retains_wrong_class_scope_and_truncation_as_failures():
    expected = pilot.build_cases()[0]
    correct = {k: expected[k] for k in ("class_id", "scope_q", "targets")}
    assert pilot.assess(json.dumps(correct), expected)["all_correct"]
    assert not pilot.assess('{"class_id":', expected)["json_valid"]
    wrong = {**correct, "class_id": "I02"}
    score = pilot.assess(json.dumps(wrong), expected)
    assert score["schema_valid"] and not score["class_correct"] and not score["all_correct"]


def test_malformed_types_and_extra_fields_do_not_crash_or_pass_schema():
    expected = pilot.build_cases()[0]
    for output in ([], {"class_id": [], "scope_q": 1, "targets": ["host01"]},
                   {"class_id": "I01", "scope_q": True, "targets": ["host01"]},
                   {"class_id": "I01", "scope_q": 1, "targets": ["host01"], "extra": "unexpected"}):
        assert not pilot.assess(json.dumps(output), expected)["schema_valid"]
