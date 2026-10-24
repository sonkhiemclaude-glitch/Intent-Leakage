from __future__ import annotations

import json

import pytest

from intent_leakage.data import load_dataset_root, load_trace_events_jsonl
from intent_leakage.schemas import Channel


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_manually_provisioned_trace_jsonl(tmp_path) -> None:
    path = tmp_path / "traces" / "events.jsonl"
    _write_jsonl(
        path,
        [
            {
                "trace_id": "t1",
                "timestamp_ns": 1_000_000_000,
                "direction": 1,
                "length_bytes": 100,
                "channel": "C_T",
                "segmentation_mode": "endpoint_separable",
            },
            {
                "trace_id": "t1",
                "timestamp_s": 1.2,
                "direction": -1,
                "length_bytes": 120,
                "channel": "CT",
                "segmentation_mode": "endpoint_separable",
                "retransmission": True,
            },
        ],
    )
    loaded = load_dataset_root(tmp_path)
    assert loaded.segmentation_modes == {"t1": "endpoint_separable"}
    trace = loaded.traces["t1"]
    assert trace.traces[0].channel is Channel.TOOL
    assert trace.traces[0].signed_sizes == (100, -120)


def test_loader_rejects_ground_truth_in_attacker_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(
        path,
        [
            {
                "trace_id": "t1",
                "timestamp_s": 0.0,
                "direction": 1,
                "length_bytes": 100,
                "channel": "CL",
                "class_id": "I01",
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden attacker fields"):
        load_trace_events_jsonl(path)


def test_loader_rejects_mixed_segmentation_modes(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    base = {
        "trace_id": "t1",
        "direction": 1,
        "length_bytes": 100,
        "channel": "CA",
    }
    _write_jsonl(
        path,
        [
            {**base, "timestamp_s": 0.0, "segmentation_mode": "passive"},
            {**base, "timestamp_s": 0.1, "segmentation_mode": "oracle_segmented"},
        ],
    )
    with pytest.raises(ValueError, match="mixes segmentation"):
        load_trace_events_jsonl(path)


def test_combined_mode_collapses_tool_and_assurance_labels(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_jsonl(
        path,
        [
            {
                "trace_id": "t1",
                "timestamp_s": 0.0,
                "direction": 1,
                "length_bytes": 100,
                "channel": "CT",
                "segmentation_mode": "combined",
            },
            {
                "trace_id": "t1",
                "timestamp_s": 0.1,
                "direction": -1,
                "length_bytes": 120,
                "channel": "CA",
                "segmentation_mode": "combined",
            },
        ],
    )
    loaded = load_trace_events_jsonl(path)
    trace = loaded.traces["t1"]
    assert len(trace.traces) == 1
    assert trace.traces[0].channel is Channel.TOOL
    assert trace.traces[0].signed_sizes == (100, -120)


def test_loader_rejects_out_of_order_and_string_boolean(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    base = {
        "trace_id": "t1",
        "direction": 1,
        "length_bytes": 100,
        "channel": "CL",
        "segmentation_mode": "passive",
    }
    _write_jsonl(
        path,
        [
            {**base, "timestamp_s": 1.0},
            {**base, "timestamp_s": 0.5},
        ],
    )
    with pytest.raises(ValueError, match="not chronological"):
        load_trace_events_jsonl(path)

    _write_jsonl(
        path,
        [{**base, "timestamp_s": 0.0, "retransmission": "false"}],
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        load_trace_events_jsonl(path)
