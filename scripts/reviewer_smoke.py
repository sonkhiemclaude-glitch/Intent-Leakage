"""Bounded, local reviewer diagnostics. No collection, cloud calls or full refits.

Twelve depth-two count-only trees on ONE existing session split; two archived
engineering PCAP replays; one constructed TLS-header/TCP-segmentation fixture.
The constructed fixture is a parser test, not encrypted experimental traffic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from extend_paper_evaluation import split_sessions
from intent_leakage.adapters.tshark import TsharkTlsParser
from intent_leakage.features.extraction import extract_channel_features


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def checksum(data):
    data += b"\0" * (len(data) % 2)
    s = sum(struct.unpack("!" + "H" * (len(data) // 2), data))
    while s >> 16:
        s = (s & 65535) + (s >> 16)
    return (~s) & 65535


def make_fixture(folder):
    """One split record, two coalesced records, retransmit, unrelated flow."""
    ip = b"\x7f\0\0\x01"
    packets = []

    def packet(usec, sport, dport, seq, ack, flags, payload=b""):
        tcp = struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 5 << 4, flags, 65535, 0, 0)
        tcp = tcp[:16] + struct.pack("!H", checksum(ip + ip + struct.pack("!BBH", 0, 6, len(tcp)+len(payload)) + tcp + payload)) + tcp[18:]
        head = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40+len(payload), len(packets)+1, 0, 64, 6, 0, ip, ip)
        head = head[:10] + struct.pack("!H", checksum(head)) + head[12:]
        frame = bytes.fromhex("0200000000020200000000010800") + head + tcp + payload
        packets.append(struct.pack("<IIII", 1, usec, len(frame), len(frame)) + frame)

    record = lambda n: b"\x17\x03\x03" + struct.pack("!H", n) + b"x" * n
    a, b = record(20), record(30)+record(40)
    packet(0, 41000, 8443, 100, 0, 2)
    packet(1000, 8443, 41000, 200, 101, 18)
    packet(2000, 41000, 8443, 101, 201, 16)
    packet(3000, 41000, 8443, 101, 201, 24, a[:9])
    packet(4000, 41000, 8443, 110, 201, 24, a[9:])
    packet(5000, 8443, 41000, 201, 126, 24, b)
    packet(6000, 8443, 41000, 201, 126, 24, b)  # duplicate TCP payload
    packet(7000, 42000, 9999, 50, 0, 24, record(60))
    path = folder / "constructed_tls12.pcap"
    path.write_bytes(struct.pack("<IHHIIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1) + b"".join(packets))
    manifest = {"trace_id": "constructed-parser-fixture", "endpoints": [
        {"client_ip": "127.0.0.1", "server_ip": "127.0.0.1", "server_port": 8443, "channel": "CT"}]}
    save(folder / "flow_manifest.json", manifest)
    return path, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--readonly", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tshark", default="tshark")
    a = p.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"kind": "reviewer_smoke_only", "independent_validation": False,
        "new_collection_runs": 0, "cloud_requests": 0, "planned_tree_fits": 12,
        "tree": {"max_depth": 2, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 0},
        "split_seed": 0, "feature": "TLS record count only; not TCP connection count",
        "archive_sha256": sha(a.archive), "script_sha256": sha(__file__),
        "parser_sha256": sha(ROOT / "src/intent_leakage/adapters/tshark.py"),
        "feature_extractor_sha256": sha(ROOT / "src/intent_leakage/features/extraction.py"),
        "pcap_selection": "read_00 from each of the two archived final engineering sessions",
        "timeline_selection": "lexicographically first matched session, q=2, first run_id per class/mode",
        "tshark_version": subprocess.check_output([a.tshark, "--version"], text=True).splitlines()[0]}
    save(out / "protocol.json", protocol)  # before fitting / replay
    with tarfile.open(a.archive) as t:
        bundle = json.load(t.extractfile("outputs/seq_bundle/input.json"))
        archived_rf = json.load(t.extractfile("outputs/rf/report.json"))
    rows = bundle["rows"]
    common = set.intersection(*[{r["session"] for r in rows if r["cohort"] == c} for c in ("mcp", "native")])
    matched = [r for r in rows if r["session"] in common]
    assert len(matched) == 504 and len(common) == 14
    accounting = []
    for c in ("mcp", "native"):
        for k in ("I01", "I02", "I15"):
            for q in (1, 2, 4):
                rr = [r for r in rows if (r["cohort"], r["class_id"], r["scope_q"]) == (c, k, q)]
                mm = [r for r in rr if r["session"] in common]
                accounting.append({"cohort": c, "class_id": k, "scope_q": q,
                    "archived_successful": len(rr), "matched_analyzed": len(mm),
                    "unmatched_successful": len(rr)-len(mm),
                    "all_attempts": None, "execution_failures": None, "verification_failures": None})
    save(out / "cohort_accounting.json", {"cells": accounting,
        "warning": "Successful bundle accounting only; null is unknown, never zero. No complete historical attempt denominator reconstructed."})
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import balanced_accuracy_score, f1_score
    results, predictions = [], []
    for c in ("mcp", "native"):
        rr = [r for r in matched if r["cohort"] == c]
        train_ids, test_ids = split_sessions(rr, 0)
        train = [r for r in rr if r["session"] in train_ids]
        test = [r for r in rr if r["session"] in test_ids]
        assert len(train) == 180 and len(test) == 72 and not train_ids & test_ids
        for target in ("class", "scope"):
            key = "class_id" if target == "class" else "scope_q"
            for view in ("ct", "ca", "ctca"):
                col = view + "__event_count"
                model = DecisionTreeClassifier(**protocol["tree"])
                model.fit([[r[col]] for r in train], [str(r[key]) for r in train])
                pred = model.predict([[r[col]] for r in test])
                truth = [str(r[key]) for r in test]
                ref = [v for v in archived_rf["results"] if (v["source"], v["destination"], v["scenario"], v["view"], v["target"], v["seed"], v["permuted_training_labels"]) == (c,c,"session_holdout",view,target,0,False)]
                assert len(ref) == 1 and set(ref[0]["test_sessions"]) == test_ids
                results.append({"cohort": c, "target": target, "view": view,
                    "balanced_accuracy": balanced_accuracy_score(truth, pred),
                    "macro_f1": f1_score(truth, pred, average="macro", zero_division=0),
                    "archived_full_rf_ba_same_split": ref[0]["balanced_accuracy"],
                    "train_sessions": sorted(train_ids), "test_sessions": sorted(test_ids),
                    "train_rows": len(train), "test_rows": len(test),
                    "tree": export_text(model, feature_names=[col])})
                predictions.extend({"cohort":c, "target":target, "view":view, "run_id":r["run_id"],
                    "truth":y, "prediction":str(v)} for r,y,v in zip(test,truth,pred))
    save(out / "predictions.json", predictions)
    parser = TsharkTlsParser(a.tshark, timeout_s=30)
    replays = []
    for batch in ("batch_001", "batch_002"):
        d = a.readonly / batch / "read_00"
        m = json.loads((d / "flow_manifest.json").read_text())
        actual = parser.parse(d / "capture.pcap", flow_manifest=m)
        expected = [json.loads(x) for x in (d / "tls_records.jsonl").read_text().splitlines()]
        keys = ("timestamp_ns", "channel", "direction", "length_bytes", "flow_id", "retransmission")
        assert [[r[k] for k in keys] for r in actual] == [[r[k] for k in keys] for r in expected]
        f = extract_channel_features(actual, channel="CA")
        assert f == extract_channel_features(expected, channel="CA")
        save(out / (batch + "_normalized.json"), actual)
        replays.append({"batch":batch, "pcap_sha256":sha(d/"capture.pcap"), "records":len(actual),
            "record_fields_exact":True, "features_exact":True, "features":f})
    fixture = out / "parser_fixture"
    fixture.mkdir()
    path, manifest = make_fixture(fixture)
    events = parser.parse(path, flow_manifest=manifest)
    expected = [(1004000000, 1, 25), (1005000000, -1, 35), (1005000000, -1, 45)]
    assert [(r["timestamp_ns"],r["direction"],r["length_bytes"]) for r in events] == expected, events
    features = extract_channel_features(events, channel="CT")
    assert features["event_count"] == 3 and features["total_bytes"] == 105
    save(fixture / "expected_records.json", events)
    save(fixture / "expected_features.json", features)
    traces = []
    session = min(common)
    for c in ("mcp", "native"):
        for k in ("I01", "I02", "I15"):
            r = min((r for r in matched if (r["cohort"],r["class_id"],r["scope_q"],r["session"]) == (c,k,2,session)), key=lambda r:r["run_id"])
            traces.append({k:r[k] for k in ("run_id","cohort","class_id","scope_q","session","sequences")})
    save(out / "representative_traces.json", traces)
    save(out / "report.json", {**protocol, "tree_results":results, "engineering_replays":replays,
        "constructed_fixture_passed":True, "constructed_fixture_tls_records":len(events),
        "limitations":["One reused session split; no CI or independent confirmation", "Record count is not connection count", "TLS 1.2 fixture tests headers, not cryptography", "Engineering PCAPs do not validate original attack captures"]})
    save(out / "checksums.json", {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps({"tree_fits":len(results), "engineering_replays":len(replays), "constructed_fixture_passed":True, "new_collection_runs":0}))


if __name__ == "__main__":
    main()
