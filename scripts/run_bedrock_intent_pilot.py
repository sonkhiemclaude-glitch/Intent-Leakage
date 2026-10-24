#!/usr/bin/env python3
"""Live, bounded LLM-only pilot on generated intents; no network actuation.

Scores catalogue classification, primary-target scope, and target extraction.
This is not a traffic attacker, fulfilment experiment, or confirmatory dataset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_leakage.adapters.bedrock import BedrockAdapter, BedrockConfig
from intent_leakage.catalog import list_intent_definitions

TEMPLATES = {
    "I01": ("host", "Prevent these compromised hosts from exchanging ordinary data traffic; preserve their management access: {targets}."),
    "I02": ("host", "These hosts are currently quarantined. Restore their normal baseline reachability: {targets}."),
    "I03": ("pair", "Deny communication for the following named endpoint pairs: {targets}."),
    "I04": ("pair", "Allow communication for the following named endpoint pairs: {targets}."),
    "I05": ("flow", "Cap each of these selected service flows at 10 Mbps: {targets}."),
    "I06": ("flow", "Raise the bandwidth allocation and forwarding priority of these selected flows: {targets}."),
    "I07": ("demand", "Route these demands around link L7 without changing their destinations: {targets}."),
    "I08": ("demand", "Restore the predefined default routing paths for these demands: {targets}."),
    "I09": ("slice", "Create isolated tenant slices with the following object identifiers: {targets}."),
    "I10": ("slice", "Expand the membership and resource allocation of these existing tenant slices: {targets}."),
    "I11": ("slice", "Delete these tenant slices and remove their derived forwarding policies: {targets}."),
    "I12": ("service", "Onboard these tenant services with their required attachments, policy objects, and reachability: {targets}."),
    "I13": ("service", "Withdraw these tenant services and remove their service-specific objects and rules: {targets}."),
    "I14": ("flow", "Redirect these selected flows to alternate endpoint sink01, leaving other flows unchanged: {targets}."),
    "I15": ("policy", "Run one bounded read-only compliance audit of these policies; return a verdict and do not install a persistent monitor: {targets}."),
}


def build_cases():
    cases = []
    for class_id, (stem, template) in TEMPLATES.items():
        for q in (1, 2, 4):
            targets = [f"{stem}{index:02d}" for index in range(1, q+1)]
            cases.append({"case_id": f"llm-pilot-{class_id}-q{q}", "class_id": class_id,
                          "scope_q": q, "targets": targets,
                          "prompt": template.format(targets=", ".join(targets))})
    return cases


def assess(text, expected):
    result = {"json_valid": False, "schema_valid": False, "class_correct": False,
              "scope_correct": False, "targets_exact": False, "all_correct": False}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return result
    result["json_valid"] = True
    if (not isinstance(parsed, dict) or set(parsed) != {"class_id", "scope_q", "targets"}
        or not isinstance(parsed["class_id"], str) or parsed["class_id"] not in TEMPLATES or type(parsed["scope_q"]) is not int
        or parsed["scope_q"] not in (1, 2, 4) or not isinstance(parsed["targets"], list)
        or any(not isinstance(v, str) for v in parsed["targets"])
        or len(parsed["targets"]) != len(set(parsed["targets"]))
        or len(parsed["targets"]) != parsed["scope_q"]):
        return result
    result["schema_valid"] = True
    result["class_correct"] = parsed["class_id"] == expected["class_id"]
    result["scope_correct"] = parsed["scope_q"] == expected["scope_q"]
    result["targets_exact"] = set(parsed["targets"]) == set(expected["targets"])
    result["all_correct"] = all(result[k] for k in
                                ("json_valid", "schema_valid", "class_correct", "scope_correct", "targets_exact"))
    result["prediction"] = parsed
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--model-label", choices=("A", "B"), action="append")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    args = parser.parse_args()
    if not args.allow_external:
        parser.error("Live Bedrock requests require --allow-external")
    token = sys.stdin.readline().rstrip("\r\n") if args.token_stdin else None
    if token is not None and (not token or any(c.isspace() for c in token)):
        parser.error("A nonempty bearer token is required on stdin")
    old = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if token:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    adapters = {}
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        models = {"A": "amazon.nova-micro-v1:0", "B": "openai.gpt-oss-20b-1:0"}
        if args.model_label:
            models = {k: v for k, v in models.items() if k in args.model_label}
        extra_fields = {label: ({"reasoning_effort": args.reasoning_effort}
                                if label == "B" and args.reasoning_effort else {}) for label in models}
        cases = build_cases()
        system = ("Map each operator request to the catalogue below. Return exactly one JSON object "
                  "with keys class_id, scope_q, targets. No markdown, explanation, or extra keys. "
                  "scope_q is the number of primary targets (1,2,4). targets must contain exactly "
                  "the primary object identifiers listed in the request; L7 and sink01 are not "
                  "primary targets. Do not execute actions or claim that a network was modified.\n" +
                  "\n".join(f"{v.class_id.value}: {v.name}; primary targets: {v.primary_target_type}"
                            for v in list_intent_definitions()))
        config = {label: BedrockConfig(model_id, args.region, False, {"temperature": 0.0},
                    128, {"mode": "standard", "total_max_attempts": 1}, "reuse",
                    read_timeout_s=45) for label, model_id in models.items()}
        tasks = [(label, case) for label in models for case in cases]
        random.Random(20261020).shuffle(tasks)
        manifest = {"run_kind": "live_llm_only_generated_intent_pilot", "synthetic": False,
            "workloads_generated": True, "paper_results_claimed": False,
            "network_capture_performed": False, "controller_actions": 0,
            "case_count": 45, "request_ceiling": len(tasks), "seed": 20261020,
            "configuration": {label: asdict(value) for label, value in config.items()},
            "additional_model_request_fields": extra_fields,
            "system_prompt": system, "cases": cases,
            "execution_order": [{"model": label, "case_id": case["case_id"]} for label, case in tasks],
            "parsing_policy": "strict full JSON; exact schema; no repair or format retries",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256((ROOT / "src/intent_leakage/adapters/bedrock.py").read_bytes()).hexdigest()}
        encoded = json.dumps(manifest, sort_keys=True)
        manifest["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        (args.output / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        adapters = {label: BedrockAdapter(value, allow_external=True) for label, value in config.items()}
        records = []
        consecutive_errors = 0
        calls = {label: 0 for label in models}
        with (args.output / "responses.jsonl").open("w") as log:
            for index, (label, case) in enumerate(tasks):
                record = {"order": index, "model_label": label, "model_id": models[label],
                          "case_id": case["case_id"], "expected_class": case["class_id"],
                          "expected_scope_q": case["scope_q"], "sdk_client_reused_before_call": calls[label] > 0}
                start = time.perf_counter()
                try:
                    request = {"system": [{"text": system}],
                               "messages": [{"role": "user", "content": [{"text": case["prompt"]}]}]}
                    if extra_fields[label]:
                        request["additionalModelRequestFields"] = extra_fields[label]
                    response = adapters[label].invoke(request)
                    text = "".join(v.get("text", "") for v in response.get("output", {}).get("message", {}).get("content", []))
                    for secret in (token, old):
                        if secret:
                            text = text.replace(secret, "<redacted>")
                    record.update({"api_success": True, "text": text, "usage": response.get("usage", {}),
                        "stop_reason": response.get("stopReason"), "assessment": assess(text, case),
                        "server_metrics": response.get("metrics", {}),
                        "http_status": response.get("ResponseMetadata", {}).get("HTTPStatusCode")})
                    consecutive_errors = 0
                except Exception as exc:
                    detail = getattr(exc, "response", {})
                    record.update({"api_success": False, "error_type": type(exc).__name__,
                                   "aws_error_code": detail.get("Error", {}).get("Code")})
                    consecutive_errors += 1
                calls[label] += 1
                record["client_elapsed_s"] = time.perf_counter()-start
                record.update({"synthetic": False, "workloads_generated": True,
                               "paper_results_claimed": False})
                log.write(json.dumps(record)+"\n");log.flush();records.append(record)
                print(json.dumps({"completed": len(records), "total": len(tasks), "model": label,
                    "case_id": case["case_id"], "api_success": record["api_success"],
                    "all_correct": record.get("assessment", {}).get("all_correct", False)}), flush=True)
                if consecutive_errors >= 3:
                    break
        summaries = []
        for label in models:
            group = [r for r in records if r["model_label"] == label]
            row = {"model_label": label, "model_id": models[label], "attempted": len(group),
                   "expected_cases": 45, "api_successes": sum(r["api_success"] for r in group)}
            for metric in ("json_valid", "schema_valid", "class_correct", "scope_correct", "targets_exact", "all_correct"):
                row[metric+"_count"] = sum(r.get("assessment", {}).get(metric, False) for r in group)
            row["input_tokens"] = sum(r.get("usage", {}).get("inputTokens", 0) for r in group)
            row["output_tokens"] = sum(r.get("usage", {}).get("outputTokens", 0) for r in group)
            summaries.append(row)
        report = {"run_kind": manifest["run_kind"], "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic": False, "workloads_generated": True, "paper_results_claimed": False,
            "credentials_persisted": False, "network_capture_performed": False,
            "controller_actions": 0, "expected_requests": len(tasks), "attempted_requests": len(records),
            "completed": len(records) == len(tasks), "models": summaries,
            "boto3": importlib.metadata.version("boto3"), "manifest_sha256": manifest["manifest_sha256"],
            "limitations": ["One generated wording per class/scope; no held-out phrase generalization",
                "Measures intent parsing by the agent, not passive traffic attack accuracy",
                "No controller, postcondition verification, PCAP, shaping or fulfilment latency",
                "No retry after malformed or truncated output; failures retained"]}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in args.output.iterdir() if p.is_file()}
        (args.output / "artifact_sha256.json").write_text(json.dumps(hashes, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["completed"] and all(r["api_success"] for r in records) else 1
    finally:
        for adapter in adapters.values():
            adapter.close()
        if token is not None:
            if old is None:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            else:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = old


if __name__ == "__main__":
    raise SystemExit(main())
