#!/usr/bin/env python3
"""Small, token-capped live Bedrock connection probe; no controller actions.

Credentials may be passed through an existing SDK environment/profile, a local
private .env, or encrypted SSH stdin. Secret values and raw AWS exception text
are never written to the report. This is a live connectivity test, not synthetic
data and not a complete networking experiment.
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
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from intent_leakage.adapters.bedrock import BedrockAdapter, BedrockConfig


def load_bearer_token(path: Path) -> str:
    """Read only the two supported key aliases as data; never source a shell."""
    values = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if text.startswith("export "):
            text = text[7:].lstrip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        if name.strip().lower() not in {"bedrock_key", "aws_bearer_token_bedrock"}:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            values.append(value)
    if not values or len(set(values)) != 1:
        raise ValueError("A single nonempty Bedrock bearer key is required")
    if any(c.isspace() for c in values[0]):
        raise ValueError("Bedrock bearer key must not contain whitespace")
    return values[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument("--env-file", type=Path)
    credentials.add_argument("--token-stdin", action="store_true")
    args = parser.parse_args()
    if not args.allow_external:
        parser.error("This live probe requires --allow-external")
    if not 1 <= args.max_output_tokens <= 128 or not 1 <= len(args.model) <= 2:
        parser.error("Smoke probe is limited to two requests, each with 1-128 output tokens")
    # Credential acquisition occurs only after command validation. On Mac Studio
    # the caller sends the token via SSH stdin, without creating a secret file.
    token = None
    try:
        if args.env_file:
            token = load_bearer_token(args.env_file)
        elif args.token_stdin:
            token = sys.stdin.readline().rstrip("\r\n")
            if not token or any(c.isspace() for c in token):
                raise ValueError("A nonempty bearer token is required on stdin")
    except (ValueError, OSError) as exc:
        print(json.dumps({"credential_setup_failed": True, "error_type": type(exc).__name__}))
        return 2
    old_token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if token is not None:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = token
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        prompt = {"messages": [{"role": "user", "content": [{"text": "Reply with the single word READY."}]}]}
        for model in args.model:
            config = BedrockConfig(model_id=model, region=args.region,
                streaming=args.streaming, decoding_parameters={"temperature": 0.0},
                max_response_tokens=args.max_output_tokens,
                retry_policy={"mode": "standard", "total_max_attempts": 1},
                connection_pooling="new_client_per_request", read_timeout_s=45)
            adapter = BedrockAdapter(config, allow_external=True)
            result = {"model_id": model, "region": args.region,
                      "streaming": args.streaming, "configuration": asdict(config)}
            start = time.perf_counter()
            try:
                if args.streaming:
                    texts, usage, stop, event_count = [], {}, None, 0
                    for event in adapter.invoke_stream(prompt):
                        event_count += 1
                        delta = event.get("contentBlockDelta", {}).get("delta", {})
                        if "text" in delta:
                            texts.append(delta["text"])
                        if "metadata" in event:
                            usage = event["metadata"].get("usage", {})
                        if "messageStop" in event:
                            stop = event["messageStop"].get("stopReason")
                    result.update({"response_text": "".join(texts), "usage": usage,
                                   "stop_reason": stop, "event_count": event_count})
                else:
                    response = adapter.invoke(prompt)
                    text = "".join(v.get("text", "") for v in response.get("output", {}).get("message", {}).get("content", []))
                    result.update({"response_text": text, "usage": response.get("usage", {}),
                        "stop_reason": response.get("stopReason"),
                        "http_status": response.get("ResponseMetadata", {}).get("HTTPStatusCode")})
                # The prompt never contains credentials; redact defensively if a
                # provider unexpectedly echoes the bearer token.
                for secret in (token, old_token):
                    if secret:
                        result["response_text"] = result["response_text"].replace(secret, "<redacted>")
                result["success"] = True
            except Exception as exc:
                detail = getattr(exc, "response", {})
                result.update({"success": False, "error_type": type(exc).__name__,
                    "aws_error_code": detail.get("Error", {}).get("Code"),
                    "http_status": detail.get("ResponseMetadata", {}).get("HTTPStatusCode")})
            finally:
                adapter.close()
            result["client_elapsed_s"] = time.perf_counter()-start
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "configuration"}), flush=True)
        report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_kind": "live_bedrock_connection_smoke", "synthetic": False,
            "paper_results_claimed": False, "network_capture_performed": False,
            "controller_actions": 0, "credentials_persisted": False,
            "host_architecture": platform.machine(), "python": platform.python_version(),
            "boto3": importlib.metadata.version("boto3"), "requests": results,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256((ROOT / "src/intent_leakage/adapters/bedrock.py").read_bytes()).hexdigest()}
        (args.output / "connection_report.json").write_text(json.dumps(report, indent=2)+"\n")
        return 0 if all(r["success"] for r in results) else 1
    finally:
        if token is not None:
            if old_token is None:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            else:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = old_token


if __name__ == "__main__":
    raise SystemExit(main())
