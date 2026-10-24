"""Local mTLS qualification; retain EVERY case, no automatic retries or EC2."""
import argparse, hashlib, importlib.metadata, json, platform, ssl, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from intent_leakage.experiments.local_transport_lab import successful_case


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    plan = {'kind': 'local_transport_engineering_qualification', 'cell_bytes': [128, 512, 1024],
            'channels': ['CT', 'CA'], 'value_bytes': [64, 4096], 'rounds': 256, 'periods_s': [.003, .01],
            'max_lateness_s': .1, 'io_timeout_s': .5, 'automatic_retries': 0,
            'diagnostic_lateness_target_fraction_of_period': .25,
            'wire_capture': False, 'external_calls': False, 'formal_paper_data': False,
            'prior_attempt': 'Initial 3ms-only matrix failed; v2 completed 23/24 with one observed cadence-guard failure. Both are retained. This run freezes the updated admission, authenticated TLS close and diagnostic code; no automatic retries.'}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2))
    source = output / 'source'; source.mkdir()
    for name in ['src/intent_leakage/shaping/framing.py', 'src/intent_leakage/adapters/local_shaped_transport.py',
                 'src/intent_leakage/experiments/local_transport_lab.py', 'tests/test_transport_framing.py',
                 'tests/test_local_shaped_transport.py', 'scripts/qualify_local_transport.py']:
        (source / Path(name).name).write_bytes((ROOT / name).read_bytes())
    results = []; started = time.monotonic()
    for period in plan['periods_s']:
        for cell in plan['cell_bytes']:
            for channel in plan['channels']:
                for size in plan['value_bytes']:
                    spec = {'period_s': period, 'cell_bytes': cell, 'channel': channel, 'value_bytes': size}
                    try:
                        result = successful_case(cell_bytes=cell, rounds=plan['rounds'], period_s=period,
                                                 channel=channel, value='x' * size)
                        result.update(success=True, specification=spec)
                    except Exception as exc:
                        result = {'success': False, 'specification': spec, 'error_type': type(exc).__name__, 'metrics': getattr(exc, 'metrics', [])}
                    results.append(result)
                    (output / 'cases.json').write_text(json.dumps(results, indent=2))
                    (output / 'checkpoint.json').write_text(json.dumps({'state': 'running', 'cases_complete': len(results), 'failed_cases': sum(not r['success'] for r in results)}))
    late = [v['lateness_s'] for r in results for m in r['metrics'] for v in m['application_cells_sent']]
    cells = sum(len(m['application_cells_sent']) for r in results for m in r['metrics'])
    passed = sum(r['success'] for r in results)
    deadline_misses = [{'specification': r['specification'], 'role': m['role'], **m['deadline_miss']} for r in results for m in r['metrics'] if 'deadline_miss' in m]
    late_over_period = sum(v['lateness_s'] > r['specification']['period_s'] for r in results for m in r['metrics'] for v in m['application_cells_sent'])
    precision_ok = not deadline_misses and all(v['lateness_s'] <= r['specification']['period_s'] * plan['diagnostic_lateness_target_fraction_of_period'] for r in results for m in r['metrics'] for v in m['application_cells_sent'])
    summary = {'state': 'complete' if passed == len(results) else 'complete_with_failures',
               'cases': len(results), 'success_cases': passed, 'failed_cases': len(results) - passed,
               'cell_bytes_sizes': plan['cell_bytes'], 'channels': plan['channels'],
               'application_cells_written': cells, 'maximum_application_write_lateness_s': max(late, default=0),
               'elapsed_s': time.monotonic() - started,
               'deadline_misses': deadline_misses, 'writes_later_than_one_period': late_over_period,
               'diagnostic_cadence_precision_target_met': precision_ok,
               'versions': {'python': platform.python_version(), 'openssl': ssl.OPENSSL_VERSION,
                            'cryptography': importlib.metadata.version('cryptography')},
               'local_mtls_transport_implemented': True,
               'causal_read_write_and_owned_rollback_verified_in_successful_cases': True,
               'external_connections': 0, 'aws_requests': 0,
               'live_controller_adapter_implemented': False, 'full_paper_ec2_ready': False,
               'privacy_or_cost_effectiveness_proven': False,
               'remaining_gates': ['Inspect all case failures and the framing/transport fault tests.',
                   'Integrate an explicit controller transaction/verification/rollback adapter; local cancellation cannot undo arbitrary remote mutations.',
                   'Validate packet/TLS-record behavior and cadence under realistic delay and retransmissions.',
                   'Fit application-message training profiles and matched-cost baselines; freeze overflow policy.',
                   'Complete 15-class semantics and fresh confirmatory manifests.'],
               'limitations': ['Only fixed application cells, not constant TLS record/IP packet sizes.',
                   'Local monotonic epoch does not solve distributed clock synchronization.',
                   'Inert dummy cells terminate at the tunnel; no controller decoy reads.',
                   'Cancellation/rollback covers only short local actions through RequestContext.commit.',
                   'No live baseline latency/byte cost or privacy comparison.']}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'checkpoint.json').write_text(json.dumps({'state': summary['state'], 'cases_complete': len(results), 'failed_cases': summary['failed_cases']}, indent=2))
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    (output / 'checksums.json').write_text(json.dumps(hashes, indent=2))
    (ROOT / 'outputs/local_transport_latest.json').write_text(json.dumps({'directory': str(output), **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args(); run(args.output)
