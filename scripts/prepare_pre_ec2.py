"""Local-only exploratory diagnostics and a fail-closed next-experiment gate.
No network, AWS credential loading, SSH, subprocess, or EC2 deployment code.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, math, sys, tarfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from evaluate_real_instrumentation import split_sessions
from intent_leakage.experiments.pre_ec2 import digest, fit_record_envelope, replay_record_envelope, padded_sequence
from intent_leakage.features.extraction import extract_channel_features

GROUPS = {
    'count_only': ['event_count'],
    'structure': ['event_count', 'direction_changes', 'turn_count', 'burst_count'],
    'size_only': ['bytes_direction_pos', 'bytes_direction_neg', 'total_bytes', 'burst_abs_bytes_mean', 'burst_abs_bytes_std', 'burst_abs_bytes_max'],
    'timing_only': ['elapsed_time_s', 'interarrival_mean_s', 'interarrival_std_s', 'interarrival_max_s', 'burst_duration_mean_s', 'burst_duration_std_s', 'burst_duration_max_s'],
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def features(sequence):
    timestamp = 0.; events = []
    for size, gap in sequence:
        timestamp += gap
        events.append({'timestamp_s': timestamp, 'length_bytes': int(abs(size)),
                       'direction': 1 if size > 0 else -1, 'channel': 'CA'})
    return extract_channel_features(events, channel='CA')


def cluster_interval(truth, prediction, sessions):
    """Conditional on this frozen fitted model; not model-selection uncertainty."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    groups = sorted(set(sessions)); rng = np.random.default_rng(84201)
    indices = {g: [i for i, s in enumerate(sessions) if s == g] for g in groups}
    values = []
    for _ in range(1000):
        ix = [i for g in rng.choice(groups, len(groups), replace=True) for i in indices[g]]
        values.append(balanced_accuracy_score(np.array(truth)[ix], np.array(prediction)[ix]))
    return {'low': float(np.quantile(values, .025)), 'high': float(np.quantile(values, .975)),
            'clusters': len(groups), 'conditional_on_fitted_model': True,
            'scope': 'Exploratory session-cluster percentile interval; correlated collection and adaptive reuse remain limitations.'}


def attack(spec, rows, output):
    import joblib, numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    cohort, target, group, seed, control = spec
    source = [r for r in rows if r['cohort'] == cohort]
    train_ids, test_ids = split_sessions({(r['session'], r['topology']) for r in source}, str(seed))
    train = [r for r in source if r['session'] in train_ids]
    test = [r for r in source if r['session'] in test_ids]
    assert not train_ids & test_ids
    bucket = max(abs(p[0]) for r in train for p in r['sequences']['ca']) if group == 'fixed_padding' else None
    allcols = sorted(features(train[0]['sequences']['ca']))
    cols = GROUPS.get(group, allcols)
    def matrix(data):
        return [[f[c] for c in cols] for f in [features(padded_sequence(r['sequences']['ca'], int(bucket))) if bucket else r['_features'] for r in data]]
    def label(r):
        return r['class_id'] if target == 'class' else 'q=' + str(r['scope_q']) if target == 'scope' else r['class_id'] + ':' + str(r['scope_q'])
    y = np.array([label(r) for r in train]); truth = [label(r) for r in test]
    if control: np.random.default_rng(seed + 10000).shuffle(y)
    model = RandomForestClassifier(n_estimators=128, min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=1)
    model.fit(matrix(train), y); prediction = model.predict(matrix(test)).tolist()
    ident = '-'.join(map(str, spec)); mp = output / 'models' / (ident + '.joblib'); joblib.dump(model, mp)
    preds = [{'run_id': r['run_id'], 'session': r['session'], 'truth': t, 'prediction': p} for r, t, p in zip(test, truth, prediction)]
    pp = output / 'predictions' / (ident + '.json'); pp.write_text(json.dumps(preds))
    result = {'id': ident, 'cohort': cohort, 'target': target, 'ablation': group, 'seed': seed,
              'permuted_training_labels': control, 'features': cols, 'balanced_accuracy': balanced_accuracy_score(truth, prediction),
              'train_sessions': sorted(train_ids), 'test_sessions': sorted(test_ids),
              'train_rows': len(train), 'test_rows': len(test), 'training_padding_bucket': bucket,
              'model_sha256': sha(mp), 'predictions_sha256': sha(pp),
              'chance': 1 / (9 if target == 'joint' else 3)}
    if bucket:
        result['padding_oversize_test_runs'] = sum(any(abs(p[0]) > bucket for p in r['sequences']['ca']) for r in test)
        result['padding_record_byte_overhead_median'] = float(np.median([sum(max(abs(p[0]), bucket) for p in r['sequences']['ca']) / sum(abs(p[0]) for p in r['sequences']['ca']) - 1 for r in test]))
    if seed == 0 and not control:
        result['conditional_cluster_interval'] = cluster_interval(truth, prediction, [r['session'] for r in test])
    return result


def run(archive, output, cpus):
    import numpy as np
    from joblib import Parallel, delayed, parallel_config
    start = time.monotonic(); expected = json.loads((archive.parent / 'provenance.json').read_text())['analysis_archive_sha256']
    if sha(archive) != expected: raise ValueError('Input archive hash mismatch')
    with tarfile.open(archive) as t:
        sequence_bytes = t.extractfile('outputs/seq_bundle/input.json').read()
        bundle = json.loads(sequence_bytes)
        for job in ['rf', 'knn256', 'knn512', 'cnn']:
            report = json.load(t.extractfile('outputs/' + job + '/report.json'))
            if job != 'rf' and hashlib.sha256(sequence_bytes).hexdigest() != report['input_sha256']:
                raise ValueError('Sequence input does not match evaluated data')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir(); (output / 'predictions').mkdir()
    cohorts = sorted({r['cohort'] for r in bundle['rows']})
    common = set.intersection(*[{r['session'] for r in bundle['rows'] if r['cohort'] == c} for c in cohorts])
    rows = [r for r in bundle['rows'] if r['session'] in common]
    assert len(rows) == 504 and len(common) == 14 and cohorts == ['mcp', 'native']
    assert all(r['sequences']['ca'] for r in rows)
    for row in rows: row['_features'] = features(row['sequences']['ca'])
    split_manifest = {}
    for seed in range(3):
        tr, te = split_sessions({(r['session'], r['topology']) for r in rows}, str(seed))
        split_manifest[str(seed)] = {'train': sorted(tr), 'test': sorted(te)}
    protocol = {'kind': 'exploratory_offline_pre_ec2', 'paper_results_claimed': False, 'deployable': False,
                'source_archive_sha256': expected, 'sequence_bundle_sha256': hashlib.sha256(sequence_bytes).hexdigest(),
                'matched_rows': len(rows), 'splits': split_manifest, 'ablation_features': GROUPS,
                'RF': {'n_estimators': 128, 'min_samples_leaf': 2, 'class_weight': 'balanced'},
                'readiness': 'Blocked for full paper grid; no external calls permitted by this runner.',
                'adaptive_reuse': 'Existing pilot test data have already been inspected; these diagnostics cannot become confirmatory results.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    specs = [(c, target, group, seed, control) for c in cohorts for target in ['class', 'scope', 'joint']
             for group in [*GROUPS, 'full', 'fixed_padding'] for seed in range(3) for control in [False, True]]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=cpus)(delayed(attack)(spec, rows, output) for spec in specs)
    groups = {}
    for r in results:
        key = (r['cohort'], r['target'], r['ablation'], r['permuted_training_labels'])
        groups.setdefault(key, []).append(r['balanced_accuracy'])
    summaries = [{'cohort': k[0], 'target': k[1], 'ablation': k[2], 'permuted': k[3],
                  'balanced_accuracy_mean': float(np.mean(v)), 'min': min(v), 'max': max(v), 'overlapping_repeats': len(v)} for k, v in groups.items()]
    capacity = []
    for cohort in cohorts:
        for seed in range(3):
            split = split_manifest[str(seed)]
            train = [r for r in rows if r['cohort'] == cohort and r['session'] in split['train']]
            test = [r for r in rows if r['cohort'] == cohort and r['session'] in split['test']]
            for period in [.02, .1]:
                for quantile in [.95, 1.0]:
                    for margin in [1., 1.25]:
                        profile = fit_record_envelope([r['sequences']['ca'] for r in train], period_s=period, quantile=quantile, margin=margin)
                        profile['training_run_ids'] = [r['run_id'] for r in train]
                        simulations = [{'run_id': r['run_id'], 'session': r['session'], 'class_id': r['class_id'], 'scope_q': r['scope_q'],
                                        **replay_record_envelope(profile, r['sequences']['ca'])} for r in test]
                        assert len({v['observable_schedule_sha256'] for v in simulations}) == 1
                        rate = sum(v['overflow'] for v in simulations) / len(simulations)
                        session_failures = sum(any(v['overflow'] for v in simulations if v['session'] == sid) for sid in split['test'])
                        capacity.append({'cohort': cohort, 'seed': seed, 'profile': profile, 'heldout_rows': len(test),
                            'overflow_rate': rate, 'sessions_with_overflow': session_failures, 'heldout_sessions': len(split['test']),
                            'median_record_byte_overhead_ratio': float(np.median([v['record_byte_overhead_ratio'] for v in simulations])),
                            'median_nominal_duration_added_s': float(np.median([v['nominal_duration_added_s'] for v in simulations])),
                            'runs': simulations})
    gate = {'ready_for_full_paper_ec2': False, 'ready_for_live_shaping_ec2': False,
        'ready_for_offline_diagnostics': True, 'ec2_contacted': False, 'aws_requests': 0,
        'blocking_requirements': [
            'Implement causal authenticated CT/CA fixed-frame transport; exact application framing, TLS recordization and connection lifetime must be measured.',
            'Bind safe decoys to an explicit controller read allowlist; decoys must not satisfy independent success checks.',
            'Implement and test all 15 class prerequisites, actual controller mutations, independent dataplane postconditions and inverse cleanup.',
            'Current pilot I01 drop does not preserve management VLAN; reconcile exact catalogue semantics before formal runs.',
            'Freeze shaping groups, training-only fitting policy, overflow response, matched-cost baseline budget and acceptable execution-success thresholds.',
            'Create fresh train/validation/test phrase families and disjoint collection seeds; existing pilot data remain exploratory.',
            'Provision and test the second model and provider/CL capture only if retained in paper claims.',
            'Validate paired privacy-cost estimators, per-class/session overflow uncertainty, and protocol conformance against captured live traffic.'
        ],
        'proposed_next_stages': [
            {'stage': 'local_transport', 'allocation': 'Mac loopback only', 'conditions': ['no_shaping', 'fixed_size_padding', 'full_envelope', 'size_only', 'cadence_only', 'no_decoys', 'no_common_termination'], 'gate': 'No malformed frames, early response delivery, replayed mutation, state-changing dummy or uncaught dependency overflow in fault tests.'},
            {'stage': 'live_shaping_regression', 'allocation': 'T4 only after readiness checks pass', 'classes': ['I01', 'I02', 'I15'], 'q': [1, 2, 4], 'topologies': 2, 'conditions': ['no_shaping', 'fixed_size_padding', 'full_envelope'], 'paired_blocks': 2, 'planned_runs': 108, 'purpose': 'Engineering qualification, excluded from confirmatory paper tests.'},
            {'stage': 'confirmatory', 'allocation': 'Not authorized by this preparation script', 'classes': 15, 'repetitions': None, 'repetitions_selection': 'Choose from pilot variance, desired interval width and budget before opening new test results.', 'test_policy': 'No tuning on test; report all frozen profiles, errors, overflows and infeasible cells.'}
        ],
        'matched_cost_comparison': 'Use a predeclared cost grid; baseline parameters chosen on validation only under the same byte/latency budgets. Report infeasibility and Pareto curves, never pick a test winner.',
        'independent_success': 'Controller state plus applicable ICMP/throughput/path/tenant-isolation probes outside protected traces; preserve foreign resources.',
        'required_faults': ['oversize', 'late_dependency', 'missing_response', 'duplicate_mutation', 'connection_reset', 'capture_drop', 'stop_mid_batch', 'cleanup_failure', 'other_project_activity'],
        'claim_limit': 'A constant synthetic schedule is true by construction; it does not establish live secrecy, execution correctness or low overhead.'}
    report = {'kind': 'offline_exploratory_diagnostics', 'paper_results_claimed': False, 'live_defense_validated': False,
              'fits': len(results), 'matched_rows': len(rows), 'elapsed_s': time.monotonic() - start,
              'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'joblib']},
              'summary': summaries, 'results': results,
              'limitations': ['Only CA record metadata replay from three-class pilots.', 'Padding operates on observed TLS record lengths, not application messages or measured wire bytes.', 'Static readiness cannot simulate causal delays introduced by shaping; nominal replay costs are not live latency.', 'Existing test data reused for exploratory diagnosis; no final profile selected using these scores.', 'Cluster intervals condition on one fitted model and a small number of held-out sessions; no formal certification.']}
    for name, data in [('report.json', report), ('capacity_replay.json', capacity), ('readiness_gate.json', gate)]:
        (output / name).write_text(json.dumps(data, indent=2))
    snapshot = output / 'source'; snapshot.mkdir()
    for source in [Path(__file__), ROOT / 'src/intent_leakage/experiments/pre_ec2.py', ROOT / 'src/intent_leakage/features/extraction.py', ROOT / 'scripts/evaluate_real_instrumentation.py']:
        (snapshot / source.name).write_bytes(source.read_bytes())
    manifest = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
    (output / 'checksums.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'output': str(output), 'fits': len(results), 'elapsed_s': report['elapsed_s'], 'ready_for_ec2': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--archive', type=Path, required=True); p.add_argument('--output', type=Path, required=True); p.add_argument('--cpus', type=int, default=4); args = p.parse_args()
    if not 1 <= args.cpus <= 8: p.error('cpus must be between 1 and 8')
    run(args.archive, args.output, args.cpus)
