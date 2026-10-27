"""Export only aggregate revision statistics from locally held detailed outputs.

The source folders are not part of the public revision supplement. No fitting,
collection, networking, packet copying or per-run export is performed.
"""
import argparse
import json
from pathlib import Path


def read(folder, name):
    return json.loads((folder / name).read_text())


def select(row, keys):
    return {key: row[key] for key in keys}


def summarize(smoke, archive):
    s = read(smoke, 'report.json')
    a = read(archive, 'report.json')
    predictions = read(smoke, 'predictions.json')
    trees = []
    for row in s['tree_results']:
        group = [x for x in predictions if all(x[k] == row[k] for k in ('cohort', 'target', 'view'))]
        labels = ['I01', 'I02', 'I15'] if row['target'] == 'class' else ['1', '2', '4']
        confusion = [[sum(x['truth'] == y and x['prediction'] == p for x in group)
                      for p in labels] for y in labels]
        assert sum(map(sum, confusion)) == row['test_rows'] == 72
        trees.append({**select(row, ('cohort', 'target', 'view', 'train_rows', 'test_rows',
                                    'balanced_accuracy', 'macro_f1', 'archived_full_rf_ba_same_split')),
                      'train_session_count': len(row['train_sessions']),
                      'test_session_count': len(row['test_sessions']),
                      'labels': labels, 'confusion': confusion})
    transfers = [select(x, ('cohort', 'target', 'view', 'train_rows', 'test_rows',
                            'balanced_accuracy', 'macro_f1', 'labels', 'confusion'))
                 for x in a['transfer_results']]
    accounting = read(smoke, 'cohort_accounting.json')
    accounting_keys = ('cohort', 'class_id', 'scope_q', 'archived_successful',
                       'matched_analyzed', 'unmatched_successful', 'all_attempts',
                       'execution_failures', 'verification_failures')
    body_keys = ('class_id', 'runs', 'exchanges', 'max_request_bytes',
                 'max_response_bytes', 'responses_above_64KiB')
    scheduler_keys = ('mode', 'loaded', 'cases', 'writes', 'median_ms', 'p95_ms',
                      'max_ms', 'later_than_10ms', 'later_than_2_5ms',
                      'runtime_abort_threshold_ms')
    assert all(x['record_fields_exact'] and x['features_exact'] for x in s['engineering_replays'])
    assert all(x['exact'] for x in a['tls12_pcap_replays'])
    return {
        'schema_version': 1,
        'release_scope': 'Code and aggregate statistics only for this revision supplement.',
        'new_collection_runs': 0, 'cloud_requests': 0, 'prospective_confirmation': False,
        'count_tree_protocol': {
            'fits': len(trees), 'max_depth': s['tree']['max_depth'],
            'min_samples_leaf': s['tree']['min_samples_leaf'],
            'class_weight': s['tree']['class_weight'],
            'random_state': s['tree']['random_state'], 'split_seed': s['split_seed'],
            'feature': s['feature'], 'splits_per_mode': 1,
        },
        'tree_results': trees,
        'transfer_protocol': {
            'saved_models': len(transfers), 'new_fits': 0,
            'selection_rule': a['selection_rule'],
            'destination_runs': a['verified_engineering_runs'],
            'destination_sessions': 2, 'topologies': 2,
            'execution': 'Direct controller driver; no MCP or LLM.',
            'engineering_protocol': a['engineering_protocol'],
            'overlap_with_original_runs': a['run_overlap_with_original_input'],
        },
        'transfer_results': transfers,
        'body_summary': [select(x, body_keys) for x in a['body_summary']],
        'scheduler_summary': [select(x, scheduler_keys) for x in a['scheduler_summary']],
        'scheduler_protocol': {
            'handler': 'Local synthetic read/write handler, not ONOS.',
            'values_bytes': [64, 1024], 'rounds': 96, 'cell_bytes': 512,
            'period_ms': 10, 'setup_lead_ms': 500, 'load_workers': 1,
            'quantile_method': 'NumPy linear interpolation',
        },
        'capture_check_summary': {
            'readonly_captures_checked': len(s['engineering_replays']),
            'readonly_records_total': sum(x['records'] for x in s['engineering_replays']),
            'readonly_records_and_features_matched': True,
            'tls12_captures_checked': len(a['tls12_pcap_replays']),
            'tls12_records_total': sum(x['records'] for x in a['tls12_pcap_replays']),
            'tls12_normalized_records_matched': True,
            'constructed_header_fixture_passed': s['constructed_fixture_passed'],
            'constructed_header_fixture_records': s['constructed_fixture_tls_records'],
        },
        'cohort_accounting': {
            'cells': [select(x, accounting_keys) for x in accounting['cells']],
            'warning': accounting['warning'],
        },
        'limitations': s['limitations'] + a['limitations'] + [
            'Public verification checks integrity and aggregate arithmetic; it does not replay private executions.',
            'Individual runs, features, predictions, traces, body observations, sender writes and additional PCAPs are withheld.',
            'Capture replay outcomes are reported aggregate observations, not independently replayable from this supplement.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.smoke, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
