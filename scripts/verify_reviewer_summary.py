"""Verify public aggregate integrity and arithmetic, without fitting or raw data.

This verifies consistency of the released tables, not independent reproduction
of private feature extraction, inference or scheduler observations.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_result(row, expected_rows):
    matrix = row['confusion']
    labels = ['I01', 'I02', 'I15'] if row['target'] == 'class' else ['1', '2', '4']
    require(row['labels'] == labels, 'Unexpected label order')
    require(len(matrix) == 3 and all(len(v) == 3 for v in matrix), 'Expected 3x3 matrix')
    require(all(type(v) is int and v >= 0 for r in matrix for v in r), 'Invalid counts')
    support = [sum(r) for r in matrix]
    require(support == [expected_rows // 3] * 3, 'Unexpected class support')
    require(row['test_rows'] == expected_rows, 'Unexpected test size')
    ba = sum(matrix[i][i] / support[i] for i in range(3)) / 3
    f1 = sum(2 * matrix[i][i] / (support[i] + sum(r[i] for r in matrix)) for i in range(3)) / 3
    require(math.isclose(ba, row['balanced_accuracy']), 'Balanced accuracy mismatch')
    require(math.isclose(f1, row['macro_f1']), 'Macro F1 mismatch')


def verify(data):
    path = data / 'reviewer_summary.json'
    expected = dict((name, digest) for digest, name in
                    (line.split(maxsplit=1) for line in (data / 'SHA256SUMS.txt').read_text().splitlines() if line.strip()))
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name], 'SHA-256 mismatch')
    summary = json.loads(path.read_text())
    require(summary['new_collection_runs'] == summary['cloud_requests'] == 0, 'Unexpected new execution')
    require(summary['prospective_confirmation'] is False, 'Incorrect validation scope')
    groups = {(c, t, v) for c in ('mcp', 'native') for t in ('class', 'scope') for v in ('ct', 'ca', 'ctca')}
    for field, rows in [('tree_results', 72), ('transfer_results', 36)]:
        results = summary[field]
        require(len(results) == 12 and {(r['cohort'], r['target'], r['view']) for r in results} == groups, 'Missing/duplicate result')
        for row in results:
            verify_result(row, rows)
    bodies = summary['body_summary']
    require(sum(r['runs'] for r in bodies) == 36 and sum(r['exchanges'] for r in bodies) == 244, 'Body count mismatch')
    require(sum(r['responses_above_64KiB'] for r in bodies) == 2, 'Capacity count mismatch')
    stress = summary['scheduler_summary']
    require(len(stress) == 4 and sum(r['cases'] for r in stress) == 8 and sum(r['writes'] for r in stress) == 1536, 'Scheduler count mismatch')
    for row in stress:
        require(0 <= row['median_ms'] <= row['p95_ms'] <= row['max_ms'], 'Invalid quantile order')
        require(0 <= row['later_than_10ms'] <= row['later_than_2_5ms'] <= row['writes'], 'Invalid lateness counts')
    cells = summary['cohort_accounting']['cells']
    require(len(cells) == 18 and sum(r['matched_analyzed'] for r in cells) == 504, 'Cohort count mismatch')
    for row in cells:
        require(row['archived_successful'] == row['matched_analyzed'] + row['unmatched_successful'], 'Accounting mismatch')
        require(all(row[k] is None for k in ('all_attempts', 'execution_failures', 'verification_failures')), 'Unknown denominators must remain null')
    return {'aggregate_tables_verified': 24, 'body_exchanges_reported': 244,
            'scheduler_writes_reported': 1536, 'new_fits': 0,
            'raw_data_replayed': False, 'verification_scope': 'Integrity and aggregate arithmetic only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    print(json.dumps(verify(parser.parse_args().data), indent=2))
