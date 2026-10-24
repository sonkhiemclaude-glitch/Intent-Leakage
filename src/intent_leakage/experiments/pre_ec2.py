"""Offline TLS-record metadata capacity probes, never live shaping evidence.

The replay retains original readiness times. It cannot model response delays
caused by buffering requests, TLS packetization, or application dependencies.
Profiles from this module MUST NOT be deployed as application-message profiles.
"""
from __future__ import annotations
import hashlib
import json
import math
from typing import Sequence
import numpy as np


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_sequence(sequence: Sequence) -> None:
    for size, gap in sequence:
        if not math.isfinite(size) or not size or not float(size).is_integer():
            raise ValueError('Signed TLS record sizes must be finite nonzero integers')
        if not math.isfinite(gap) or gap < 0:
            raise ValueError('Intervals must be finite and nonnegative')


def slot_requirements(sequence: Sequence, period_s: float) -> dict:
    """FIFO capacity with staggered direction slots and immutable readiness."""
    if not math.isfinite(period_s) or period_s <= 0:
        raise ValueError('period_s must be positive and finite')
    validate_sequence(sequence)
    last = {-1: -1, 1: -1}; counts = {-1: 0, 1: 0}; buckets = {-1: 0, 1: 0}
    ready = 0.0
    for signed_size, gap in sequence:
        ready += gap; direction = 1 if signed_size > 0 else -1
        offset = 0.0 if direction == 1 else period_s / 2
        index = max(last[direction] + 1, 0, math.ceil((ready - offset) / period_s - 1e-9))
        last[direction] = index; counts[direction] += 1
        buckets[direction] = max(buckets[direction], int(abs(signed_size)))
    return {'rounds': max(last.values()) + 1, 'counts': counts, 'buckets': buckets,
            'elapsed_s': ready, 'record_bytes': sum(abs(p[0]) for p in sequence)}


def fit_record_envelope(training_sequences: Sequence, *, period_s: float,
                        quantile: float, margin: float) -> dict:
    if not training_sequences or not 0 < quantile <= 1 or not math.isfinite(margin) or margin < 1:
        raise ValueError('Require training data, valid quantile, margin >= 1')
    requirements = [slot_requirements(s, period_s) for s in training_sequences]
    rounds = max(1, math.ceil(float(np.quantile([r['rounds'] for r in requirements], quantile, method='higher')) * margin))
    sizes = [abs(p[0]) for s in training_sequences for p in s]
    bucket = max(64, math.ceil(float(np.quantile(sizes or [64], quantile, method='higher')) * margin / 64) * 64)
    return {'schema_version': 1, 'kind': 'offline_tls_record_capacity_only',
            'deployable': False, 'quantile': quantile, 'margin': margin,
            'period_s': period_s, 'rounds': rounds, 'bucket_bytes': bucket,
            'termination_s': rounds * period_s,
            'training_sequences_sha256': digest(training_sequences),
            'limitation': 'Original readiness frozen; no causal application replay or live transport.'}


def replay_record_envelope(profile: dict, sequence: Sequence) -> dict:
    if profile.get('kind') != 'offline_tls_record_capacity_only' or profile.get('deployable') is not False:
        raise ValueError('Only non-deployable metadata capacity profiles are accepted')
    req = slot_requirements(sequence, profile['period_s'])
    size_overflow = max(req['buckets'].values()) > profile['bucket_bytes']
    capacity_overflow = req['rounds'] > profile['rounds']
    padded_bytes = 2 * profile['rounds'] * profile['bucket_bytes']
    canonical = {k: profile[k] for k in ('period_s', 'rounds', 'bucket_bytes', 'termination_s')}
    return {'size_overflow': size_overflow, 'capacity_overflow': capacity_overflow,
            'overflow': size_overflow or capacity_overflow,
            'original_record_bytes': req['record_bytes'], 'scheduled_record_bytes': padded_bytes,
            'record_byte_overhead_ratio': padded_bytes / req['record_bytes'] - 1 if req['record_bytes'] else None,
            'nominal_duration_added_s': profile['termination_s'] - req['elapsed_s'],
            'observable_schedule_sha256': digest(canonical),
            'execution_success_measured': False}


def padded_sequence(sequence: Sequence, bucket: int) -> list:
    if bucket <= 0:
        raise ValueError('Padding bucket must be positive')
    validate_sequence(sequence)
    return [[(1 if size > 0 else -1) * max(abs(size), bucket), gap] for size, gap in sequence]
