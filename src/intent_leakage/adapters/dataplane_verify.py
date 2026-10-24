"""Bounded, observable ICMP postconditions, outside captured attacker features."""
from __future__ import annotations

import math
import re
import time
from typing import Callable


class DataplaneVerificationError(RuntimeError):
    def __init__(self, evidence: dict):
        self.evidence = evidence
        super().__init__(f"Dataplane postcondition failed: {evidence}")


def verify_reachability(
    probe: Callable[[float], str], expected: bool, *, timeout_s: float = 8.0,
    interval_s: float = 0.2, clock=time.monotonic, sleep=time.sleep,
) -> dict:
    """Allow bounded positive convergence; a reachable quarantine fails immediately.

    probe receives the remaining budget and must bound its own execution. Every
    result is retained, including transient failures. This does not install rules,
    flush neighbors or retry controller operations. A negative check is one probe.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout_s must be positive and finite')
    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError('interval_s must be positive and finite')
    start = clock()
    evidence = {'expected_reachable': expected, 'budget_s': timeout_s, 'probes': []}
    while clock() - start < timeout_s:
        before = clock()
        output = probe(timeout_s - (before - start))
        after = clock()
        reachable = re.search(r'\b1 (?:packets )?received\b', output) is not None
        valid = re.search(r'\b1 (?:packets? )?transmitted\b', output) is not None
        evidence['probes'].append({'start_s': before - start, 'end_s': after - start,
                                   'reachable': reachable, 'valid_icmp_result': valid, 'output': output})
        evidence.update(elapsed_s=after - start, success=valid and reachable == expected and after - start <= timeout_s)
        if evidence['success']:
            return evidence
        if not expected:
            break
        remaining = timeout_s - (clock() - start)
        if remaining > 0:
            sleep(min(interval_s, remaining))
    evidence.update(elapsed_s=clock() - start, success=False)
    raise DataplaneVerificationError(evidence)
