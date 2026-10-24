"""Deterministic application-cell qualification on a controlled logical clock.

Uses the real codec/reassembler. Handler completion is scheduled only after a
complete request arrives. No TLS, controller I/O or OS timing is simulated as
measured evidence. Planning callbacks must not mutate external resources.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import math
from .framing import CellCodec, CellKind, Reassembler


@dataclass(frozen=True)
class PlannedReply:
    payload: bytes
    processing_s: float


def run_logical_session(requests, planner, *, cell_bytes=512, rounds=128,
                        period_s=.01, max_message_bytes=65536, stop_at=None):
    if type(rounds) is not int or rounds <= 0 or not math.isfinite(period_s) or period_s <= 0:
        raise ValueError('invalid logical envelope')
    if stop_at is not None and (not math.isfinite(stop_at) or stop_at < 0): raise ValueError('invalid stop time')
    codec = CellCodec(cell_bytes=cell_bytes, channel='CT', max_message_bytes=max_message_bytes)
    req = Reassembler(codec, expected_kind=CellKind.REQUEST)
    resp = Reassembler(codec, expected_kind=CellKind.RESPONSE)
    outgoing = {'agent': deque(), 'bridge': deque()}; pending = {}; events = []; completed = []; writes = []
    handler_available = 0.; counter = 0; termination = rounds * period_s
    def event(when, priority, kind, data):
        nonlocal counter
        counter += 1; heapq.heappush(events, (when, priority, counter, kind, data))
    for mid, (ready, payload) in enumerate(requests, 1):
        if not math.isfinite(ready) or ready < 0: raise ValueError('invalid request readiness')
        parts = codec.fragment(CellKind.REQUEST, mid, payload)
        event(ready, 0, 'admit', (mid, parts)); pending[mid] = None
    for i in range(rounds):
        event(i * period_s, 2, 'cell', ('agent', i))
        event((i + .5) * period_s, 2, 'cell', ('bridge', i))
    failures = []; handler_events = []
    while events:
        when, _, _, kind, data = heapq.heappop(events)
        if when >= termination or (stop_at is not None and when >= stop_at): break
        if kind == 'admit':
            mid, parts = data
            if len(outgoing['agent']) + len(parts) > rounds - sum(v['role']=='agent' for v in writes):
                failures.append('slot_admission'); break
            outgoing['agent'].extend(parts)
        elif kind == 'ready_reply':
            mid, payload = data
            outgoing['bridge'].extend(codec.fragment(CellKind.RESPONSE, mid, payload))
        else:
            role, sequence = data; part = outgoing[role].popleft() if outgoing[role] else None
            wire = codec.encode(sequence, part)
            writes.append({'at_s': when, 'role': role, 'bytes': len(wire), 'dummy': part is None})
            incoming = (req if role == 'agent' else resp).accept(codec.decode(wire, expected_sequence=sequence))
            if incoming:
                mid, payload = incoming
                if role == 'agent':
                    start = max(when, handler_available)
                    reply = planner(payload)
                    if not isinstance(reply, PlannedReply) or not math.isfinite(reply.processing_s) or reply.processing_s < 0:
                        raise ValueError('invalid planned reply')
                    # Check the response bound now, without truncating or growing cells.
                    codec.fragment(CellKind.RESPONSE, mid, reply.payload)
                    finish = start + reply.processing_s; handler_available = finish
                    handler_events.append({'id': mid, 'request_complete_s': when, 'started_s': start, 'completed_s': finish})
                    event(finish, 1, 'ready_reply', (mid, reply.payload))
                else:
                    if mid not in pending or pending[mid] is not None: raise ValueError('unexpected or duplicate response')
                    pending[mid] = when; completed.append({'id': mid, 'received_s': when, 'payload': payload})
    stopped = stop_at is not None and stop_at < termination
    if stopped: failures.append('explicit_stop')
    if any(v is None for v in pending.values()) or outgoing['agent'] or outgoing['bridge'] or req.incomplete or resp.incomplete:
        failures.append('unfinished_at_termination')
    return {'success': not failures, 'failures': failures, 'writes': writes,
            'handlers': handler_events, 'responses': completed,
            'termination_s': min(termination, stop_at) if stopped else termination,
            'kind': 'logical_clock_only', 'live_timing_measured': False}
