from dataclasses import replace
import json
import socket
import ssl
import threading
import time
import pytest
pytest.importorskip('cryptography')
pytestmark = pytest.mark.realtime
from intent_leakage.adapters.local_shaped_transport import (LocalTransportProfile, LocalShapedEndpoint,
    TransportError, EnvelopeOverflow, TransportCancelled)
from intent_leakage.shaping.framing import CellKind, FrameError
from intent_leakage.experiments.local_transport_lab import authenticated_socketpair, make_endpoints, LocalPolicyLab, successful_case


def profile(**kw):
    values = dict(cell_bytes=256, rounds=24, period_s=.005, max_lateness_s=.1, io_timeout_s=.5)
    values.update(kw); return LocalTransportProfile(**values)


def write(value='new'):
    return json.dumps({'op': 'write', 'resource': 'owned_policy', 'value': value}).encode()


def finish(agent, bridge):
    agent.close(); bridge.close()
    return agent.wait(2), bridge.wait(2)


@pytest.mark.parametrize('cell_bytes', [128, 512, 1024])
def test_actual_mtls_fragmentation_causality_fixed_horizon_and_rollback(cell_bytes):
    result = successful_case(cell_bytes=cell_bytes)
    assert result['tls'] == 'TLSv1.3'
    for m in result['metrics']:
        assert m['success'] and len(m['application_cells_sent']) == result['profile']['rounds']
        assert {e['application_bytes'] for e in m['application_cells_sent']} == {cell_bytes}
        assert m['closed_at'] >= m['planned_termination']
    assert result['handler_calls'] == 2 and result['causal_read_after_acknowledgment']
    assert result['owned_state_restored'] and result['foreign_state_preserved']
    assert not any(result['claims'].values())


def test_dummy_only_session_never_calls_handler_and_no_early_close():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab)
        a.start(); b.start(); metrics = [a.wait(2), b.wait(2)]
        assert all(m['success'] and m['dummy_cells_sent'] == 24 for m in metrics)
        assert not lab.calls and lab.cleanup_count == 1
        assert all(m['closed_at'] >= m['planned_termination'] for m in metrics)


def test_oversize_request_and_queue_limit_do_not_reach_handler():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(max_pending=1, max_message_bytes=128), lab=lab)
        with pytest.raises(FrameError): a.submit(b'x' * 129)
        first = a.submit(write())
        with pytest.raises(EnvelopeOverflow): a.submit(write('extra'))
        a.start(); b.start(); assert json.loads(first.result(2))['value'] == 'new'
        assert a.wait(2)['success'] and b.wait(2)['success']
        assert len(lab.calls) == 1 and lab.state['owned_policy'] == 'original'


def test_delayed_handler_cannot_mutate_after_termination_or_cleanup():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(delay_s=.2); a, b = make_endpoints(pair, profile=profile(rounds=8), lab=lab)
        future = a.submit(write()); a.start(); b.start()
        assert not a.wait(2)['success'] and not b.wait(2)['success']
        with pytest.raises(TransportError): future.result()
        time.sleep(.25)
        assert not lab.calls and lab.state == {'foreign': 'keep', 'owned_policy': 'original'}
        assert lab.cleanup_count == 1


def test_handler_failure_after_local_commit_rolls_back_owned_state():
    class FailingLab(LocalPolicyLab):
        def handle(self, payload, context):
            super().handle(payload, context)
            raise ValueError('injected after commit')
    with authenticated_socketpair() as pair:
        lab = FailingLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab)
        future = a.submit(write()); a.start(); b.start()
        assert not a.wait(2)['success'] and not b.wait(2)['success']
        with pytest.raises(TransportError): future.result()
        assert len(lab.calls) == 1 and lab.state['owned_policy'] == 'original'
        assert lab.state['foreign'] == 'keep' and lab.cleanup_count == 1


def test_duplicate_application_message_does_not_repeat_mutation():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab)
        future = a.submit(write())
        # Authenticated malicious peer sends a second message with the same ID.
        a._enqueue(a.codec.fragment(CellKind.REQUEST, 1, write('second')))
        a.start(); b.start(); a.wait(2); metric = b.wait(2)
        assert metric['error_type'] == 'FrameError' and len(lab.calls) <= 1
        assert lab.state['owned_policy'] == 'original' and lab.cleanup_count == 1


@pytest.mark.parametrize('mismatch', ['profile', 'session'])
def test_mismatched_binding_rejected_before_application(mismatch):
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); p = profile(); a, b = make_endpoints(pair, profile=p, lab=lab, bridge_profile=replace(p, cell_bytes=512) if mismatch == 'profile' else None)
        if mismatch == 'session': b.session_id = bytes(16)
        a.start(); b.start(); assert not a.wait(2)['success'] and not b.wait(2)['success']
        assert not lab.calls and lab.cleanup_count == 1


def test_untrusted_client_certificate_rejected():
    with pytest.raises(ssl.SSLError):
        with authenticated_socketpair(trust_client=False): pytest.fail('untrusted client accepted')


def test_non_tls_socket_rejected():
    left, right = socket.socketpair()
    try:
        with pytest.raises(TransportError): LocalShapedEndpoint(left, role='agent', profile=profile(), start_at=time.monotonic(), session_id=bytes(16))
    finally: left.close(); right.close()


def test_explicit_stop_is_bounded_and_cleanup_idempotent():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=100), lab=lab)
        future = a.submit(write('x' * 10000)); a.start(); b.start(); time.sleep(.13)
        metrics = finish(a, b); a.close(); b.close()
        assert all(not m['success'] for m in metrics)
        with pytest.raises(TransportError): future.result()
        assert lab.cleanup_count == 1 and lab.state['owned_policy'] == 'original'


def test_cleanup_failure_is_not_reported_as_success():
    class CleanupFailure(LocalPolicyLab):
        def cleanup(self):
            super().cleanup(); raise RuntimeError('injected cleanup failure')
    with authenticated_socketpair() as pair:
        lab = CleanupFailure(); a, b = make_endpoints(pair, profile=profile(rounds=4), lab=lab)
        a.start(); b.start(); a.wait(2); m = b.wait(2)
        assert not m['success'] and m['error_type'] == 'RuntimeError' and m['cleanup_calls'] == 1


def test_allowlist_rejects_foreign_resource_before_mutation():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab)
        future = a.submit(json.dumps({'op': 'write', 'resource': 'foreign', 'value': 'changed'}).encode())
        a.start(); b.start(); a.wait(2); assert not b.wait(2)['success']
        assert not lab.calls and lab.state['foreign'] == 'keep'


def test_fragment_capacity_rejection_before_start():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=1), lab=lab)
        with pytest.raises(EnvelopeOverflow): a.submit(write('x' * 1000))
        # close() before start still closes TLS and invokes cleanup exactly once.
        finish(a, b); assert lab.cleanup_count == 1 and not lab.calls


def test_late_start_fails_without_handler_calls():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab, lead_s=-1)
        a.start(); b.start(); assert not a.wait(2)['success'] and not b.wait(2)['success']
        assert not lab.calls and lab.cleanup_count == 1


def test_incomplete_request_does_not_execute_handler():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=4), lab=lab)
        # Simulate a verified peer that truncates an application message but
        # keeps emitting dummy cells. No partial command can reach the lab.
        pieces = a.codec.fragment(CellKind.REQUEST, 1, write('x' * 1000))
        a._outgoing.append(pieces[0]); a._outgoing_messages = 1
        a.start(); b.start(); a.wait(2); metric = b.wait(2)
        assert not metric['success'] and metric['error_type'] == 'EnvelopeOverflow'
        assert not lab.calls and lab.cleanup_count == 1


def test_response_overflow_after_commit_still_rolls_back():
    class OversizeResponse(LocalPolicyLab):
        def handle(self, payload, context):
            super().handle(payload, context)
            return bytes(129)
    with authenticated_socketpair() as pair:
        lab = OversizeResponse(); a, b = make_endpoints(pair, profile=profile(max_message_bytes=128), lab=lab)
        future = a.submit(write()); a.start(); b.start(); a.wait(2); metric = b.wait(2)
        assert not metric['success'] and metric['error_type'] == 'FrameError'
        assert len(lab.calls) == 1 and lab.state['owned_policy'] == 'original'
        assert lab.cleanup_count == 1


def test_cleanup_does_not_revert_concurrent_foreign_change():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=40), lab=lab)
        future = a.submit(write()); a.start(); b.start(); future.result(2)
        lab.state['foreign'] = 'concurrent update'
        a.wait(2); b.wait(2)
        assert lab.state == {'foreign': 'concurrent update', 'owned_policy': 'original'}


def test_local_queues_retain_write_order():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=40), lab=lab)
        first = a.submit(write('first')); second = a.submit(write('second'))
        read = a.submit(json.dumps({'op': 'read', 'resource': 'owned_policy'}).encode())
        a.start(); b.start()
        assert json.loads(first.result(2))['value'] == 'first'
        assert json.loads(second.result(2))['value'] == 'second'
        assert json.loads(read.result(2))['value'] == 'second'
        assert a.wait(2)['success'] and b.wait(2)['success']
        assert [v['op'] for v in lab.calls] == ['write', 'write', 'read']


def test_slot_admission_failure_does_not_leave_a_pending_request():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=4), lab=lab)
        # A queued fragmented request occupies all four remaining send slots.
        a.submit(write('x' * 650))
        with pytest.raises(EnvelopeOverflow): a.submit(write())
        assert len(a._pending) == 1 and a._next_id == 2
        finish(a, b)


def test_closed_slot_budget_cannot_accept_new_requests():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=1), lab=lab)
        a._cells_reserved = 1
        with pytest.raises(EnvelopeOverflow): a.submit(write())
        assert not a._pending and not a._outgoing
        finish(a, b)


def test_successful_sessions_exchange_authenticated_tls_close_notify():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(), lab=lab)
        a.start(); b.start(); metrics = [a.wait(2), b.wait(2)]
        assert all(m['success'] and m['tls_close_notify_completed'] for m in metrics)


def test_injected_cadence_overrun_fails_closed_and_records_deadline():
    # Isolate the guard from peer scheduling: an overloaded OS may make the
    # other endpoint fail first, which must not invalidate this unit test.
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); p = profile(rounds=20, period_s=.01, max_lateness_s=.02)
        a, b = make_endpoints(pair, profile=p, lab=lab)
        with pytest.raises(EnvelopeOverflow) as captured:
            a._wait_slot(time.monotonic() - .05)
        a._abort(captured.value)
        metric, _ = finish(a, b)
        assert not metric['success'] and metric['failure_code'] == 'cadence_deadline'
        assert metric['deadline_miss']['lateness_s'] > p.max_lateness_s
        assert not lab.calls and lab.state['owned_policy'] == 'original' and lab.cleanup_count == 1


def test_missing_peer_cell_times_out_and_cleans_up():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); p = profile(rounds=40, io_timeout_s=.2)
        a, b = make_endpoints(pair, profile=p, lab=lab)
        # Authenticated bridge intentionally withholds its response/dummy cells.
        b._send = lambda index: None
        a.submit(write()); a.start(); b.start(); metrics = [a.wait(2), b.wait(2)]
        assert all(not m['success'] for m in metrics)
        assert lab.cleanup_count == 1 and lab.state['owned_policy'] == 'original'


def test_context_rejects_expired_commit_even_if_pump_has_not_cancelled_it():
    from intent_leakage.adapters.local_shaped_transport import RequestContext
    context = RequestContext(deadline=time.monotonic() - 1)
    state = []
    with pytest.raises(TransportCancelled): context.commit(lambda: state.append('late'))
    assert not state


def test_expired_epoch_rejects_admission_before_any_cell_or_future():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); a, b = make_endpoints(pair, profile=profile(rounds=1), lab=lab, lead_s=-1)
        with pytest.raises(EnvelopeOverflow): a.submit(write())
        assert not a._pending and not a._outgoing
        finish(a, b)


def test_slow_drip_cannot_reset_whole_cell_timeout_indefinitely():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); p = profile(rounds=40, io_timeout_s=.3)
        a, b = make_endpoints(pair, profile=p, lab=lab)
        def drip(index):
            cell = b.codec.encode(index)
            for byte in cell:
                b.sock.sendall(bytes([byte])); time.sleep(.08)
        b._send = drip
        a.submit(write()); started = time.monotonic(); a.start(); b.start()
        metrics = [a.wait(2), b.wait(2)]
        assert time.monotonic() - started < 1.5
        assert not any(m['success'] for m in metrics)
        assert lab.cleanup_count == 1 and lab.state['owned_policy'] == 'original'


def test_setup_lead_does_not_consume_first_cell_io_budget():
    with authenticated_socketpair() as pair:
        lab = LocalPolicyLab(); p = profile(rounds=12, period_s=.01, io_timeout_s=.1)
        a, b = make_endpoints(pair, profile=p, lab=lab, lead_s=.5)
        a.start(); b.start(); metrics = [a.wait(2), b.wait(2)]
        assert all(m['success'] for m in metrics)
        assert all(len(m['application_cells_sent'])==12 for m in metrics)
