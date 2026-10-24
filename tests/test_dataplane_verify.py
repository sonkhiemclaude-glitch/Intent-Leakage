import pytest

from intent_leakage.adapters.dataplane_verify import (
    DataplaneVerificationError, verify_reachability,
)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_positive_convergence_keeps_failed_probes():
    clock = Clock()
    replies = iter(['1 transmitted, 0 received', '1 transmitted, 1 received'])

    def probe(remaining):
        assert remaining > 0
        clock.sleep(.1)
        return next(replies)

    result = verify_reachability(probe, True, clock=clock, sleep=clock.sleep)
    assert result['success']
    assert [p['reachable'] for p in result['probes']] == [False, True]
    assert result['elapsed_s'] == pytest.approx(.4)


def test_permanent_failure_stops_at_deadline_with_evidence():
    clock = Clock()

    def probe(remaining):
        clock.sleep(min(1., remaining))
        return '1 transmitted, 0 received, Destination Host Unreachable'

    with pytest.raises(DataplaneVerificationError) as caught:
        verify_reachability(probe, True, timeout_s=3, clock=clock, sleep=clock.sleep)
    assert clock.now == 3
    assert not caught.value.evidence['success']
    assert len(caught.value.evidence['probes']) == 3


def test_blocked_postcondition_does_not_retry_a_reachable_probe():
    clock = Clock()
    calls = []

    def probe(remaining):
        calls.append(remaining)
        return '1 packets transmitted, 1 received, 0% packet loss'

    with pytest.raises(DataplaneVerificationError):
        verify_reachability(probe, False, clock=clock, sleep=clock.sleep)
    assert len(calls) == 1


def test_late_success_does_not_pass_deadline():
    clock = Clock()

    def probe(remaining):
        clock.sleep(remaining + .01)
        return '1 transmitted, 1 received'

    with pytest.raises(DataplaneVerificationError):
        verify_reachability(probe, True, clock=clock, sleep=clock.sleep)


def test_probe_execution_error_cannot_count_as_blocked():
    clock = Clock()
    with pytest.raises(DataplaneVerificationError):
        verify_reachability(lambda _: 'ping: command not found', False,
                            clock=clock, sleep=clock.sleep)


@pytest.mark.parametrize('timeout,interval', [(0,.2),(float('inf'),.2),(8,0)])
def test_invalid_bounds_rejected(timeout, interval):
    with pytest.raises(ValueError):
        verify_reachability(lambda _: '', True, timeout_s=timeout, interval_s=interval)
