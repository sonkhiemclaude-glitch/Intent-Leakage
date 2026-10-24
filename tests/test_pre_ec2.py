import pytest
from intent_leakage.experiments.pre_ec2 import fit_record_envelope, replay_record_envelope, padded_sequence, slot_requirements


def test_profiles_are_train_only_and_not_deployable():
    training = [[[100, 0], [-200, .02]], [[300, .1]]]
    p = fit_record_envelope(training, period_s=.02, quantile=1, margin=1)
    original = dict(p)
    assert all(not replay_record_envelope(p, s)['overflow'] for s in training)
    assert replay_record_envelope(p, [[99999, 100]])['overflow']
    assert p == original and p['deployable'] is False


def test_padding_keeps_count_direction_timing_and_does_not_shrink():
    seq = [[100, 0], [-800, .1], [50, .2]]
    assert padded_sequence(seq, 256) == [[256, 0], [-800, .1], [256, .2]]


def test_overflow_separate_from_fixed_schedule_and_success():
    p = fit_record_envelope([[[100, 0]]], period_s=.02, quantile=1, margin=1)
    a = replay_record_envelope(p, [[100, 0]])
    b = replay_record_envelope(p, [[999, 10]])
    assert a['observable_schedule_sha256'] == b['observable_schedule_sha256']
    assert not a['overflow'] and b['size_overflow'] and b['capacity_overflow']
    assert not a['execution_success_measured']


def test_direction_fifo_allocates_extra_capacity():
    assert slot_requirements([[100, 0], [100, 0]], .02)['rounds'] == 2
    assert slot_requirements([[100, 0], [-100, 0]], .02)['rounds'] == 1


@pytest.mark.parametrize('sequence', [[[0, 0]], [[100, -1]], [[1.2, 0]], [[100, float('nan')]]])
def test_invalid_trace_rejected(sequence):
    with pytest.raises(ValueError):
        slot_requirements(sequence, .02)
