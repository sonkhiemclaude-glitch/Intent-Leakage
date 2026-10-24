from dataclasses import replace
import pytest
from intent_leakage.shaping.framing import CellCodec, CellKind, FrameError, Reassembler, HEADER


@pytest.mark.parametrize('size', [0, 1, 92, 93, 65536])
def test_fragmentation_round_trip_with_exact_cell_size(size):
    codec = CellCodec(cell_bytes=128, channel='CT')
    payload = bytes(i % 251 for i in range(size))
    receiver = Reassembler(codec, expected_kind=CellKind.REQUEST)
    result = None
    for sequence, fragment in enumerate(codec.fragment(CellKind.REQUEST, 1, payload)):
        cell = codec.encode(sequence, fragment)
        assert len(cell) == 128
        result = receiver.accept(codec.decode(cell, expected_sequence=sequence))
    assert result == (1, payload) and not receiver.incomplete


def test_dummy_is_not_a_message_and_has_fixed_size():
    codec = CellCodec(cell_bytes=512, channel='CA')
    receiver = Reassembler(codec, expected_kind=CellKind.REQUEST)
    assert receiver.accept(codec.decode(codec.encode(0), expected_sequence=0)) is None
    assert not receiver.seen and not receiver.incomplete


@pytest.mark.parametrize('field,value', [(0, b'BAD!'), (1, 2), (2, 9), (3, 2), (4, 1), (5, 99), (7, 5), (8, 0), (9, 99999)])
def test_malformed_authenticated_peer_frames_are_rejected(field, value):
    codec = CellCodec(cell_bytes=128, channel='CT')
    original = codec.encode(0, codec.fragment(CellKind.REQUEST, 1, b'hello')[0])
    values = list(HEADER.unpack_from(original)); values[field] = value
    bad = HEADER.pack(*values) + original[HEADER.size:]
    with pytest.raises(FrameError): codec.decode(bad, expected_sequence=0)


def test_replay_rejected_at_both_cell_and_application_layers():
    codec = CellCodec(cell_bytes=128, channel='CT')
    f = codec.fragment(CellKind.REQUEST, 1, b'policy')[0]
    cell = codec.encode(0, f)
    with pytest.raises(FrameError): codec.decode(cell, expected_sequence=1)
    receiver = Reassembler(codec, expected_kind=CellKind.REQUEST)
    assert receiver.accept(f) == (1, b'policy')
    with pytest.raises(FrameError): receiver.accept(codec.decode(codec.encode(1, f), expected_sequence=1))


def test_interleaving_order_and_wrong_direction_rejected():
    codec = CellCodec(cell_bytes=128, channel='CT')
    parts = codec.fragment(CellKind.REQUEST, 1, b'x' * 200)
    receiver = Reassembler(codec, expected_kind=CellKind.REQUEST)
    assert receiver.accept(parts[0]) is None
    with pytest.raises(FrameError): receiver.accept(parts[2])
    with pytest.raises(FrameError): receiver.accept(codec.fragment(CellKind.REQUEST, 2, b'y')[0])
    with pytest.raises(FrameError): receiver.accept(replace(parts[1], kind=CellKind.RESPONSE))


def test_bounds_and_nonzero_padding_rejected():
    codec = CellCodec(cell_bytes=128, channel='CT', max_message_bytes=100)
    with pytest.raises(FrameError): codec.fragment(CellKind.REQUEST, 1, b'x' * 101)
    cell = codec.encode(0, codec.fragment(CellKind.REQUEST, 1, b'a')[0])
    with pytest.raises(FrameError): codec.decode(cell[:-1] + b'X', expected_sequence=0)
    with pytest.raises(FrameError): codec.decode(cell[:-1], expected_sequence=0)
    receiver = Reassembler(codec, expected_kind=CellKind.REQUEST, max_messages=1)
    receiver.accept(codec.fragment(CellKind.REQUEST, 1, b'a')[0])
    with pytest.raises(FrameError): receiver.accept(codec.fragment(CellKind.REQUEST, 2, b'a')[0])
