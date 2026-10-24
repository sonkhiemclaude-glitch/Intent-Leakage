"""Bounded fixed-size application cells for the local authenticated tunnel.

These cells are plaintext inputs to TLS, NOT TLS records or IP packets.
Dummy cells terminate at the tunnel and never invoke an application handler.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
import struct


class FrameError(ValueError):
    pass


class CellKind(IntEnum):
    DUMMY = 0
    REQUEST = 1
    RESPONSE = 2


# magic, version, kind, channel, flags, cell sequence, message ID,
# fragment index, fragment count, payload length
HEADER = struct.Struct('!4sBBBBQQIII')
MAGIC = b'ILF1'
CHANNELS = {'CT': 1, 'CA': 2}


@dataclass(frozen=True)
class Fragment:
    kind: CellKind
    message_id: int
    index: int
    count: int
    payload: bytes


class CellCodec:
    def __init__(self, *, cell_bytes: int, channel: str, max_message_bytes: int = 65536):
        if type(cell_bytes) is not int or not 64 <= cell_bytes <= 16384:
            raise FrameError('cell_bytes must be an integer in [64, 16384]')
        if channel not in CHANNELS:
            raise FrameError('only CT and CA local channels are supported')
        if type(max_message_bytes) is not int or not 1 <= max_message_bytes <= 1048576:
            raise FrameError('invalid message size bound')
        self.cell_bytes = cell_bytes; self.channel = channel
        self.max_message_bytes = max_message_bytes
        self.capacity = cell_bytes - HEADER.size
        self.max_fragments = math.ceil(max_message_bytes / self.capacity)

    def fragment(self, kind: CellKind, message_id: int, payload: bytes) -> tuple[Fragment, ...]:
        if kind not in (CellKind.REQUEST, CellKind.RESPONSE):
            raise FrameError('real messages must be requests or responses')
        if type(message_id) is not int or not 1 <= message_id < 2**64:
            raise FrameError('message ID outside allowed range')
        if type(payload) is not bytes or len(payload) > self.max_message_bytes:
            raise FrameError('payload exceeds bound or is not bytes')
        count = max(1, math.ceil(len(payload) / self.capacity))
        return tuple(Fragment(kind, message_id, i, count, payload[i*self.capacity:(i+1)*self.capacity]) for i in range(count))

    def encode(self, sequence: int, fragment: Fragment | None = None) -> bytes:
        if type(sequence) is not int or not 0 <= sequence < 2**64:
            raise FrameError('invalid cell sequence')
        f = fragment or Fragment(CellKind.DUMMY, 0, 0, 0, b'')
        payload = f.payload
        if len(payload) > self.capacity:
            raise FrameError('fragment exceeds cell capacity')
        cell = HEADER.pack(MAGIC, 1, int(f.kind), CHANNELS[self.channel], 0,
                           sequence, f.message_id, f.index, f.count, len(payload))
        # Decode our own headers too: sender errors must fail before release.
        cell += payload + bytes(self.capacity - len(payload))
        self.decode(cell, expected_sequence=sequence)
        return cell

    def decode(self, cell: bytes, *, expected_sequence: int) -> Fragment | None:
        if len(cell) != self.cell_bytes:
            raise FrameError('wrong cell length')
        magic, version, kind, channel, flags, seq, mid, index, count, length = HEADER.unpack_from(cell)
        if magic != MAGIC or version != 1 or flags or channel != CHANNELS[self.channel]:
            raise FrameError('invalid protocol header')
        if seq != expected_sequence:
            raise FrameError('replayed or out-of-order cell')
        try: kind = CellKind(kind)
        except ValueError as exc: raise FrameError('unknown cell kind') from exc
        if length > self.capacity or any(cell[HEADER.size + length:]):
            raise FrameError('invalid length or nonzero padding')
        if kind == CellKind.DUMMY:
            if any((mid, index, count, length)):
                raise FrameError('dummy must not carry application metadata')
            return None
        if not mid or not 1 <= count <= self.max_fragments or index >= count:
            raise FrameError('invalid fragment metadata')
        if index < count - 1 and length != self.capacity:
            raise FrameError('nonfinal fragments must fill their cells')
        if (count - 1) * self.capacity + length > self.max_message_bytes and index == count - 1:
            raise FrameError('reassembled message would exceed bound')
        if count > 1 and index == count - 1 and not length:
            raise FrameError('empty final fragment')
        return Fragment(kind, mid, index, count, cell[HEADER.size:HEADER.size + length])


class Reassembler:
    """Strict FIFO messages with a bounded replay history per TLS session."""
    def __init__(self, codec: CellCodec, *, expected_kind: CellKind, max_messages: int = 256):
        if expected_kind not in (CellKind.REQUEST, CellKind.RESPONSE):
            raise FrameError('invalid receiving role')
        if type(max_messages) is not int or not 1 <= max_messages <= 10000:
            raise FrameError('invalid replay-history bound')
        self.codec = codec; self.expected_kind = expected_kind; self.max_messages = max_messages
        self.seen: set[int] = set(); self.active: Fragment | None = None
        self.buffer = bytearray(); self.next_index = 0

    def accept(self, fragment: Fragment | None) -> tuple[int, bytes] | None:
        if fragment is None:
            return None
        if fragment.kind != self.expected_kind or fragment.message_id in self.seen:
            raise FrameError('wrong direction or duplicate application message')
        if self.active is None:
            if fragment.index != 0 or len(self.seen) >= self.max_messages:
                raise FrameError('missing initial fragment or replay-history limit')
            self.active = fragment; self.next_index = 0
        if (fragment.message_id, fragment.count, fragment.index) != (self.active.message_id, self.active.count, self.next_index):
            raise FrameError('interleaved or out-of-order message fragments')
        self.buffer.extend(fragment.payload); self.next_index += 1
        if len(self.buffer) > self.codec.max_message_bytes:
            raise FrameError('message exceeds bound')
        if self.next_index == fragment.count:
            result = (fragment.message_id, bytes(self.buffer)); self.seen.add(fragment.message_id)
            self.active = None; self.buffer.clear(); self.next_index = 0
            return result
        return None

    @property
    def incomplete(self) -> bool:
        return self.active is not None
