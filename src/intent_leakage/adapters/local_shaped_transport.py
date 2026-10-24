"""Local-only fixed-cadence, mutually authenticated TLS transport prototype.

AF_UNIX by default; explicitly enabled IPv4 loopback is co-located only.
Both peers share one local monotonic clock. This is NOT an ONOS
adapter, a provider proxy, or a claim about constant TLS/IP packet traces.
Handlers must use RequestContext.commit for local mutations. Arbitrary remote
side effects are unsupported: cooperative cancellation cannot undo those.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import ipaddress
import json
import math
import socket
import ssl
import threading
import time
from typing import Callable
from ..shaping.framing import CellCodec, CellKind, FrameError, Reassembler


class TransportError(RuntimeError): pass
class EnvelopeOverflow(TransportError):
    def __init__(self, message, *, code='envelope_overflow'):
        super().__init__(message); self.code = code
class TransportCancelled(TransportError): pass


@dataclass(frozen=True)
class LocalTransportProfile:
    channel: str = 'CT'
    cell_bytes: int = 512
    rounds: int = 64
    period_s: float = .02
    max_message_bytes: int = 65536
    max_pending: int = 8
    max_messages: int = 256
    max_lateness_s: float = .05
    io_timeout_s: float = .2

    def __post_init__(self):
        CellCodec(cell_bytes=self.cell_bytes, channel=self.channel, max_message_bytes=self.max_message_bytes)
        for name, value, limit in [('rounds', self.rounds, 10000), ('max_pending', self.max_pending, 32), ('max_messages', self.max_messages, 4096)]:
            if type(value) is not int or not 1 <= value <= limit: raise ValueError('invalid ' + name)
        for name in ['period_s', 'max_lateness_s', 'io_timeout_s']:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0: raise ValueError('invalid ' + name)
        if self.rounds * self.period_s > 120: raise ValueError('local prototype limited to two minutes')

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class RequestContext:
    def __init__(self, *, deadline: float | None = None):
        if deadline is not None and not math.isfinite(deadline): raise ValueError('invalid commit deadline')
        self._lock = threading.RLock(); self._active = True; self._deadline = deadline

    def check_active(self):
        """Check admission/result publication; does not cancel remote I/O."""
        with self._lock:
            if not self._active or (self._deadline is not None and time.monotonic() >= self._deadline):
                raise TransportCancelled('transaction window closed')

    def commit(self, action: Callable):
        """A short atomic LOCAL action; never blocking I/O or remote mutation."""
        with self._lock:
            if not self._active or (self._deadline is not None and time.monotonic() >= self._deadline):
                raise TransportCancelled('transaction window closed')
            return action()

    @property
    def deadline(self):
        return self._deadline

    def cancel(self):
        with self._lock: self._active = False


class LocalShapedEndpoint:
    def __init__(self, sock: ssl.SSLSocket, *, role: str, profile: LocalTransportProfile,
                 start_at: float, session_id: bytes, handler: Callable | None = None,
                 cleanup: Callable | None = None, allow_loopback_tcp: bool = False):
        if not isinstance(sock, ssl.SSLSocket):
            raise TransportError('authenticated TLS socket required')
        if sock.family == socket.AF_UNIX:
            self.transport_kind='unix'
        elif sock.family == socket.AF_INET and allow_loopback_tcp is True:
            if not all(ipaddress.ip_address(addr[0]).is_loopback for addr in (sock.getsockname(),sock.getpeername())):
                raise TransportError('TCP transport restricted to co-located IPv4 loopback')
            self.transport_kind='ipv4_loopback'
        else:
            raise TransportError('AF_UNIX or explicitly enabled loopback TLS required')
        if sock.version() != 'TLSv1.3' or sock.context.verify_mode != ssl.CERT_REQUIRED or not sock.getpeercert():
            raise TransportError('TLS 1.3 with verified peer certificates is required')
        if role == 'agent' and (not sock.context.check_hostname or sock.server_hostname != 'localhost'):
            raise TransportError('local agent must verify the localhost server identity')
        if role not in ('agent', 'bridge') or (role == 'bridge') != (handler is not None):
            raise ValueError('bridge needs a handler; agent must not have one')
        if type(session_id) is not bytes or len(session_id) != 16 or not math.isfinite(start_at):
            raise ValueError('invalid local session binding')
        self.sock = sock; self.role = role; self.profile = profile; self.start_at = start_at; self.session_id = session_id
        self.handler = handler; self.cleanup = cleanup
        self.context = RequestContext(deadline=start_at + profile.rounds * profile.period_s)
        self.codec = CellCodec(cell_bytes=profile.cell_bytes, channel=profile.channel, max_message_bytes=profile.max_message_bytes)
        self.receiver = Reassembler(self.codec, expected_kind=CellKind.RESPONSE if role == 'agent' else CellKind.REQUEST, max_messages=profile.max_messages)
        self._lock = threading.RLock(); self._stop = threading.Event(); self._done = threading.Event()
        self._outgoing = deque(); self._outgoing_messages = 0; self._pending: dict[int, Future] = {}
        self._next_id = 1; self._server_inflight = 0; self._cells_reserved = 0; self._closed = False; self._started = False
        self._error: Exception | None = None; self._cleanup_calls = 0
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='il-local-handler') if handler else None
        self._thread = threading.Thread(target=self._run, name='il-local-' + role, daemon=True)
        self.metrics = {'role': role, 'transport_kind':self.transport_kind, 'application_cells_sent': [], 'complete_messages_received': 0,
                        'dummy_cells_sent': 0, 'handler_tasks_submitted': 0, 'handler_timing': [], 'receive_timing': [], 'phase': 'created', 'planned_termination': start_at + profile.rounds * profile.period_s}

    def start(self):
        with self._lock:
            if self._started or self._closed: raise TransportError('endpoint already started or closed')
            self._started = True; self._thread.start()

    def submit(self, payload: bytes) -> Future:
        with self._lock:
            if self.role != 'agent' or self._closed or self._stop.is_set(): raise TransportCancelled('endpoint not accepting requests')
            if time.monotonic() >= self.start_at + self.profile.rounds * self.profile.period_s:
                raise EnvelopeOverflow('request arrived after public termination', code='admission_deadline')
            if len(self._pending) >= self.profile.max_pending or self._next_id > self.profile.max_messages:
                raise EnvelopeOverflow('request queue or message limit exceeded')
            mid = self._next_id
            fragments = self.codec.fragment(CellKind.REQUEST, mid, payload)
            if len(fragments) > self.profile.rounds:
                raise EnvelopeOverflow('request cannot fit even an empty envelope')
            self._enqueue(fragments)
            self._next_id += 1; future = Future(); self._pending[mid] = future
            return future

    def _enqueue(self, fragments):
        if self._outgoing_messages >= self.profile.max_pending:
            raise EnvelopeOverflow('bounded transmit queue exhausted')
        if len(self._outgoing) + len(fragments) > self.profile.rounds - self._cells_reserved:
            raise EnvelopeOverflow('remaining public slots cannot carry queued message', code='slot_admission')
        self._outgoing.extend(fragments); self._outgoing_messages += 1

    def _preamble(self):
        # Separate fixed-length setup before the common application epoch.
        binding = hashlib.sha256((self.profile.sha256 + self.start_at.hex() + self.transport_kind).encode()).digest()
        role = b'A' if self.role == 'agent' else b'B'
        expected = b'B' if self.role == 'agent' else b'A'
        preamble = b'ILP1' + role + self.session_id + binding + bytes(75)
        self._send_exact(preamble); peer = self._read_exact(128)
        if peer != b'ILP1' + expected + self.session_id + binding + bytes(75):
            raise TransportError('peer profile, session or local epoch mismatch')

    def _send_exact(self, data):
        deadline = time.monotonic() + self.profile.io_timeout_s
        sent = 0
        while sent < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('whole-cell send deadline expired')
            self.sock.settimeout(remaining)
            count = self.sock.send(data[sent:])
            if not count: raise TransportError('peer stopped receiving')
            sent += count

    def _read_exact(self, size):
        chunks = bytearray(); deadline = time.monotonic() + self.profile.io_timeout_s
        while len(chunks) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0: raise TimeoutError('whole-cell receive deadline expired')
            self.sock.settimeout(remaining)
            part = self.sock.recv(size - len(chunks))
            if not part: raise TransportError('peer connection ended before envelope completion')
            chunks.extend(part)
        return bytes(chunks)

    def _wait_slot(self, when):
        delay = when - time.monotonic()
        if delay > 0 and self._stop.wait(delay): raise TransportCancelled('stop requested')
        if self._stop.is_set(): raise TransportCancelled('stop requested')
        lateness = time.monotonic() - when
        if lateness > self.profile.max_lateness_s:
            self.metrics['deadline_miss'] = {'planned_at': when, 'lateness_s': lateness, 'limit_s': self.profile.max_lateness_s}
            raise EnvelopeOverflow('public cadence deadline missed', code='cadence_deadline')

    def _send(self, index):
        when = self.start_at + index * self.profile.period_s
        if self.role == 'bridge': when += self.profile.period_s / 2
        entered_wait = time.monotonic()
        self._wait_slot(when)
        woke = time.monotonic()
        with self._lock:
            locked = time.monotonic()
            fragment = self._outgoing.popleft() if self._outgoing else None
            if fragment and fragment.index == fragment.count - 1: self._outgoing_messages -= 1
            encoding_started = time.monotonic()
            cell = self.codec.encode(index, fragment)
            encoding_finished = time.monotonic()
            self._cells_reserved += 1
        actual = time.monotonic(); self._send_exact(cell)
        io_finished = time.monotonic()
        self.metrics['application_cells_sent'].append({'sequence': index, 'planned_at': when, 'write_started_at': actual,
                                                       'lateness_s': actual - when, 'application_bytes': len(cell),
                                                       'entered_wait_at': entered_wait, 'woke_at': woke,
                                                       'late_before_wait_s': max(0., entered_wait-when),
                                                       'wait_wakeup_delay_s': max(0.,woke-max(when,entered_wait)),
                                                       'lock_wait_s': locked-woke, 'encode_s': encoding_finished-encoding_started,
                                                       'write_io_s': io_finished-actual})
        if fragment is None: self.metrics['dummy_cells_sent'] += 1

    def _receive(self, index):
        read_started = time.monotonic()
        raw = self._read_exact(self.profile.cell_bytes)
        read_finished = time.monotonic()
        fragment = self.codec.decode(raw, expected_sequence=index)
        self.metrics['receive_timing'].append({'sequence':index,'io_started_at':read_started,'io_finished_at':read_finished,'io_s':read_finished-read_started})
        message = self.receiver.accept(fragment)
        if message is None: return
        mid, payload = message; self.metrics['complete_messages_received'] += 1
        if self.role == 'agent':
            with self._lock:
                future = self._pending.pop(mid, None)
                if future is None: raise FrameError('response has no outstanding request')
                # User cancellation cannot cancel a mutation already on its way.
                if not future.cancelled(): future.set_result(payload)
        else:
            with self._lock:
                if self._server_inflight >= self.profile.max_pending:
                    raise EnvelopeOverflow('bounded handler queue exhausted')
                self._server_inflight += 1; self.metrics['handler_tasks_submitted'] += 1
                queued = time.monotonic()
                task = self._worker.submit(self._invoke_handler, payload, queued)
                task.add_done_callback(lambda completed: self._handler_finished(mid, completed))

    def _invoke_handler(self, payload, queued):
        started = time.monotonic()
        try:
            return self.handler(payload, self.context)
        finally:
            finished = time.monotonic()
            with self._lock:
                if not self._closed:
                    self.metrics['handler_timing'].append({'queued_at':queued,'started_at':started,'finished_at':finished,'queue_delay_s':started-queued,'execution_s':finished-started})

    def _handler_finished(self, mid, task):
        try:
            payload = task.result()
            fragments = self.codec.fragment(CellKind.RESPONSE, mid, payload)
            with self._lock:
                if self._closed or self._stop.is_set(): return
                self._enqueue(fragments); self._server_inflight -= 1
        except Exception as exc:
            self._abort(exc)

    def _abort(self, exc):
        if isinstance(exc, OSError):
            wrapped = TransportError('authenticated transport I/O failed')
            wrapped.__cause__ = exc
            exc = wrapped
        with self._lock:
            if self._closed: return
            if self._error is None: self._error = exc
            self._stop.set()
        self.context.cancel()
        try: self.sock.shutdown(socket.SHUT_RDWR)
        except OSError: pass

    def _run(self):
        try:
            self.metrics['phase'] = 'setup'
            self.sock.settimeout(self.profile.io_timeout_s); self._preamble()
            if time.monotonic() >= self.start_at: raise EnvelopeOverflow('setup missed public start', code='setup_deadline')
            # Do not spend a whole-cell I/O timeout waiting BEFORE the public
            # epoch. Setup lead and active-frame timeout are separate budgets.
            self.metrics['phase'] = 'waiting_public_start'
            self._wait_slot(self.start_at)
            for index in range(self.profile.rounds):
                self.metrics['round'] = index
                order = (self._send, self._receive) if self.role == 'agent' else (self._receive, self._send)
                for step in order:
                    self.metrics['phase'] = 'send' if step == self._send else 'receive'
                    step(index)
            self.metrics['phase'] = 'termination'
            self._wait_slot(self.metrics['planned_termination'])
            with self._lock:
                if self._pending or self._outgoing or self._server_inflight or self.receiver.incomplete:
                    raise EnvelopeOverflow('unfinished request, response or dependency at common termination', code='unfinished_at_termination')
            self.metrics['phase'] = 'tls_close'
            # An authenticated close_notify exchange, bounded by io_timeout_s,
            # avoids treating abrupt TLS EOF as successful common termination.
            self.sock.settimeout(self.profile.io_timeout_s)
            raw = self.sock.unwrap()
            raw.close()
            self.metrics['tls_close_notify_completed'] = True
            self.metrics['phase'] = 'complete'
        except Exception as exc:
            self._abort(exc)
        finally:
            self._finish()

    def _finish(self):
        self.context.cancel()
        with self._lock:
            if self._closed: return
            self._closed = True; self._stop.set()
            error = self._error or TransportCancelled('envelope closed')
            for future in self._pending.values():
                if not future.done(): future.set_exception(error)
            self.metrics['unfinished_at_close'] = {'pending_requests': len(self._pending), 'outgoing_fragments': len(self._outgoing), 'handler_inflight': self._server_inflight, 'incomplete_message': self.receiver.incomplete}
            self._pending.clear(); self._outgoing.clear(); self._outgoing_messages = 0
            self.receiver.buffer.clear(); self.receiver.active = None; self.receiver.seen.clear()
        if self._worker: self._worker.shutdown(wait=False, cancel_futures=True)
        try:
            if self.cleanup:
                self._cleanup_calls += 1; self.cleanup()
        except Exception as exc:
            if self._error is None: self._error = exc
        finally:
            self.sock.close(); self.metrics['closed_at'] = time.monotonic()
            self.metrics['cleanup_calls'] = self._cleanup_calls
            self.metrics['error_type'] = type(self._error).__name__ if self._error else None
            self.metrics['failure_code'] = getattr(self._error, 'code', None)
            self.metrics['error_cause_type'] = type(self._error.__cause__).__name__ if self._error and self._error.__cause__ else None
            self.metrics['success'] = self._error is None
            self._done.set()

    def close(self):
        with self._lock:
            if self._closed: return
        self._abort(TransportCancelled('explicit stop'))
        if not self._started: self._finish()

    def wait(self, timeout: float):
        if not self._done.wait(timeout): raise TimeoutError('local transport did not terminate in time')
        return dict(self.metrics)
