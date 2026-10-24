"""Disposable mTLS/AF_UNIX lab. No AWS, SSH, real controller, or IP connection."""
from __future__ import annotations
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import secrets
import socket
import ssl
import tempfile
import threading
import time
from ..adapters.local_shaped_transport import LocalShapedEndpoint, LocalTransportProfile


def _certificates(folder):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
    now = datetime.now(timezone.utc)
    root_key = ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Disposable Intent Leakage Local Lab CA')])
    root = (x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name).public_key(root_key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True).sign(root_key, hashes.SHA256()))
    (folder / 'ca.pem').write_bytes(root.public_bytes(serialization.Encoding.PEM))
    for role, usage in [('server', ExtendedKeyUsageOID.SERVER_AUTH), ('client', ExtendedKeyUsageOID.CLIENT_AUTH)]:
        key = ec.generate_private_key(ec.SECP256R1())
        cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'localhost')]))
                .issuer_name(root.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(hours=1))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False)
                .add_extension(x509.ExtendedKeyUsage([usage]), critical=False).sign(root_key, hashes.SHA256()))
        path = folder / (role + '.key'); path.touch(mode=0o600)
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        (folder / (role + '.pem')).write_bytes(cert.public_bytes(serialization.Encoding.PEM))


@contextmanager
def authenticated_socketpair(*, trust_client=True, loopback_tcp=False):
    """Fresh CA/keys per pair; files removed on success and every failure path."""
    with tempfile.TemporaryDirectory(prefix='il-local-tls-') as temporary:
        folder = Path(temporary); _certificates(folder)
        server = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        for context, role in [(server, 'server'), (client, 'client')]:
            context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_3
            context.verify_mode = ssl.CERT_REQUIRED
            context.options |= ssl.OP_NO_COMPRESSION
            context.load_cert_chain(str(folder / (role + '.pem')), str(folder / (role + '.key')))
            if role == 'client' or trust_client: context.load_verify_locations(str(folder / 'ca.pem'))
        server.num_tickets = 0
        if loopback_tcp:
            listener=socket.socket(socket.AF_INET,socket.SOCK_STREAM);left=None
            try:
                listener.settimeout(2);listener.bind(('127.0.0.1',0));listener.listen(1)
                left=socket.socket(socket.AF_INET,socket.SOCK_STREAM);left.settimeout(2)
                left.connect(listener.getsockname());right,_=listener.accept()
                for stream in (left,right):stream.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
            except BaseException:
                if left is not None:left.close()
                raise
            finally:listener.close()
        else:
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        left.settimeout(2); right.settimeout(2); a = b = None
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(server.wrap_socket, right, server_side=True)
                try:
                    a = client.wrap_socket(left, server_hostname='localhost')
                    b = future.result(timeout=3)
                except Exception:
                    if a: a.close()
                    left.close(); right.close()
                    try: future.result(timeout=3)
                    except Exception: pass
                    raise
            yield a, b
        finally:
            for stream in [a, b, left, right]:
                if stream is not None: stream.close()


class LocalPolicyLab:
    """A small independent local-state fixture; deliberately not an ONOS shim."""
    def __init__(self, *, state=None, delay_s=0.):
        self.state = state if state is not None else {'foreign': 'keep', 'owned_policy': 'original'}
        self._snapshot = {k: self.state[k] for k in ['owned_policy'] if k in self.state}
        self._lock = threading.RLock(); self.calls = []; self.cleanup_count = 0; self.delay_s = delay_s

    def handle(self, payload, context):
        request = json.loads(payload)
        if not isinstance(request, dict) or request.get('op') not in ('read', 'write') or request.get('resource') != 'owned_policy':
            raise ValueError('operation or resource outside the local allowlist')
        op = request['op']
        if set(request) != ({'op', 'resource', 'value'} if op == 'write' else {'op', 'resource'}):
            raise ValueError('unexpected application fields')
        if op == 'write' and (not isinstance(request['value'], str) or len(request['value']) > 60000):
            raise ValueError('invalid local policy value')
        if self.delay_s: time.sleep(self.delay_s)
        def operation():
            with self._lock:
                if op == 'write': self.state['owned_policy'] = request['value']
                self.calls.append({'op': op, 'committed_at': time.monotonic()})
                return json.dumps({'value': self.state.get('owned_policy')}).encode()
        return context.commit(operation)

    def cleanup(self):
        with self._lock:
            self.cleanup_count += 1
            if 'owned_policy' in self._snapshot: self.state['owned_policy'] = self._snapshot['owned_policy']
            else: self.state.pop('owned_policy', None)


def make_endpoints(sockets, *, profile, lab, lead_s=.1, bridge_profile=None, allow_loopback_tcp=False):
    epoch = time.monotonic() + lead_s; sid = secrets.token_bytes(16)
    a = LocalShapedEndpoint(sockets[0], role='agent', profile=profile, start_at=epoch, session_id=sid, allow_loopback_tcp=allow_loopback_tcp)
    b = LocalShapedEndpoint(sockets[1], role='bridge', profile=bridge_profile or profile, start_at=epoch,
                           session_id=sid, handler=lab.handle, cleanup=lab.cleanup,allow_loopback_tcp=allow_loopback_tcp)
    return a, b


class LocalQualificationError(RuntimeError):
    def __init__(self, metrics):
        super().__init__('local transport failed')
        self.metrics = metrics


def successful_case(*, cell_bytes=512, rounds=128, period_s=.005, value='x' * 1000, channel='CT', lab=None, lead_s=.1):
    profile = LocalTransportProfile(channel=channel, cell_bytes=cell_bytes, rounds=rounds, period_s=period_s,
                                    max_lateness_s=.1, io_timeout_s=.5)
    lab = lab if lab is not None else LocalPolicyLab(); before = dict(lab.state)
    with authenticated_socketpair() as pair:
        agent, bridge = make_endpoints(pair, profile=profile, lab=lab, lead_s=lead_s)
        tls_version = pair[0].version()
        write = json.dumps({'op': 'write', 'resource': 'owned_policy', 'value': value}).encode()
        try:
            future = agent.submit(write); agent.start(); bridge.start()
            actual = json.loads(future.result(timeout=3)); acknowledged_at = time.monotonic()
            read = agent.submit(json.dumps({'op': 'read', 'resource': 'owned_policy'}).encode())
            readback = json.loads(read.result(timeout=3))
            assert actual['value'] == readback['value'] == value
            metrics = [agent.wait(3), bridge.wait(3)]
            if not all(m['success'] for m in metrics): raise LocalQualificationError(metrics)
        except BaseException as exc:
            agent.close(); bridge.close()
            stopped = [agent.wait(2), bridge.wait(2)]
            if isinstance(exc, Exception): exc.metrics = stopped
            raise
        assert len(lab.calls) == 2 and lab.calls[1]['committed_at'] >= acknowledged_at
        assert lab.state == before and lab.cleanup_count == 1
        assert all(len(m['application_cells_sent']) == rounds for m in metrics)
        return {'case': 'successful_fragmented_write_then_read', 'profile': profile.__dict__, 'profile_sha256': profile.sha256,
                'tls': tls_version, 'peer_certificates_verified': True,
                'application_payload_bytes': len(write), 'metrics': metrics,
                'dummy_triggered_handler': False, 'handler_calls': len(lab.calls),
                'causal_read_after_acknowledgment': True, 'owned_state_restored': True,
                'foreign_state_preserved': lab.state['foreign'] == before['foreign'],
                'claims': {'live_controller_validated': False, 'tls_record_constancy_validated': False,
                           'wire_overhead_measured': False, 'distributional_secrecy_proven': False}}
