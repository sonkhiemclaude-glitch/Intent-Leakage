"""Visible TLS record metadata using Wireshark TCP reassembly, never decryption.

Record timestamps are the completion frame's capture timestamp. Length includes
its five-byte TLS header. TCP retransmission frames are excluded. TLS 1.3 outer
application-data records cannot distinguish encrypted handshake from application
payload; session policy must be recorded separately. PCAPNG is supported by tshark.
"""
from __future__ import annotations

import csv
import io
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pcap import PcapParserAdapter

FIELDS = ('frame.number', 'frame.time_epoch', 'ip.src', 'ip.dst', 'tcp.srcport',
          'tcp.dstport', 'tcp.stream', 'tcp.analysis.retransmission',
          'tcp.analysis.spurious_retransmission', 'tls.record.content_type',
          'tls.record.length', 'tls.record.opaque_type')


class TsharkTlsParser(PcapParserAdapter):
    def __init__(self, executable: str = 'tshark', *, timeout_s: float = 120) -> None:
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.executable, self.timeout_s = executable, timeout_s

    @staticmethod
    def decode_rows(text: str, *, flow_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
        trace_id = flow_manifest['trace_id']
        endpoints = flow_manifest['endpoints']
        if not isinstance(trace_id, str) or not trace_id.strip() or not endpoints:
            raise ValueError('A trace_id and endpoint mapping are required')
        seen = set()
        for ep in endpoints:
            if ep['channel'] not in ('CL', 'CT', 'CA'):
                raise ValueError('Unknown channel')
            if not isinstance(ep['server_port'], int) or not 1 <= ep['server_port'] <= 65535:
                raise ValueError('Invalid server port')
            key = (ep['client_ip'], ep['server_ip'], ep['server_port'])
            if key in seen:
                raise ValueError('Ambiguous endpoint mapping')
            seen.add(key)
        events = []
        for row in csv.reader(io.StringIO(text), delimiter='\t', quoting=csv.QUOTE_NONE):
            if not row:
                continue
            # Legacy synthetic fixtures have12 fields; live v2 adds TLS1.3
            # opaque outer types without changing the first12 field positions.
            if len(row)==len(FIELDS)-1:row.append('')
            if len(row) != len(FIELDS):
                raise ValueError('Unexpected tshark field schema')
            frame, stamp, src, dst, sport, dport, stream, retrans, spurious, types, lengths, opaque_types = row
            if not lengths or retrans or spurious:
                continue
            matches = []
            for ep in endpoints:
                if src == ep['client_ip'] and dst == ep['server_ip'] and dport == str(ep['server_port']):
                    matches.append((ep, 1))
                elif src == ep['server_ip'] and dst == ep['client_ip'] and sport == str(ep['server_port']):
                    matches.append((ep, -1))
            if not matches:
                continue
            if len(matches) != 1:
                raise ValueError('Ambiguous packet direction')
            ep, direction = matches[0]
            record_types=[v for v in types.split(',') if v]
            opaque=[v for v in opaque_types.split(',') if v]
            if any(v!='23' for v in opaque):raise ValueError('Invalid TLS1.3 outer type')
            record_types+=opaque
            record_lengths=lengths.split(',')
            if len(record_types) != len(record_lengths):
                raise ValueError('TLS record fields are not aligned')
            timestamp_ns = int(Decimal(stamp) * 1_000_000_000)
            if timestamp_ns < 0:
                raise ValueError('Invalid timestamp')
            if any(int(kind) not in (20,21,22,23,24) for kind in record_types):raise ValueError('Invalid visible TLS type')
            # Type order is not used as a feature. Keep original LENGTH order;
            # tshark exposes clear and opaque record types in separate arrays.
            for length in record_lengths:
                payload = int(length)
                if not 0 <= payload <= 18432:
                    raise ValueError('Invalid visible TLS record header')
                events.append({'trace_id': trace_id, 'timestamp_ns': timestamp_ns,
                    'channel': ep['channel'], 'direction': direction,
                    'length_bytes': payload + 5, 'flow_id': f'tcp:{stream}',
                    'retransmission': False, 'transport_unit': 'tls_record',
                    'segmentation_mode': 'endpoint_separable'})
        return sorted(events, key=lambda e: e['timestamp_ns'])

    def parse(self, capture_path: str | Path, *, flow_manifest: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        path = Path(capture_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Validate mapping before invoking the external parser.
        self.decode_rows('', flow_manifest=flow_manifest)
        args = [self.executable, '-n', '-2', '-r', str(path),
                '-o', 'tcp.desegment_tcp_streams:TRUE', '-o', 'tls.desegment_ssl_records:TRUE']
        for port in sorted({e['server_port'] for e in flow_manifest['endpoints']}):
            args += ['-d', f'tcp.port=={port},tls']
        args += ['-T', 'fields', '-E', 'occurrence=a', '-E', 'aggregator=,']
        for field in FIELDS:
            args += ['-e', field]
        result = subprocess.run(args, text=True, capture_output=True, check=True, timeout=self.timeout_s)
        return self.decode_rows(result.stdout, flow_manifest=flow_manifest)
