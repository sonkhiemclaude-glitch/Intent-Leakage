"""Integrity-checked join of signed pilot logs and attacker-visible TLS metadata.

This produces instrumentation-only features; it does not assign research splits
or allow these pilot rows to stand in for the prospective confirmatory grid.
"""
from __future__ import annotations
import csv
import base64
import hashlib
import json
from pathlib import Path
from typing import Any
from .io import load_trace_events_jsonl
from ..features.extraction import extract_channel_features

EVENT_FIELDS={'trace_id','timestamp_ns','timestamp_s','channel','direction','length_bytes',
              'flow_id','retransmission','transport_unit','segmentation_mode'}

def _contained(root: Path, relative: str) -> Path:
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Artifact path is outside the dataset or missing')
    return path


def build_pilot_features(root: str | Path, *, public_key_sha256: str, output: str | Path) -> dict[str, Any]:
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError('Feature output already exists')
    public=(root/'signing_public_key.pem').read_bytes()
    if hashlib.sha256(public).hexdigest()!=public_key_sha256:
        raise ValueError('Signing key does not match authenticated key hash')
    key=load_pem_public_key(public)
    if not isinstance(key,Ed25519PublicKey):raise ValueError('Expected Ed25519 public key')
    rows=[];seen=set();feature_names=None
    for line in (root/'manifests/runs.jsonl').read_text().splitlines():
        envelope=json.loads(line);record=envelope['record']
        canonical=json.dumps(record,sort_keys=True,separators=(',',':')).encode()
        key.verify(base64.b64decode(envelope['signature_ed25519'],validate=True),canonical)
        run_id=record['run_id']
        if run_id in seen:raise ValueError('Duplicate run identity')
        seen.add(run_id)
        if not record['success'] or not record['cleanup_verified']:
            raise ValueError('Failed runs require explicit failure analysis, not silent removal')
        if record['split']!='instrumentation_only' or record['paper_results_claimed'] is not False:
            raise ValueError('This converter accepts instrumentation-only pilot records')
        for prefix in ('capture','trace'):
            path=_contained(root,record[prefix+'_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record[prefix+'_sha256']:
                raise ValueError(f'{prefix} checksum mismatch')
        path=_contained(root,record['trace_path'])
        raw=[json.loads(line) for line in path.read_text().splitlines()]
        if not raw or len(raw)!=record['tls_records']:raise ValueError('Trace record count mismatch')
        for event in raw:
            if set(event)-EVENT_FIELDS:raise ValueError('Unexpected fields in attacker-visible events')
            if event['trace_id']!=run_id:raise ValueError('Trace/manifest identity mismatch')
            if event['channel'] not in ('CT','CA') or event['transport_unit']!='tls_record' or event['segmentation_mode']!='endpoint_separable':
                raise ValueError('Incompatible pilot observation semantics')
        dataset=load_trace_events_jsonl(path)
        events=dataset.traces[run_id]
        features={}
        for channel in ('CT','CA'):
            features.update(extract_channel_features(events,channel=channel,prefix=channel.lower()+'__'))
        features.update(extract_channel_features(events,prefix='ctca__'))
        if feature_names is None:feature_names=list(features)
        if list(features)!=feature_names:raise ValueError('Feature schema drift')
        rows.append({'run_id':run_id,'class_id':record['class_id'],'scope_q':record['scope_q'],
                     'repetition':record['repetition'],'split':'instrumentation_only',**features})
    if not rows:raise ValueError('No signed runs')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'rows':len(rows),'feature_columns':feature_names,'label_columns':['run_id','class_id','scope_q','repetition','split'],
            'public_key_sha256':public_key_sha256,'features_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'paper_results_claimed':False,'purpose':'instrumentation_only'}
