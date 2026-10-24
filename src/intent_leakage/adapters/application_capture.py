"""Private, bounded HTTP-body capture. A completed run is required for fitting.

Elapsed time includes the unshaped HTTP exchange; it is NOT server processing
latency and must not silently become a causal shaping service-time estimate.
"""
from __future__ import annotations
import hashlib,json,math,os,re
from pathlib import Path
from .application_observation import ApplicationExchange


def _write_new(path,data):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def _hash(data):return hashlib.sha256(data).hexdigest()


class ApplicationCapture:
    def __init__(self,root,run_id,*,max_bytes=64*1024*1024,max_exchanges=4096):
        if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',run_id):raise ValueError('invalid run ID')
        if max_bytes<=0 or max_exchanges<=0:raise ValueError('invalid capture bounds')
        self.root=Path(root)/run_id;self.root.mkdir(parents=True,mode=0o700,exist_ok=False)
        self.run_id=run_id;self.phase='setup';self.rows=[];self.total=0;self.pending=[]
        self.max_bytes=max_bytes;self.max_exchanges=max_exchanges;self.closed=False;self.errors=[]
        self._checkpoint(False)
    def _checkpoint(self,complete):
        data={'schema_version':1,'run_id':self.run_id,'complete':complete,'exchanges':self.rows,
              'body_bytes':self.total,'errors':self.errors,'unshaped':True,
              'timing_semantics':'whole HTTP exchange, NOT isolated server processing',
              'eligible_for_paper_test':False}
        temp=self.root/'manifest.next';_write_new(temp,json.dumps(data,indent=2).encode())
        os.replace(temp,self.root/'manifest.json')
        fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def observer(self,channel):
        if channel not in ('CT','CA'):raise ValueError('unknown channel')
        def capture(exchange):
            try:self._append(channel,exchange)
            except Exception as exc:self.errors.append(type(exc).__name__);raise
        return capture
    def _append(self,channel,e):
        if self.closed:raise ValueError('capture already closed')
        if not isinstance(e,ApplicationExchange) or not math.isfinite(e.elapsed_s) or e.elapsed_s<0:raise ValueError('invalid exchange')
        if self.phase not in ('setup','preparation','active','cleanup'):raise ValueError('invalid phase')
        n=len(e.request_body)+len(e.response_body)
        if len(self.rows)>=self.max_exchanges or self.total+n>self.max_bytes:raise ValueError('capture quota exceeded')
        idx=len(self.rows);row={**e.metadata(),'index':idx,'channel':channel,'phase':self.phase}
        for key,body in [('request',e.request_body),('response',e.response_body)]:
            name=f'{idx:06d}.{key}.bin';self.pending.append((name,body));row[key+'_file']=name
        self.rows.append(row);self.total+=n
        # No disk I/O in the observed request path. Bounded memory is committed
        # after capture/cleanup; process loss leaves the initial incomplete manifest.
    def finish(self,*,run_success,observer_errors=()):
        if self.closed:raise ValueError('capture already closed')
        self.errors.extend(observer_errors);self.closed=True
        try:
            for name,body in self.pending:_write_new(self.root/name,body)
        except Exception as exc:
            self.errors.append(type(exc).__name__)
        finally:self.pending.clear()
        complete=bool(run_success and self.rows and not self.errors)
        self._checkpoint(complete)
        return {'complete':complete,'exchanges':len(self.rows),'body_bytes':self.total,
                'manifest_sha256':_hash((self.root/'manifest.json').read_bytes())}


def verify_capture(root,*,expected_sha256=None):
    root=Path(root);raw=(root/'manifest.json').read_bytes()
    if expected_sha256 and _hash(raw)!=expected_sha256:raise ValueError('manifest checksum mismatch')
    m=json.loads(raw)
    if not m['complete'] or m['errors'] or not m['unshaped'] or not m['exchanges']:raise ValueError('incomplete application capture')
    total=0;seen=set()
    for idx,row in enumerate(m['exchanges']):
        if row['index']!=idx or row['channel'] not in ('CT','CA') or row['phase'] not in ('setup','preparation','active','cleanup'):raise ValueError('invalid row')
        for key in ('request','response'):
            name=row[key+'_file']
            if name!=f'{idx:06d}.{key}.bin' or name in seen:raise ValueError('unexpected body path')
            seen.add(name);path=root/name
            if path.is_symlink():raise ValueError('symlink body rejected')
            body=path.read_bytes();total+=len(body)
            if len(body)!=row[key+'_body_bytes'] or _hash(body)!=row[key+'_body_sha256']:raise ValueError('body checksum mismatch')
    if total!=m['body_bytes']:raise ValueError('body byte total mismatch')
    return m
