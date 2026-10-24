"""Journaled pilot-only controller binding with conservative uncertain outcomes.

No connections are opened here. Live use requires an explicitly enabled existing
ONOS adapter. Only the historical I01 drop-rule semantics are supported; this
is NOT the management-preserving I01 or a 15-class implementation.
"""
from __future__ import annotations
import copy,fcntl,hashlib,json,os,re
from pathlib import Path
from contextlib import contextmanager
from .base import require_external_execution
from .mcp_onos_tools import validate_call
APP='org.intentleakage.pilot'


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def identity(flow):return str(flow['deviceId']),str(flow['id'])
def fingerprint(flow):return digest({k:flow.get(k) for k in ['deviceId','id','appId','selector','treatment','priority','isPermanent']})


class OnosPilotFlowBackend:
    def __init__(self,adapter,*,allow_external=False):
        require_external_execution(allow_external);self.adapter=adapter
    def read(self):return self.adapter.assurance_read({'kind':'flows'})['flows']
    def install(self,hosts):return self.adapter.apply_operation({'kind':'quarantine_hosts','host_ids':hosts})['rules']
    def delete(self,device,flow):
        key=(device,flow)
        if key not in self.adapter._owned:raise TransactionError('current ONOS session has no acknowledged ownership; reconcile first')
        result=self.adapter.request('DELETE','flows/'+device+'/'+flow)
        self.adapter._owned.discard(key)
        return result


class FixtureFlowBackend:
    """In-memory API double; never evidence of ONOS/dataplane correctness."""
    def __init__(self):
        self.flows={('of:0000000000000001','1'):{'deviceId':'of:0000000000000001','id':'1','appId':'foreign','selector':{'criteria':[]},'treatment':{'instructions':[]},'priority':1,'isPermanent':True,'state':'ADDED'}}
        self.next_id=2;self.posts=0;self.deletes=[];self.fail_after_install=False
    def read(self):return copy.deepcopy(list(self.flows.values()))
    def install(self,hosts):
        self.posts+=1;ack=[]
        for host in hosts:
            for kind in ['ETH_SRC','ETH_DST']:
                key=('of:0000000000000001',str(self.next_id));self.next_id+=1
                self.flows[key]={'deviceId':key[0],'id':key[1],'appId':APP,'selector':{'criteria':[{'type':kind,'mac':host.split('/')[0]}]},'treatment':{'instructions':[]},'priority':50000,'isPermanent':True,'state':'ADDED'}
                ack.append({'device_id':key[0],'flow_id':key[1]})
        if self.fail_after_install:raise TimeoutError('injected lost acknowledgment')
        return ack
    def delete(self,device,flow):self.deletes.append((device,flow));self.flows.pop((device,flow),None)


class TransactionError(RuntimeError):pass


class OwnedFlowTransaction:
    def __init__(self,backend,journal:Path):
        if not isinstance(backend,(FixtureFlowBackend,OnosPilotFlowBackend)):raise ValueError('unregistered controller binding')
        self.backend=backend;self.path=Path(journal);self.path.parent.mkdir(parents=True,exist_ok=True)
    @contextmanager
    def _lock(self):
        fd=os.open(str(self.path)+'.lock',os.O_CREAT|os.O_RDWR,0o600)
        try:
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise TransactionError('transaction journal busy') from exc
            yield
        finally:os.close(fd)
    def _save(self,state):
        temp=self.path.with_suffix('.tmp');fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,self.path)
        fd=os.open(self.path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def execute(self,transaction_id,hosts):
        validate_call('quarantine_hosts',{'host_ids':hosts})
        hosts=list(hosts)
        if not re.fullmatch('[0-9a-f]{64}',transaction_id):raise ValueError('frozen transaction ID required')
        spec={'id':transaction_id,'hosts':hosts,'kind':'quarantine_pilot_only'}
        with self._lock():
            if self.path.exists():
                old=json.loads(self.path.read_text())
                if old['spec']!=spec:raise TransactionError('journal belongs to a different operation')
                if old['state'] in ('applied','cleaned'):
                    return {**old,'replayed_from_journal':True,'current_state_reverified':False}
                raise TransactionError('uncertain/incomplete journal; never replay a mutation')
            baseline=self.backend.read();base={identity(f):fingerprint(f) for f in baseline}
            state={'spec':spec,'state':'prepared','baseline':[{'identity':list(k),'fingerprint':v} for k,v in base.items()],
                   'owned':[],'controller_verified':False,'independent_dataplane_verified':False,'formal_I01_semantics':False}
            self._save(state);state['state']='mutation_inflight';self._save(state)
            try:
                ack=self.backend.install(list(hosts))
                keys=[(str(v['device_id']),str(v['flow_id'])) for v in ack]
                if not keys or len(set(keys))!=len(keys) or any(k in base for k in keys):raise TransactionError('invalid owned acknowledgment')
                if any(not re.fullmatch('of:[0-9a-f]{16}',k[0]) or not re.fullmatch('-?[0-9]+',k[1]) for k in keys):raise TransactionError('invalid rule identity')
                state['owned']=[{'identity':list(k),'fingerprint':None} for k in keys];self._save(state)
                current={identity(f):f for f in self.backend.read()};macs={h.split('/')[0].lower() for h in hosts};observed=set()
                for record in state['owned']:
                    flow=current.get(tuple(record['identity']))
                    if not flow or flow.get('appId')!=APP or flow.get('state')!='ADDED':raise TransactionError('owned rule not installed')
                    criteria=flow.get('selector',{}).get('criteria',[])
                    if len(criteria)!=1 or criteria[0].get('type') not in ('ETH_SRC','ETH_DST') or criteria[0].get('mac','').lower() not in macs:raise TransactionError('selector outside frozen targets')
                    if flow.get('treatment')!={'instructions':[]} or flow.get('priority')!=50000 or flow.get('isPermanent') is not True:raise TransactionError('unexpected owned rule semantics')
                    observed.add((criteria[0]['mac'].lower(),criteria[0]['type']));record['fingerprint']=fingerprint(flow)
                if observed!={(m,t) for m in macs for t in ['ETH_SRC','ETH_DST']}:raise TransactionError('incomplete quarantine directions')
                if any(k not in current or fingerprint(current[k])!=v for k,v in base.items()):raise TransactionError('baseline changed concurrently')
                state.update(state='applied',controller_verified=True);self._save(state);return state
            except Exception as exc:
                state.update(state='uncertain',error_type=type(exc).__name__);self._save(state);raise
    def rollback(self):
        with self._lock():
            state=json.loads(self.path.read_text())
            if state['state']=='cleaned':return state
            # Unknown acknowledgments must never be swept using appId alone.
            if state['state']!='applied':raise TransactionError('uncertain mutation requires reconciliation; automatic deletion refused')
            try:
                current={identity(f):f for f in self.backend.read()}
                for record in state['owned']:
                    key=tuple(record['identity']);flow=current.get(key)
                    if flow and fingerprint(flow)!=record['fingerprint']:raise TransactionError('owned rule changed; do not delete')
                state['state']='rollback_inflight';self._save(state)
                for record in state['owned']:
                    key=tuple(record['identity'])
                    if key in current:self.backend.delete(*key)
                after={identity(f):f for f in self.backend.read()}
                if any(tuple(v['identity']) in after for v in state['owned']):raise TransactionError('owned removal not independently read back')
                if any(tuple(v['identity']) not in after or fingerprint(after[tuple(v['identity'])])!=v['fingerprint'] for v in state['baseline']):raise TransactionError('baseline preservation unverified')
                state.update(state='cleaned',controller_cleanup_verified=True);self._save(state);return state
            except Exception as exc:
                state.update(state='rollback_uncertain',error_type=type(exc).__name__);self._save(state);raise
