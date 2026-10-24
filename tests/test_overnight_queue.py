import importlib.util
import pytest
from pathlib import Path
spec=importlib.util.spec_from_file_location('night',Path(__file__).resolve().parents[1]/'deployment/overnight/common.py')
night=importlib.util.module_from_spec(spec);spec.loader.exec_module(night)

def test_ownership_needs_exact_name_and_label():
 foreign,owned=night.classify_containers('aaa il-night-onos overnight\nbbb other-project\nccc il-night-mininet wrong-owner\nddd il-night-verify overnight\n')
 assert foreign==['other-project','il-night-mininet'] and owned==['aaa','ddd']

def test_only_owned_labeled_containers_are_removed(monkeypatch):
 from subprocess import CompletedProcess
 calls=[]
 def run(args,timeout=15):
  calls.append(args)
  if 'inspect' in args:return CompletedProcess(args,0,'overnight\n' if args[-1]=='il-night-onos' else 'different-project\n','')
  return CompletedProcess(args,0,'','')
 monkeypatch.setattr(night,'command',run);night.cleanup()
 assert [a[-1] for a in calls if 'rm' in a]==['il-night-onos']

def test_descendant_guard_does_not_exclude_foreign_work(tmp_path):
 for pid,ppid in [(10,1),(11,10),(12,11),(20,1)]:
  p=tmp_path/str(pid);p.mkdir();(p/'stat').write_text(f'{pid} (a name) S {ppid} 0 0')
 assert night.descendants(10,tmp_path)=={10,11,12}

def test_checkpoint_keeps_last_commit_if_replace_fails(tmp_path,monkeypatch):
 import json
 path=tmp_path/'checkpoint.json';night.atomic_json(path,{'completed':['one']})
 original=Path.replace
 def fail(self,target):
  if target==path:raise OSError('injected interruption before commit')
  return original(self,target)
 monkeypatch.setattr(Path,'replace',fail)
 import pytest
 with pytest.raises(OSError):night.atomic_json(path,{'completed':['one','two']})
 assert json.loads(path.read_text())=={'completed':['one']}


def load_supervisor(tmp_path,monkeypatch):
 import json,signal,sys,time
 from types import SimpleNamespace
 previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
 monkeypatch.setitem(sys.modules,'common',night)
 spec=importlib.util.spec_from_file_location('queue_under_test',Path(__file__).resolve().parents[1]/'deployment/overnight/supervisor.py')
 module=importlib.util.module_from_spec(spec)
 try:spec.loader.exec_module(module)
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
 module.ROOT=tmp_path/'queue';module.ROOT.mkdir()
 ticks=[1000.]
 def sleep(seconds):ticks[0]+=seconds
 module.time=SimpleNamespace(time=lambda:ticks[0],monotonic=lambda:ticks[0],sleep=sleep,strftime=time.strftime,gmtime=time.gmtime)
 module.busy_reasons=lambda _:[];module.cleanup=lambda:None
 config={'tasks':[{'id':'batch_004','topology':'two_switch','seed':20261103}],
         'deadline_epoch':2000,'max_failures_per_task':1,'image':'frozen-test-image'}
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 return module


def test_first_worker_failure_stops_without_spending_another_attempt(tmp_path,monkeypatch):
 import json
 module=load_supervisor(tmp_path,monkeypatch);starts=[]
 class FailedWorker:
  def poll(self):return 1
  def wait(self,**kwargs):return 1
 def spawn(*args,**kwargs):starts.append(kwargs['env']['IL_BATCH_ROOT']);return FailedWorker()
 monkeypatch.setattr(module.subprocess,'Popen',spawn)
 assert module.main()==0
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert state['state']=='failed' and state['failures']=={'batch_004':1}
 assert len(starts)==1 and state['completed']=={}


def test_verification_failure_cannot_commit_or_advance_to_next_task(tmp_path,monkeypatch):
 import json
 from subprocess import CompletedProcess
 module=load_supervisor(tmp_path,monkeypatch);starts=[]
 config=json.loads((module.ROOT/'queue.json').read_text())
 config['tasks'].append({'id':'batch_005','topology':'four_ring','seed':20261104})
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 class Worker:
  def poll(self):return 0
  def wait(self,**kwargs):return 0
 def spawn(*args,**kwargs):starts.append(kwargs['env']['IL_BATCH_ROOT']);return Worker()
 monkeypatch.setattr(module.subprocess,'Popen',spawn)
 monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:CompletedProcess(args,1,'','injected signature failure'))
 module.main()
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert state['state']=='failed' and state['completed']=={}
 assert len(starts)==1 and state['attempts']=={'batch_004':1}
 assert state['events'][-1]['event']=='verification_failed'


def test_expired_deadline_never_starts_worker(tmp_path,monkeypatch):
 import json
 module=load_supervisor(tmp_path,monkeypatch)
 config=json.loads((module.ROOT/'queue.json').read_text());config['deadline_epoch']=999
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 def unexpected(*args,**kwargs):raise AssertionError('worker started after deadline')
 monkeypatch.setattr(module.subprocess,'Popen',unexpected)
 module.main()
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert state['state']=='deadline_reached' and not state['attempts']


def test_resume_preserves_committed_batch_and_skips_its_execution(tmp_path,monkeypatch):
 import json
 module=load_supervisor(tmp_path,monkeypatch);starts=[]
 config=json.loads((module.ROOT/'queue.json').read_text())
 config['tasks'].insert(0,{'id':'batch_003','topology':'four_ring','seed':20261102})
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 committed={'verified_runs':18,'manifest_sha256':'previous-immutable-evidence'}
 previous={'completed':{'batch_003':committed},'verified_runs':18,
           'attempts':{'batch_003':1},'failures':{},'events':[],'state':'stopped'}
 (module.ROOT/'checkpoint.json').write_text(json.dumps(previous))
 class FailedWorker:
  def poll(self):return 1
  def wait(self,**kwargs):return 1
 def spawn(*args,**kwargs):starts.append(kwargs['env']['IL_BATCH_ROOT']);return FailedWorker()
 monkeypatch.setattr(module.subprocess,'Popen',spawn)
 module.main()
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert state['completed']=={'batch_003':committed} and state['verified_runs']==18
 assert len(starts)==1 and 'batch_004' in starts[0]


@pytest.mark.parametrize('global_deadline',[False,True])
def test_inflight_hung_worker_is_cancelled_at_deadline(tmp_path,monkeypatch,global_deadline):
 import json,signal
 module=load_supervisor(tmp_path,monkeypatch);starts=[];signals=[]
 config=json.loads((module.ROOT/'queue.json').read_text())
 config.update(batch_timeout_s=100 if global_deadline else 20,
               deadline_epoch=1040 if global_deadline else 2000)
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 class HungWorker:
  code=None
  def poll(self):return self.code
  def send_signal(self,sig):signals.append(sig);self.code=130
  def wait(self,**kwargs):return self.code
 def spawn(*args,**kwargs):starts.append(kwargs['env']['IL_BATCH_ROOT']);return HungWorker()
 monkeypatch.setattr(module.subprocess,'Popen',spawn)
 module.main()
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert len(starts)==1 and signals==[signal.SIGINT] and not state['completed']
 if global_deadline:
  assert state['state']=='deadline_reached' and state['events'][-1]['event']=='deadline'
  assert not state['failures']
 else:
  assert state['state']=='failed' and state['events'][-1]['event']=='batch_timeout'
  assert state['failures']=={'batch_004':1}


def test_verifier_timeout_is_bounded_by_remaining_global_window(tmp_path,monkeypatch):
 import json,subprocess
 module=load_supervisor(tmp_path,monkeypatch)
 config=json.loads((module.ROOT/'queue.json').read_text());config['deadline_epoch']=1040
 (module.ROOT/'queue.json').write_text(json.dumps(config))
 class Worker:
  def poll(self):return 0
  def wait(self,**kwargs):return 0
 monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kwargs:Worker())
 timeouts=[]
 def verify(*args,**kwargs):
  timeouts.append(kwargs['timeout']);module.time.sleep(kwargs['timeout'])
  raise subprocess.TimeoutExpired(args,kwargs['timeout'])
 monkeypatch.setattr(module.subprocess,'run',verify)
 module.main()
 state=json.loads((module.ROOT/'checkpoint.json').read_text())
 assert timeouts==[10] and state['state']=='deadline_reached'
 assert not state['completed'] and state['events'][-1]['event']=='verification_timeout'
