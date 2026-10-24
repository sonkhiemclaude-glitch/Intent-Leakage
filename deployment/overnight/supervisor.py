"""Durable bounded queue. Checkpoint only verified batches; retain every attempt."""
import json,os,signal,subprocess,time,traceback,fcntl
from pathlib import Path
from common import atomic_json,busy_reasons,cleanup
ROOT=Path(__file__).resolve().parent
STOP=False
signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))

def persist(state):
 state['heartbeat_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());atomic_json(ROOT/'checkpoint.json',state)

def terminate(child):
 if child.poll() is None:
  child.send_signal(signal.SIGINT)
  try:child.wait(timeout=45)
  except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
 cleanup()

def main():
 lock=(ROOT.parent/'.intent-leakage-testbed.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 config=json.loads((ROOT/'queue.json').read_text())
 state=json.loads((ROOT/'checkpoint.json').read_text()) if (ROOT/'checkpoint.json').exists() else {'completed':{},'attempts':{},'failures':{},'state':'created','started_epoch':time.time(),'events':[]}
 cleanup() # These names are reserved exclusively for this service.
 tasks=config['tasks'];deadline=config['deadline_epoch'];child=None
 try:
  for task in tasks:
   task_id=task['id']
   if task_id in state['completed']:continue
   quiet_since=None
   while not STOP and time.time()<deadline:
    if state['failures'].get(task_id,0)>=config.get('max_failures_per_task',3):
     state.update(state='failed',error='Execution failure limit reached in '+task_id);persist(state);return 0
    reasons=busy_reasons(os.getpid())
    if reasons:
     quiet_since=None
     state.update(state='waiting_for_idle',active_task=task_id,busy_reasons=reasons);persist(state);time.sleep(10);continue
    if quiet_since is None:quiet_since=time.monotonic()
    if time.monotonic()-quiet_since<30:
     state.update(state='waiting_for_idle_quiet_window',active_task=task_id,busy_reasons=[]);persist(state);time.sleep(5);continue
    attempt=state['attempts'].get(task_id,0)+1;state['attempts'][task_id]=attempt
    work=ROOT/'batches'/task_id/f'attempt_{attempt:03d}';work.mkdir(parents=True)
    state.update(state='running',active_task=task_id,active_attempt=str(work.relative_to(ROOT)),busy_reasons=[]);persist(state)
    env=dict(os.environ,IL_BATCH_ROOT=str(work),IL_SOURCE_ROOT=str(ROOT/'worker'),IL_PILOT_TOPOLOGY=task['topology'],IL_PILOT_SEED=str(task['seed']),IL_SUPERVISOR_PID=str(os.getpid()))
    if config.get('protocol_revision') in ('bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):
     env.update(IL_NATIVE_PROVIDER_PYTHON=config['native_provider_python'],IL_NATIVE_CREDENTIAL_FILE=str(ROOT/'.bedrock-token'))
    with (work/'worker.log').open('w') as log:
     child=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'worker/run_stage.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
     batch_deadline=time.time()+config.get('batch_timeout_s',1800);interrupted=None
     while child.poll() is None:
      time.sleep(5)
      reasons=busy_reasons(os.getpid())
      if STOP or time.time()>=deadline or time.time()>=batch_deadline or reasons:
       interrupted='service_stopping' if STOP else 'deadline' if time.time()>=deadline else 'batch_timeout' if time.time()>=batch_deadline else 'other_project_active'
       state['events'].append({'task':task_id,'attempt':attempt,'event':interrupted,'time':time.time(),'reasons':reasons})
       terminate(child);break
      manifest=work/'artifacts/manifests/runs.jsonl'
      try:state['active_observed_records']=len(manifest.read_text().splitlines())
      except OSError:state['active_observed_records']=0
      persist(state)
     code=child.wait();child=None
    if interrupted:
     quiet_since=None
     if interrupted=='batch_timeout':state['failures'][task_id]=state['failures'].get(task_id,0)+1
     persist(state)
     if STOP or time.time()>=deadline:break
     time.sleep(10);continue
    if code!=0:
     quiet_since=None
     state['failures'][task_id]=state['failures'].get(task_id,0)+1
     state['events'].append({'task':task_id,'attempt':attempt,'event':'worker_failed','exit_code':code,'time':time.time()});persist(state);time.sleep(10);continue
    # Verify signatures/checksums using the already-tested image, with no network.
    remaining=deadline-time.time()
    if remaining<=0:
     state.update(state='deadline_reached');persist(state);return 0
    try:
     verify=subprocess.run(['sudo','-n','docker','run','--pull','never','--rm','--name','il-night-verify','--label','intent-leakage.owner=overnight','--network','none','--cpus','1','--memory','512m','-v',str(work/'artifacts')+':/artifacts:ro','-v',str(ROOT/'verify_batch.py')+':/verify.py:ro','-v',str(ROOT/'worker/src')+':/source:ro','-e','PYTHONPATH=/source',config['image'],'python3','/verify.py','/artifacts'],capture_output=True,text=True,timeout=min(120,remaining))
    except subprocess.TimeoutExpired:
     state['events'].append({'task':task_id,'attempt':attempt,'event':'verification_timeout','time':time.time()})
     state.update(state='deadline_reached' if time.time()>=deadline else 'failed',error='Verification timeout in '+task_id)
     persist(state);return 0 # finally removes the verifier if Docker CLI timed out.
    (work/'verification.log').write_text(verify.stdout+verify.stderr)
    if verify.returncode:
     quiet_since=None
     state['failures'][task_id]=state['failures'].get(task_id,0)+1;state['events'].append({'task':task_id,'attempt':attempt,'event':'verification_failed','time':time.time()});persist(state);continue
    for file in work.rglob('*'):
     if file.is_file():
      with file.open('rb') as durable:os.fsync(durable.fileno())
    result=json.loads(verify.stdout);result['attempt_path']=str(work.relative_to(ROOT));state['completed'][task_id]=result
    state['verified_runs']=sum(r['verified_runs'] for r in state['completed'].values());persist(state);break
   if STOP or time.time()>=deadline:break
  state['state']='complete' if len(state['completed'])==len(tasks) else 'stopped' if STOP else 'deadline_reached';persist(state)
 finally:
  if child:terminate(child)
  cleanup()
if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();cleanup();raise
