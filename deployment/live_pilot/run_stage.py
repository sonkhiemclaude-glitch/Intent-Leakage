"""Bounded isolated smoke; always remove this project's containers afterwards."""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
ONOS='il-pilot-onos';MN='il-pilot-mininet';IMAGE='intent-leakage/pilot:20261020'
def run(args,timeout=60,check=True):
 print('+',' '.join(args),flush=True)
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout,cwd=ROOT)
 if p.stdout:print(p.stdout[-10000:],flush=True)
 if p.stderr:print(p.stderr[-5000:],flush=True)
 if check and p.returncode:raise RuntimeError(f'command failed rc={p.returncode}: {args[:3]}')
 return p

def docker(*args,timeout=60,check=True):return run(['sudo','-n','docker',*args],timeout,check)
def snapshot():
 d={}
 for k,a in {'route':['ip','-j','route','show'],'rules':['ip','-j','rule','show'],'qdisc':['tc','-j','qdisc','show','dev','ens5'],'ens5':['ip','-j','address','show','dev','ens5']}.items():
  d[k]=json.loads(run(a).stdout)
 d['ovs_module_loaded']=Path('/sys/module/openvswitch').exists()
 return d
status={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'controller_instrumentation_pilot':True,'success':False}
before=snapshot();(OUT/'host_before.json').write_text(json.dumps(before,indent=2))
# Abort on another project becoming active between the audit and deployment.
existing=docker('ps','--format','{{.Names}}').stdout.strip()
if existing:raise RuntimeError('Active containers detected; refusing concurrent testbed')
others=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  name=(p/'comm').read_text().strip();uid=(p/'status').read_text().split('Uid:')[1].split()[0]
  if uid=='1000' and name.startswith(('python','iperf','tcpdump','netshaper')):others.append({'pid':p.name,'name':name})
 except (FileNotFoundError,PermissionError,IndexError):pass
if others:raise RuntimeError('Other project processes active: '+json.dumps(others))
try:
 docker('build','--tag',IMAGE,'.',timeout=600)
 validation=docker('run','--rm','--network','none','-e','PYTHONPATH=/source','-v',str(ROOT/'src')+':/source:ro','-v',str(ROOT/'validate_reassembly.py')+':/validate.py:ro',IMAGE,'python3','/validate.py')
 (OUT/'reassembly_validation.json').write_text(validation.stdout)
 onos_image='onosproject/onos@sha256:bc844aaafd64e6b3834c7043bad83fa1eecc6afb6984e9b4887c12e1a307a7c2'
 docker('run','-d','--name',ONOS,'--label','project=intent-leakage','--network','none','--cpus','2','--memory','6g','--pids-limit','2048','--restart','no','-e','ONOS_APPS=drivers,openflow,proxyarp,hostprovider,lldpprovider,fwd',onos_image)
 docker('run','-d','--name',MN,'--label','project=intent-leakage','--network','container:'+ONOS,'--privileged','--cpus','2','--memory','4g','--pids-limit','1024','--restart','no','-e','PYTHONPATH=/source','-v','/lib/modules:/lib/modules:ro','-v',str(OUT)+':/artifacts','-v',str(ROOT/'pilot.py')+':/pilot.py:ro','-v',str(ROOT/'src')+':/source:ro',IMAGE)
 docker('exec',MN,'/usr/share/openvswitch/scripts/ovs-ctl','--system-id=random','start')
 for i in range(60):
  p=docker('exec',MN,'python3','-c',"import urllib.request;req=urllib.request.Request('http://127.0.0.1:8181/onos/v1/devices',headers={'Authorization':'Basic b25vczpyb2Nrcw=='});print(urllib.request.urlopen(req,timeout=3).status)",timeout=8,check=False)
  if p.returncode==0:break
  time.sleep(3)
 else:raise RuntimeError('ONOS REST not ready')
 (OUT/'parser_version.txt').write_text(docker('exec',MN,'tshark','--version').stdout)
 docker('exec',MN,'python3','-u','/pilot.py',timeout=1200)
 status['success']=True
except (Exception,KeyboardInterrupt) as e:
 status['error']=str(e);traceback.print_exc()
finally:
 for name in (ONOS,MN):
  p=docker('logs',name,check=False);(OUT/(name+'.log')).write_text(p.stdout+p.stderr)
 for name in (MN,ONOS):docker('rm','-f',name,check=False)
 after=snapshot();(OUT/'host_after.json').write_text(json.dumps(after,indent=2))
 def stable(v):
  if isinstance(v,dict):return {k:stable(x) for k,x in v.items() if k not in ('valid_life_time','preferred_life_time')}
  if isinstance(v,list):return [stable(x) for x in v]
  return v
 status['host_network_unchanged']={k:stable(before[k])==stable(after[k]) for k in ('route','rules','qdisc','ens5')}
 status['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (OUT/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 import tarfile
 with tarfile.open(ROOT/'results.tar.gz','w:gz') as archive:archive.add(OUT,arcname='artifacts')
if not status['success']:raise SystemExit(1)
