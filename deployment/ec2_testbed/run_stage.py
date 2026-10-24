"""Bounded isolated smoke; always remove this project's containers afterwards."""
import json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
ONOS='il-onos-20261020';MN='il-mininet-20261020';IMAGE='intent-leakage/mininet:20261020'
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
status={'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'infrastructure_smoke_only':True,'success':False}
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
 docker('pull','onosproject/onos:2.7.0',timeout=360)
 docker('build','--tag',IMAGE,'.',timeout=600)
 for image in ('onosproject/onos:2.7.0',IMAGE):
  data=json.loads(docker('image','inspect',image).stdout)
  (OUT/('onos_image.json' if image.startswith('onos') else 'mininet_image.json')).write_text(json.dumps([{k:d.get(k) for k in ('Id','RepoDigests','Created','Architecture','Os','Size')} for d in data],indent=2))
 for count in (1,2):
  docker('run','-d','--name',ONOS,'--label','project=intent-leakage','--network','none','--cpus','2','--memory','6g','--pids-limit','2048','--restart','no','-e','ONOS_APPS=drivers,openflow,proxyarp,hostprovider,lldpprovider,fwd','onosproject/onos:2.7.0')
  docker('run','-d','--name',MN,'--label','project=intent-leakage','--network','container:'+ONOS,'--privileged','--cpus','2','--memory','4g','--pids-limit','1024','--restart','no','-v','/lib/modules:/lib/modules:ro','-v',str(OUT)+':/artifacts','-v',str(ROOT/'smoke.py')+':/smoke.py:ro',IMAGE)
  docker('exec',MN,'/usr/share/openvswitch/scripts/ovs-ctl','--system-id=random','start')
  for i in range(60):
   p=docker('exec',MN,'curl','-fsS','--max-time','3','-u','onos:rocks','http://127.0.0.1:8181/onos/v1/devices',timeout=8,check=False)
   if p.returncode==0:
    (OUT/'onos_rest_ready.json').write_text(p.stdout);break
   time.sleep(3)
  else:raise RuntimeError('ONOS REST not ready')
  for name,args in {'package_versions.txt':['dpkg-query','-W','mininet','openvswitch-switch','tcpdump','python3'],'ovs_version.txt':['ovs-vsctl','--version']}.items():
   (OUT/name).write_text(docker('exec',MN,*args).stdout)
  docker('exec',MN,'python3','/smoke.py',str(count),timeout=180)
  (OUT/f'container_stats_{count}.txt').write_text(docker('stats','--no-stream',ONOS,MN).stdout)
  for name in (ONOS,MN):
   p=docker('logs',name,check=False)
   (OUT/f'{name}_{count}.log').write_text(p.stdout+p.stderr)
  for name in (MN,ONOS):docker('rm','-f',name)
 all_results=[]
 for count in (1,2):all_results.extend(json.loads((OUT/f'smoke_results_{count}.json').read_text()))
 # The containing directory is owned by ubuntu; an older container may own the file.
 (OUT/'smoke_results.json').unlink(missing_ok=True)
 (OUT/'smoke_results.json').write_text(json.dumps(all_results,indent=2))
 status['success']=True
except (Exception, KeyboardInterrupt) as e:
 status['error']=str(e);traceback.print_exc()
finally:
 for name in (ONOS,MN):
  p=docker('logs',name,check=False)
  (OUT/(name+'.log')).write_text(p.stdout+p.stderr)
 for name in (MN,ONOS):docker('rm','-f',name,check=False)
 after=snapshot();(OUT/'host_after.json').write_text(json.dumps(after,indent=2))
 def stable(value):
  if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in ('valid_life_time','preferred_life_time')}
  if isinstance(value,list):return [stable(v) for v in value]
  return value
 status['host_network_unchanged']={k:stable(before[k])==stable(after[k]) for k in ('route','rules','qdisc','ens5')}
 status['comparison_note']='DHCP address lifetimes age naturally and are excluded; raw snapshots preserved.'
 status['module_loaded_before']=before['ovs_module_loaded'];status['module_loaded_after']=after['ovs_module_loaded']
 status['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (OUT/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if not status['success']:raise SystemExit(1)
