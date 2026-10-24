"""Scheduling guards; never stop or alter another project's resources."""
import json,os,subprocess,shutil,time
from pathlib import Path
OWN_NAMES={'il-night-onos','il-night-mininet','il-night-validate','il-night-verify'}
WORKLOAD_PREFIXES=('python','java','node','iperf','tcpdump','tshark','netshaper','stress','ffmpeg','Rscript')

def command(args,timeout=15):
 return subprocess.run(args,capture_output=True,text=True,timeout=timeout)

def descendants(root_pid,proc_root=Path('/proc')):
 parents={}
 for p in proc_root.iterdir():
  if not p.name.isdigit():continue
  try:parents[int(p.name)]=int((p/'stat').read_text().rsplit(')',1)[1].split()[1])
  except (OSError,ValueError,IndexError):continue
 result={root_pid}
 while True:
  more={p for p,pp in parents.items() if pp in result}
  if more<=result:return result
  result|=more

def classify_containers(text):
 rows=[line.split() for line in text.splitlines() if line.strip()]
 return [r[1] for r in rows if len(r)<3 or r[1] not in OWN_NAMES or r[2]!='overnight'], [r[0] for r in rows if len(r)>=3 and r[1] in OWN_NAMES and r[2]=='overnight']

def busy_reasons(root_pid):
 p=command(['sudo','-n','docker','ps','--format','{{.ID}} {{.Names}} {{.Label "intent-leakage.owner"}}'])
 if p.returncode:return ['docker_unavailable']
 foreign,own_ids=classify_containers(p.stdout);reasons=['foreign_container:'+n for n in foreign]
 owned=descendants(root_pid)
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name) in owned:continue
  try:
   name=(proc/'comm').read_text().strip()
   if not name.startswith(WORKLOAD_PREFIXES):continue
   cg=(proc/'cgroup').read_text()
   if any(cid in cg for cid in own_ids):continue
   reasons.append('other_workload:'+proc.name+':'+name)
  except (OSError,IndexError):continue
 mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
 if mem['MemAvailable']<8*1024*1024:reasons.append('available_ram_below_8GiB')
 if os.getloadavg()[0]>6:reasons.append('load_average_above_6')
 if shutil.disk_usage('/').free<15*1024**3:reasons.append('disk_free_below_15GiB')
 return reasons

def cleanup():
 for name in sorted(OWN_NAMES):
  label=command(['sudo','-n','docker','inspect','--format','{{index .Config.Labels "intent-leakage.owner"}}',name])
  if label.returncode==0 and label.stdout.strip()=='overnight':command(['sudo','-n','docker','rm','-f',name],timeout=20)

def atomic_json(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2))
 with temp.open('rb') as f:os.fsync(f.fileno())
 temp.replace(path)
 fd=os.open(str(path.parent),os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
