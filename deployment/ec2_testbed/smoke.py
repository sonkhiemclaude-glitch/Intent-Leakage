"""Infrastructure smoke only: these topologies are not frozen paper TA/TB."""
import json, subprocess, time, urllib.request, base64, sys
from pathlib import Path
import mininet.net as mininet_module
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.topo import Topo

# Container resource limits are explicit; do not let Mininet tune global sysctls.
mininet_module.fixLimits = lambda: None
OUT=Path('/artifacts'); OUT.mkdir(exist_ok=True)
def api(path):
 req=urllib.request.Request('http://127.0.0.1:8181/onos/v1/'+path)
 req.add_header('Authorization','Basic '+base64.b64encode(b'onos:rocks').decode())
 with urllib.request.urlopen(req,timeout=10) as r:return json.load(r)
results=[]
for count in (int(sys.argv[1]),):
 topo=Topo()
 switches=[topo.addSwitch('s'+str(i+1),dpid=f'{i+1:016x}',protocols='OpenFlow13') for i in range(count)]
 for i in range(4):
  h=topo.addHost('h'+str(i+1)); topo.addLink(h,switches[i%count])
 if count==2:topo.addLink(*switches)
 net=Mininet(topo=topo,controller=None,switch=OVSSwitch,autoSetMacs=True)
 net.addController('c0',controller=RemoteController,ip='127.0.0.1',port=6653)
 capture=None
 try:
  capture=subprocess.Popen(['tcpdump','-U','-i','lo','-w',str(OUT/f'openflow_{count}switch.pcap'),'tcp port 6653'],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  net.start()
  if not net.waitConnected(timeout=45):raise RuntimeError('OpenFlow connection timeout')
  time.sleep(8)
  loss=net.pingAll(timeout='2')
  # First discovery may require ARP/host learning; preserve it and check steady state.
  time.sleep(5); steady=net.pingAll(timeout='2')
  devices=api('devices');hosts=api('hosts');links=api('links')
  connected=[d for d in devices.get('devices',[]) if d.get('available')]
  row={'switch_count':count,'host_count':4,'initial_loss_percent':loss,'steady_loss_percent':steady,'devices':devices,'hosts':hosts,'links':links,'available_switches':len(connected)}
  row['openflow_protocol']='OpenFlow13'
  row['passed']=steady==0 and len(connected)==count and len(hosts.get('hosts',[]))==4 and len(links.get('links',[]))==2*(count-1)
  results.append(row)
  (OUT/f'smoke_results_{count}.json').write_text(json.dumps(results,indent=2))
  if not row['passed']:raise RuntimeError('Topology smoke assertions failed')
 finally:
  net.stop()
  if capture:
   capture.terminate()
   try:capture.communicate(timeout=5)
   except subprocess.TimeoutExpired:capture.kill();capture.communicate()
print(json.dumps({'infrastructure_smoke_only':True,'topologies':results}))
