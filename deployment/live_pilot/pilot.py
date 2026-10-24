"""Real controller instrumentation pilot; no LLM/MCP or shaping claim."""
import base64,hashlib,http.server,json,os,random,re,selectors,signal,ssl,subprocess,sys,threading,time,urllib.request,uuid
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import mininet.net as mn_module
from mininet.net import Mininet
from mininet.node import RemoteController,OVSSwitch
from mininet.topo import Topo
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.tshark import TsharkTlsParser
mn_module.fixLimits=lambda:None
OUT=Path('/artifacts');OUT.mkdir(exist_ok=True)
for folder in ('captures','traces','manifests'):(OUT/folder).mkdir(exist_ok=True)
AUTH='Basic '+base64.b64encode(b'onos:rocks').decode()

class Proxy(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):self.forward()
 def do_POST(self):self.forward()
 def do_DELETE(self):self.forward()
 def forward(self):
  if self.headers.get('Authorization')!=AUTH:
   self.send_error(401);return
  if self.server.server_port==8444 and self.command!='GET':
   self.send_error(405);return
  if not re.fullmatch(r'/onos/v1/(devices|hosts|links|flows)(/[A-Za-z0-9:%._-]+)*',self.path):
   self.send_error(404);return
  length=int(self.headers.get('Content-Length','0'))
  if not 0<=length<=65536:self.send_error(413);return
  data=self.rfile.read(length) if length else None
  headers={'Authorization':AUTH,'Accept':'application/json'}
  if data is not None:headers['Content-Type']='application/json'
  req=urllib.request.Request('http://127.0.0.1:8181'+self.path,data=data,method=self.command,headers=headers)
  try:
   with urllib.request.urlopen(req,timeout=10) as response:code=response.status;body=response.read(8*1024*1024)
  except urllib.error.HTTPError as e:code=e.code;body=e.read()
  self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.send_header('Connection','close');self.end_headers();self.wfile.write(body)

subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2','-subj','/CN=Intent Leakage pilot','-addext','subjectAltName=IP:127.0.0.1','-keyout','/tmp/pilot.key','-out','/tmp/pilot.crt'],check=True,capture_output=True)
(OUT/'tls_certificate.pem').write_bytes(Path('/tmp/pilot.crt').read_bytes())
key=Ed25519PrivateKey.generate()
(OUT/'signing_public_key.pem').write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
servers=[]
for port in (8443,8444):
 server=http.server.ThreadingHTTPServer(('127.0.0.1',port),Proxy)
 context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=context.maximum_version=ssl.TLSVersion.TLSv1_2
 context.load_cert_chain('/tmp/pilot.crt','/tmp/pilot.key');server.socket=context.wrap_socket(server.socket,server_side=True)
 threading.Thread(target=server.serve_forever,daemon=True).start();servers.append(server)

def client(port):return OnosRestAdapter(f'https://127.0.0.1:{port}',username='onos',password='rocks',ca_file='/tmp/pilot.crt',allow_external=True)
ct,ca=client(8443),client(8444)
topo=Topo();s1=topo.addSwitch('s1',dpid='0000000000000001',protocols='OpenFlow13');s2=topo.addSwitch('s2',dpid='0000000000000002',protocols='OpenFlow13');topo.addLink(s1,s2)
for i in range(8):topo.addLink(topo.addHost('h'+str(i+1)),s1 if i<4 else s2)
net=Mininet(topo=topo,controller=None,switch=OVSSwitch,autoSetMacs=True)
net.addController('c0',controller=RemoteController,ip='127.0.0.1',port=6653)
seed=20261020;order=[(c,q,r) for c in ('I01','I02','I15') for q in (1,2,4) for r in range(2)];random.Random(seed).shuffle(order)
protocol={'schema_version':1,'purpose':'controller_instrumentation_pilot','paper_results_claimed':False,'seed':seed,'order':order,'transport':'HTTP/1.1 over TLS1.2 via local TLS terminators','session_policy':'new TCP/TLS connection per REST call','channels':{'CT':8443,'CA':8444},'capture_interface':'lo','capture_mode':'libpcap immediate mode','capture_tail_s':0.2,'retransmission_policy':'exclude tshark retransmission frames','tls_length':'visible record payload plus 5 byte header','timestamp':'completion frame capture time','split':'instrumentation_only','shaping':'none','agent':'deterministic operation driver; no LLM or MCP','reset':'owned rules removed and dataplane checked; same topology/controller retained within pilot','topology':{'name':'pilot_two_switch_eight_host','switches':2,'hosts':8,'links':[['s1','s2']]+[[f'h{i+1}','s1' if i<4 else 's2'] for i in range(8)]}}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
manifest=(OUT/'manifests/runs.jsonl').open('w');capture=None

def ping(host,expect):
 target=net.get('h8').IP();result=host.cmd(f'ping -c 1 -W 1 {target}');ok='1 received' in result
 if ok!=expect:raise RuntimeError(f'Dataplane assertion {host.name} expected={expect}: {result}')

def wait_owned(expected):
 for _ in range(50):
  flows=ct.assurance_read({'kind':'flows'})['flows'];owned=[f for f in flows if f.get('appId')=='org.intentleakage.pilot' and f.get('state') not in ('REMOVED',)]
  if (len(owned)==expected and all(f.get('state')=='ADDED' for f in owned)) or (expected==0 and not owned):return
  time.sleep(.2)
 raise RuntimeError('Owned flow state failed to settle')

try:
 net.start()
 if not net.waitConnected(timeout=45):raise RuntimeError('OpenFlow timeout')
 time.sleep(8);net.pingAll(timeout='1');time.sleep(3)
 if net.pingAll(timeout='1')!=0:raise RuntimeError('Baseline topology is not connected')
 for cls,q,rep in order:
  run_id=uuid.uuid4().hex;cap=OUT/'captures'/f'{run_id}.pcap';eventsfile=OUT/'traces'/f'{run_id}.jsonl'
  record={'run_id':run_id,'class_id':cls,'scope_q':q,'repetition':rep,'split':'instrumentation_only','synthetic':False,'paper_results_claimed':False,'success':False,'topology_id':protocol['topology']['name'],'condition':'no_shaping','targets':[f'00:00:00:00:00:{i+1:02X}/None' for i in range(q)]}
  t0=time.monotonic();runerror=None
  try:
   ct.cleanup({'kind':'remove_owned_rules'});wait_owned(0)
   for i in range(q):ping(net.get('h'+str(i+1)),True)
   ping(net.get('h7'),True)
   if cls in ('I02','I15'):
    ct.apply_operation({'kind':'quarantine_hosts','host_ids':record['targets']});wait_owned(2*q)
   for i in range(q):ping(net.get('h'+str(i+1)),cls=='I01')
   capture=subprocess.Popen(['tcpdump','--immediate-mode','-U','-s','0','-i','lo','-w',str(cap),'tcp port 8443 or tcp port 8444'],stderr=subprocess.PIPE,stdout=subprocess.DEVNULL)
   ready=b'';selector=selectors.DefaultSelector();selector.register(capture.stderr,selectors.EVENT_READ)
   deadline=time.monotonic()+5
   while b'listening on' not in ready:
    if time.monotonic()>deadline or capture.poll() is not None:raise RuntimeError('Capture readiness failed')
    if selector.select(.2):ready+=os.read(capture.stderr.fileno(),4096)
   selector.close();begin=time.monotonic()
   if cls=='I01':ct.apply_operation({'kind':'quarantine_hosts','host_ids':record['targets']})
   elif cls=='I02':ct.cleanup({'kind':'remove_owned_rules'})
   operation_end=time.monotonic()
   # Polling for installation is control-plane tooling, and is part of captured CT.
   if cls!='I15':wait_owned(0 if cls=='I02' else 2*q)
   if cls=='I15':
    record['audit_policy_ids']=[{'device_id':d,'flow_id':f} for d,f in sorted(ct._owned)[::2]]
    audited=[ca.assurance_read({'kind':'flow','device_id':d,'flow_id':f}) for d,f in sorted(ct._owned)[::2]]
    if len(audited)!=q or any(not a['flows'] for a in audited):raise RuntimeError('Audit failed')
   else:
    flows=ca.assurance_read({'kind':'flows'})['flows']
    count=sum(f.get('appId')=='org.intentleakage.pilot' and f.get('state')=='ADDED' for f in flows)
    if count!=(0 if cls=='I02' else 2*q):raise RuntimeError('Assurance failed')
   record['operation_latency_s']=operation_end-begin;record['through_assurance_s']=time.monotonic()-begin
   time.sleep(.2) # Fixed drain window; excluded from operation/assurance latency.
   capture.send_signal(signal.SIGINT);_,tail=capture.communicate(timeout=5);capture=None
   (OUT/'captures'/f'{run_id}.tcpdump.txt').write_bytes(ready+tail)
   record['capture_sha256']=hashlib.sha256(cap.read_bytes()).hexdigest();record['capture_path']=str(cap.relative_to(OUT))
   for i in range(q):ping(net.get('h'+str(i+1)),cls=='I02')
   ping(net.get('h7'),True)
   # Fully held-out dataplane checks precede trace parsing and never enter features.
   mapping={'trace_id':run_id,'endpoints':[{'client_ip':'127.0.0.1','server_ip':'127.0.0.1','server_port':p,'channel':ch} for p,ch in ((8443,'CT'),(8444,'CA'))]}
   events=TsharkTlsParser().parse(cap,flow_manifest=mapping)
   if not events or not any(e['channel']=='CA' for e in events):raise RuntimeError('No assurance TLS records parsed')
   eventsfile.write_text(''.join(json.dumps(e)+'\n' for e in events));record['trace_path']=str(eventsfile.relative_to(OUT));record['trace_sha256']=hashlib.sha256(eventsfile.read_bytes()).hexdigest();record['tls_records']=len(events)
   record['success']=True
  except Exception as e:record['error_type']=type(e).__name__;record['error']=str(e);runerror=e
  finally:
   if capture:
    capture.send_signal(signal.SIGINT);capture.communicate(timeout=5);capture=None
   try:
    ct.cleanup({'kind':'remove_owned_rules'});wait_owned(0)
    for i in range(q):ping(net.get('h'+str(i+1)),True)
    record['cleanup_verified']=True
   except Exception as e:record['cleanup_verified']=False;record['cleanup_error']=str(e);runerror=e
   record['total_wall_s']=time.monotonic()-t0
   canonical=json.dumps(record,sort_keys=True,separators=(',',':')).encode();envelope={'record':record,'signature_ed25519':base64.b64encode(key.sign(canonical)).decode()}
   manifest.write(json.dumps(envelope)+'\n');manifest.flush();print(json.dumps({k:record.get(k) for k in ('run_id','class_id','scope_q','success','tls_records','error')}),flush=True)
  if runerror:raise runerror
finally:
 manifest.close()
 if capture:capture.send_signal(signal.SIGINT);capture.communicate(timeout=5)
 net.stop()
 for s in servers:s.shutdown();s.server_close()
