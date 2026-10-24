"""Synthetic TCP fixture validates parser behavior; never a research trace."""
import json,struct,socket,tempfile
from pathlib import Path
from intent_leakage.adapters.tshark import TsharkTlsParser

def checksum(data):
 if len(data)%2:data+=b'\0'
 n=sum(struct.unpack('!'+str(len(data)//2)+'H',data));n=(n>>16)+(n&65535);n+=(n>>16)
 return (~n)&65535

def frame(reverse,seq,ack,flags,payload=b''):
 src,dst=('10.77.0.2','10.77.0.1') if reverse else ('10.77.0.1','10.77.0.2')
 sport,dport=(8443,45000) if reverse else (45000,8443)
 a,b=socket.inet_aton(src),socket.inet_aton(dst)
 tcp=struct.pack('!HHIIBBHHH',sport,dport,seq,ack,80,flags,65535,0,0)+payload
 pseudo=a+b+struct.pack('!BBH',0,6,len(tcp));chk=checksum(pseudo+tcp)
 tcp=tcp[:16]+struct.pack('!H',chk)+tcp[18:]
 ip=struct.pack('!BBHHHBBH4s4s',69,0,20+len(tcp),1,0,64,6,0,a,b)
 ip=ip[:10]+struct.pack('!H',checksum(ip))+ip[12:]
 return b'\x00'*12+b'\x08\x00'+ip+tcp

def record(n):return b'\x17\x03\x03'+struct.pack('!H',n)+b'x'*n
r=record(20)
packets=[frame(False,100,0,2),frame(True,500,101,18),frame(False,101,501,16),
 frame(False,101,501,24,r[:7]),frame(False,101,501,24,r[:7]),
 frame(False,108,501,24,r[7:]),frame(True,501,126,24,record(7)+record(11))]
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'fixture.pcap'
 with p.open('wb') as f:
  f.write(struct.pack('<IHHIIII',0xa1b2c3d4,2,4,0,0,65535,1))
  for i,pkt in enumerate(packets):f.write(struct.pack('<IIII',1700000000,i*1000,len(pkt),len(pkt))+pkt)
 events=TsharkTlsParser().parse(p,flow_manifest={'trace_id':'synthetic_parser_fixture','endpoints':[{'client_ip':'10.77.0.1','server_ip':'10.77.0.2','server_port':8443,'channel':'CT'}]})
 expected=[(1,25),(-1,12),(-1,16)]
 actual=[(e['direction'],e['length_bytes']) for e in events]
 if actual!=expected:raise AssertionError({'expected':expected,'actual':actual,'events':events})
 if events[0]['timestamp_ns']!=1700000000005000000:raise AssertionError('Record completion timestamp mismatch')
 print(json.dumps({'synthetic_fixture':True,'passed':True,'tests':['TCP fragmentation','TCP retransmission exclusion','coalesced TLS records','completion timestamps'],'records':len(events)}))
