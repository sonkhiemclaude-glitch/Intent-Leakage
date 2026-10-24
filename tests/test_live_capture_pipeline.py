import pytest
from intent_leakage.adapters.tshark import TsharkTlsParser
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.base import ExternalExecutionDisabledError

MAP={'trace_id':'opaque','endpoints':[{'client_ip':'127.0.0.1','server_ip':'127.0.0.1','server_port':8443,'channel':'CA'}]}
def row(*,src='50000',dst='8443',types='23',lengths='40',retrans='',stamp='1234.000000001'):
 return '\t'.join(['1',stamp,'127.0.0.1','127.0.0.1',src,dst,'0',retrans,'',types,lengths])+'\n'

def test_coalesced_records_keep_tls_lengths_not_tcp_packet_length():
 events=TsharkTlsParser.decode_rows(row(types='23,23',lengths='40,80'),flow_manifest=MAP)
 assert [e['length_bytes'] for e in events]==[45,85]
 assert all(e['timestamp_ns']==1234000000001 for e in events)
 assert all('class_id' not in e and e['direction']==1 for e in events)

def test_loopback_direction_and_retransmission_dedup():
 events=TsharkTlsParser.decode_rows(row(src='8443',dst='50000')+row(retrans='1'),flow_manifest=MAP)
 assert len(events)==1 and events[0]['direction']==-1

def test_ambiguous_endpoints_and_unaligned_record_arrays_fail_closed():
 with pytest.raises(ValueError,match='Ambiguous'):
  TsharkTlsParser.decode_rows('',flow_manifest={**MAP,'endpoints':MAP['endpoints']*2})
 with pytest.raises(ValueError,match='aligned'):
  TsharkTlsParser.decode_rows(row(types='23,23'),flow_manifest=MAP)

def test_invalid_record_size_and_empty_mapping():
 with pytest.raises(ValueError):TsharkTlsParser.decode_rows(row(lengths='20000'),flow_manifest=MAP)
 with pytest.raises(ValueError):TsharkTlsParser.decode_rows('',flow_manifest={'trace_id':'x','endpoints':[]})

def test_onos_external_gate_precedes_cert_or_network_access():
 with pytest.raises(ExternalExecutionDisabledError):
  OnosRestAdapter('https://127.0.0.1:8443',username='x',password='x',ca_file='missing')

def fake_onos():
 a=object.__new__(OnosRestAdapter);a._owned=set();calls=[]
 def request(method,path,payload=None):
  calls.append((method,path,payload))
  if path=='hosts':return {'hosts':[{'id':'h1','mac':'00:00:00:00:00:01','locations':[{'elementId':'of:0001'}]}]}
  if method=='POST':return {'flows':[{'deviceId':'of:0001','flowId':'100'},{'deviceId':'of:0001','flowId':'101'}]}
  return {}
 a.request=request
 return a,calls

def test_real_drop_payload_and_id_scoped_idempotent_cleanup():
 a,calls=fake_onos();a.apply_operation({'kind':'quarantine_hosts','host_ids':['h1']})
 payload=calls[-1][2];assert len(payload['flows'])==2
 assert {f['selector']['criteria'][0]['type'] for f in payload['flows']}=={'ETH_SRC','ETH_DST'}
 assert all(f['treatment']['instructions']==[] for f in payload['flows'])
 a.cleanup({'kind':'remove_owned_rules'});a.cleanup({'kind':'remove_owned_rules'})
 assert [c[1] for c in calls if c[0]=='DELETE']==['flows/of:0001/100','flows/of:0001/101']

def test_missing_host_or_unregistered_operation_never_mutates():
 a,calls=fake_onos()
 with pytest.raises(ValueError):a.apply_operation({'kind':'quarantine_hosts','host_ids':['unknown']})
 with pytest.raises(ValueError):a.apply_operation({'kind':'delete_slice','host_ids':['h1']})
 assert not any(c[0] in ('POST','DELETE') for c in calls)

def test_signed_join_rejects_tampering_and_labels_in_trace(tmp_path):
 import base64,hashlib,json
 from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
 from cryptography.hazmat.primitives import serialization
 from cryptography.exceptions import InvalidSignature
 from intent_leakage.data.capture_pipeline import build_pilot_features
 key=Ed25519PrivateKey.generate();public=key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo)
 (tmp_path/'signing_public_key.pem').write_bytes(public);(tmp_path/'manifests').mkdir()
 (tmp_path/'capture.pcap').write_bytes(b'authenticated capture fixture')
 events=TsharkTlsParser.decode_rows(row(),flow_manifest=MAP)
 (tmp_path/'trace.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
 r={'run_id':'opaque','class_id':'I01','scope_q':1,'repetition':0,'success':True,'cleanup_verified':True,'split':'instrumentation_only','paper_results_claimed':False,'tls_records':1}
 for prefix,name in [('capture','capture.pcap'),('trace','trace.jsonl')]:
  r[prefix+'_path']=name;r[prefix+'_sha256']=hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
 def save():
  sig=base64.b64encode(key.sign(json.dumps(r,sort_keys=True,separators=(',',':')).encode())).decode()
  (tmp_path/'manifests/runs.jsonl').write_text(json.dumps({'record':r,'signature_ed25519':sig})+'\n')
 save();kw={'public_key_sha256':hashlib.sha256(public).hexdigest(),'output':tmp_path/'features.csv'}
 report=build_pilot_features(tmp_path,**kw);assert report['rows']==1 and len(report['feature_columns'])==57
 (tmp_path/'features.csv').unlink()
 r['class_id']='I02' # A label edit without resigning must fail.
 (tmp_path/'manifests/runs.jsonl').write_text((tmp_path/'manifests/runs.jsonl').read_text().replace('I01','I02'))
 with pytest.raises(InvalidSignature):build_pilot_features(tmp_path,**kw)
 save();(tmp_path/'capture.pcap').write_bytes(b'changed')
 with pytest.raises(ValueError,match='checksum'):build_pilot_features(tmp_path,**kw)
 (tmp_path/'capture.pcap').write_bytes(b'authenticated capture fixture')
 events[0]['scope_q']=1 # Even a truth field omitted from the older loader denylist must fail.
 (tmp_path/'trace.jsonl').write_text(json.dumps(events[0])+'\n');r['trace_sha256']=hashlib.sha256((tmp_path/'trace.jsonl').read_bytes()).hexdigest();save()
 with pytest.raises(ValueError,match='Unexpected fields'):build_pilot_features(tmp_path,**kw)

def test_capture_preserves_partial_start_and_stop_is_idempotent(tmp_path):
 from intent_leakage.adapters.tcpdump import TcpdumpCapture
 from intent_leakage.orchestration.capture import CaptureSpec
 executable=tmp_path/'fail_capture'
 executable.write_text("#!/usr/bin/env python3\nimport sys,pathlib\npathlib.Path(sys.argv[sys.argv.index('-w')+1]).write_bytes(b'partial capture evidence')\nsys.stderr.write('forced capture startup failure\\n')\nsys.exit(2)\n")
 executable.chmod(0o700)
 output=tmp_path/'capture.pcap'
 cap=TcpdumpCapture(allow_external=True,executable=str(executable),drain_s=0)
 spec=CaptureSpec('lo',(('127.0.0.1','127.0.0.1'),),'tcp port 8443','host capture clock','restart','CT','passive')
 with pytest.raises(RuntimeError,match='before readiness'):cap.start(spec,output)
 first=cap.stop_and_store();second=cap.stop_and_store()
 assert first is second and first.exit_code==2 and first.artifact.byte_count>0


def test_tls13_opaque_and_mixed_handshake_fields_keep_length_order():
 text=row(types='22,20',lengths='122,1,23,62,809,95,69').rstrip('\n')+'\t23,23,23,23,23\n'
 events=TsharkTlsParser.decode_rows(text,flow_manifest=MAP)
 assert [e['length_bytes'] for e in events]==[127,6,28,67,814,100,74]
 opaque_only=row(types='',lengths='4113').rstrip('\n')+'\t23\n'
 assert TsharkTlsParser.decode_rows(opaque_only,flow_manifest=MAP)[0]['length_bytes']==4118


def test_tls13_opaque_field_must_be_outer_application_data_and_aligned():
 import pytest
 for tail in ('22','23,23'):
  text=row(types='',lengths='4113').rstrip('\n')+'\t'+tail+'\n'
  with pytest.raises(ValueError):TsharkTlsParser.decode_rows(text,flow_manifest=MAP)
