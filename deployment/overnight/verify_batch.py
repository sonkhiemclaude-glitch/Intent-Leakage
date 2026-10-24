"""Verify completed instrumentation evidence locally on EC2, no secret needed."""
import base64,hashlib,json,re,sys
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_public_key
root=Path(sys.argv[1]);status=json.loads((root/'status.json').read_text())
protocol=json.loads((root/'protocol.json').read_text())
assert status['success'] and all(status['host_network_unchanged'].values())
public=(root/'signing_public_key.pem').read_bytes();key=load_pem_public_key(public)
records=[];seen=set();cells={}
for line in (root/'manifests/runs.jsonl').read_text().splitlines():
 envelope=json.loads(line);record=envelope['record'];key.verify(base64.b64decode(envelope['signature_ed25519']),json.dumps(record,sort_keys=True,separators=(',',':')).encode())
 assert record['success'] and record['cleanup_verified'] and record['split']=='instrumentation_only'
 if protocol.get('protocol_revision') in ('application_bodies_convergence_v1','dataplane_convergence_v2','mcp_stdio_convergence_v1','bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):
  assert record['protocol_revision']==protocol['protocol_revision']
  checks=record['dataplane_checks'];q=record['scope_q']
  assert len(checks)==4*q+2
  for phase in ('preflight','prepared','postcondition','cleanup'):
   subset=[c for c in checks if c['phase']==phase]
   hosts={f'h{i+1}' for i in range(q)}
   if phase in ('preflight','postcondition'):hosts.add('h7')
   assert len(subset)==len(hosts) and {c['host'] for c in subset}==hosts
   for check in subset:
    expected=(phase in ('preflight','cleanup') or check['host']=='h7' or (phase=='prepared' and record['class_id']=='I01') or (phase=='postcondition' and record['class_id']=='I02'))
    assert check['expected_reachable']==expected
    assert check['success'] and check['elapsed_s']<=check['budget_s']==8
    assert check['probes'] and check['probes'][-1]['valid_icmp_result']
    assert check['probes'][-1]['reachable']==expected
    if not expected:assert len(check['probes'])==1
 if protocol.get('protocol_revision') in ('mcp_stdio_convergence_v1','bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):
  metadata=record['mcp'];assert metadata['protocol_version']=='2025-06-18'
  assert metadata['server_revision']==protocol['mcp']['server_revision']=='intent-leakage-onos-stdio-v1'
  assert metadata['tool_schema_sha256']==protocol['mcp']['tool_schema_sha256']
  assert re.fullmatch('[0-9a-f]{64}',metadata['tool_schema_sha256'])
  calls=metadata['rpc_events'];assert calls and all(c['outcome']=='result' for c in calls)
  assert all(c['method']=='tools/call' and c['tool_name'] in {'lookup_hosts','quarantine_hosts','release_owned','read_control_flows','read_flows','audit_policies'} for c in calls)
  assert len({c['request_id'] for c in calls})==len(calls)
 if protocol.get('protocol_revision') in ('bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):
  agent=record['native_agent'];assert agent['state']==('required_tools_completed' if protocol['protocol_revision']=='bedrock_mcp_convergence_v3' else 'model_completed') and not agent['fulfillment_verified']
  if protocol['protocol_revision']=='bedrock_mcp_convergence_v3':assert agent['completion_policy']=='required_tool_sequence'
  assert agent['model_id']=='amazon.nova-micro-v1:0' and 1<=agent['model_requests']<=4
  assert agent['model_config']['retry_policy']['total_max_attempts']==1
  expected={'I01':['quarantine_hosts','read_flows'],'I02':['release_owned','read_flows'],'I15':['audit_policies']}[record['class_id']]
  assert [e['name'] for e in agent['tool_events']]==expected and all(e['outcome']=='returned' for e in agent['tool_events'])
  if record['class_id']=='I01':assert agent['tool_events'][0]['arguments']=={'host_ids':record['targets']}
  if record['class_id']=='I15':assert agent['tool_events'][0]['arguments']=={'policies':record['audit_policy_ids']}
 if protocol.get('protocol_revision')=='application_bodies_convergence_v1':
  from intent_leakage.adapters.application_capture import verify_capture
  receipt=record['application_capture'];assert receipt['complete']
  assert receipt['directory']=='application_bodies/'+record['run_id']
  bodies=verify_capture(root/receipt['directory'],expected_sha256=receipt['manifest_sha256'])
  assert bodies['run_id']==record['run_id'] and len(bodies['exchanges'])==receipt['exchanges'] and bodies['body_bytes']==receipt['body_bytes']
  assert any(e['phase']=='active' and e['channel']=='CA' for e in bodies['exchanges'])
 assert record['run_id'] not in seen;seen.add(record['run_id'])
 for prefix in ('capture','trace'):
  path=(root/record[prefix+'_path']).resolve();assert path.is_relative_to(root.resolve())
  assert hashlib.sha256(path.read_bytes()).hexdigest()==record[prefix+'_sha256']
 events=[json.loads(x) for x in (root/record['trace_path']).read_text().splitlines()]
 assert len(events)==record['tls_records'] and all(e['trace_id']==record['run_id'] for e in events)
 text=(root/'captures'/f"{record['run_id']}.tcpdump.txt").read_text();drop=re.search(r'(\d+) packets dropped by kernel',text);assert drop and int(drop.group(1))==0
 cell=record['class_id']+':'+str(record['scope_q']);cells[cell]=cells.get(cell,0)+1;records.append(record)
assert len(records)==18 and len(cells)==9 and all(n==2 for n in cells.values())
if protocol.get('protocol_revision') in ('bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):
 usage=json.loads((root/'provider_usage.json').read_text());assert usage['failures']==0 and usage['requests']<=72
 assert usage['requests']==sum(r['native_agent']['model_requests'] for r in records)
 assert usage['input_tokens']==sum(e['usage']['inputTokens'] for r in records for e in r['native_agent']['model_events'])
 assert usage['output_tokens']==sum(e['usage']['outputTokens'] for r in records for e in r['native_agent']['model_events'])
print(json.dumps({'verified_runs':len(records),'tls_records':sum(r['tls_records'] for r in records),'cells':cells,'protocol_revision':protocol.get('protocol_revision','single_ping_v1'),'public_key_sha256':hashlib.sha256(public).hexdigest(),'manifest_sha256':hashlib.sha256((root/'manifests/runs.jsonl').read_bytes()).hexdigest()}))
