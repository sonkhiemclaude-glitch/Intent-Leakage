"""Synthetic signed evidence tests the verifier, never research observations."""
import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from intent_leakage.adapters.application_capture import ApplicationCapture
from intent_leakage.adapters.application_observation import ApplicationExchange
ROOT=Path(__file__).resolve().parents[1]


def write_fixture(root):
    for name in ('manifests','captures','traces'):(root/name).mkdir()
    (root/'status.json').write_text(json.dumps({'success':True,'host_network_unchanged':{'route':True,'rules':True,'qdisc':True,'ens5':True}}))
    (root/'protocol.json').write_text(json.dumps({'protocol_revision':'application_bodies_convergence_v1'}))
    key=Ed25519PrivateKey.generate()
    (root/'signing_public_key.pem').write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
    envelopes=[]
    for cls in ('I01','I02','I15'):
        for q in (1,2,4):
            for rep in range(2):
                run=f'{cls}_{q}_{rep}';r={'run_id':run,'class_id':cls,'scope_q':q,'success':True,'cleanup_verified':True,'split':'instrumentation_only','protocol_revision':'application_bodies_convergence_v1','dataplane_checks':[]}
                for phase in ('preflight','prepared','postcondition','cleanup'):
                    hosts=[f'h{i+1}' for i in range(q)]+(['h7'] if phase in ('preflight','postcondition') else [])
                    for host in hosts:
                        expected=phase in ('preflight','cleanup') or host=='h7' or (phase=='prepared' and cls=='I01') or (phase=='postcondition' and cls=='I02')
                        r['dataplane_checks'].append({'phase':phase,'host':host,'expected_reachable':expected,'success':True,'elapsed_s':.1,'budget_s':8,'probes':[{'valid_icmp_result':True,'reachable':expected}]})
                for prefix,suffix,data in [('capture','pcap',b'SYNTHETIC_FIXTURE'),('trace','jsonl',(json.dumps({'trace_id':run,'channel':'CA'})+'\n').encode())]:
                    path=root/('captures' if prefix=='capture' else 'traces')/(run+'.'+suffix);path.write_bytes(data)
                    r[prefix+'_path']=str(path.relative_to(root));r[prefix+'_sha256']=hashlib.sha256(data).hexdigest()
                r['tls_records']=1;(root/'captures'/(run+'.tcpdump.txt')).write_text('0 packets dropped by kernel\n')
                c=ApplicationCapture(root/'application_bodies',run);c.phase='active';c.observer('CA')(ApplicationExchange('GET','flows',b'',b'{"flows":[]}',.01))
                r['application_capture']={**c.finish(run_success=True),'directory':'application_bodies/'+run}
                canonical=json.dumps(r,sort_keys=True,separators=(',',':')).encode()
                envelopes.append(json.dumps({'record':r,'signature_ed25519':base64.b64encode(key.sign(canonical)).decode()}))
    (root/'manifests/runs.jsonl').write_text('\n'.join(envelopes)+'\n')


def verify(root):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
    return subprocess.run([sys.executable,str(ROOT/'deployment/overnight/verify_batch.py'),str(root)],capture_output=True,text=True,env=env)


def test_application_batch_accepts_complete_signed_fixture_rejects_body_tampering(tmp_path):
    write_fixture(tmp_path);result=verify(tmp_path)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['verified_runs']==18
    next((tmp_path/'application_bodies').glob('*/000000.response.bin')).write_bytes(b'tampered')
    result=verify(tmp_path);assert result.returncode!=0 and 'checksum mismatch' in result.stderr


def test_inventory_retains_controller_session_groups_and_rejects_duplicate_batches(tmp_path):
    import importlib.util,pytest
    spec=importlib.util.spec_from_file_location('inventory',ROOT/'scripts/inventory_application_collection.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    write_fixture(tmp_path);result=module.inventory([tmp_path])
    assert result['active_exchange_count']==18 and result['independent_controller_session_count']==1
    assert not result['live_profile_ready'] and not result['processing_time_measured']
    with pytest.raises(ValueError,match='duplicate batch'):module.inventory([tmp_path,tmp_path])
