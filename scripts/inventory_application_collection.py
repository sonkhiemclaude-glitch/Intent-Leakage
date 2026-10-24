"""Verify retrieved qualification batches and inventory active HTTP bodies.

No fitting, no test-driven profile sizing, no exchange-time-as-service-time.
"""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from intent_leakage.adapters.application_capture import verify_capture


def inventory(inputs):
    sessions=[];seen=set();rows=[]
    for root in inputs:
        root=Path(root).resolve();protocol=json.loads((root/'protocol.json').read_text())
        if protocol.get('protocol_revision')!='application_bodies_convergence_v1':raise ValueError('mixed protocol cohorts refused')
        result=subprocess.run([sys.executable,str(ROOT/'deployment/overnight/verify_batch.py'),str(root)],capture_output=True,text=True,
                              env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
        if result.returncode:raise ValueError('batch verification failed: '+str(root))
        receipt=json.loads(result.stdout);session_id=receipt['manifest_sha256']
        if session_id in seen:raise ValueError('duplicate batch manifest')
        seen.add(session_id)
        sessions.append({'session_id':session_id,'verification':receipt,'seed':protocol.get('seed'),'topology':protocol.get('topology')})
        for line in (root/'manifests/runs.jsonl').read_text().splitlines():
            r=json.loads(line)['record'];capture=r['application_capture']
            manifest=verify_capture(root/capture['directory'],expected_sha256=capture['manifest_sha256'])
            for row in manifest['exchanges']:
                if row['phase']!='active':continue
                rows.append({'session_id':session_id,'run_id':r['run_id'],'class_id':r['class_id'],'scope_q':r['scope_q'],
                             **{k:row[k] for k in ('index','channel','method','resource','request_body_bytes','response_body_bytes',
                                                  'request_body_sha256','response_body_sha256','unshaped_exchange_elapsed_s')}})
    if not rows:raise ValueError('no active exchanges')
    return {'state':'verified_engineering_inventory','sessions':sessions,'active_exchanges':rows,
            'active_exchange_count':len(rows),'independent_controller_session_count':len(sessions),
            'confirmatory_test_eligible':False,'live_profile_ready':False,'processing_time_measured':False,
            'next_gate':'Collect disjoint assigned training/validation sessions; declare timing model before profile fitting.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();data=inventory(args.input)
    # Refuse overwrite of a previously verified inventory.
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    print(json.dumps({'state':data['state'],'sessions':len(data['sessions']),'active_exchanges':data['active_exchange_count'],'live_profile_ready':False}))


if __name__=='__main__':main()
