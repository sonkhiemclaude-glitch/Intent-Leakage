"""Run the frozen real-data evaluation chain ONLY after a complete native queue."""
import hashlib,json,os,signal,subprocess,tarfile,time,sys
from pathlib import Path
from common import busy_reasons
ROOT=Path(__file__).resolve().parent


def main():
    config=json.loads((ROOT/'config.json').read_text());deadline=time.monotonic()+1800
    def persist(state,**fields):
        p=ROOT/'checkpoint.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({'state':state,'updated_epoch':time.time(),**fields},indent=2));temp.replace(p)
    def step(name,args):
        persist('running',step=name)
        with (ROOT/'worker.log').open('a') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(5);reasons=busy_reasons(os.getpid())
                if reasons or time.monotonic()>deadline:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise RuntimeError('Interrupted for '+str(reasons or ['deadline']))
            if child.returncode:raise RuntimeError('Step failed: '+name+', rc='+str(child.returncode))
    try:
        parent=Path(config['native_root']);cp=json.loads((parent/'checkpoint.json').read_text())
        assert cp['state']=='complete' and cp['verified_runs']==config['expected_native_runs'] and len(cp['completed'])==config['expected_native_runs']//18
        while busy_reasons(os.getpid()):
            persist('waiting_for_idle')
            if time.monotonic()>deadline:raise RuntimeError('Idle window exhausted')
            time.sleep(10)
        with tarfile.open(ROOT/'source.tar.gz') as archive:archive.extractall(ROOT,filter='data')
        python=config['python'];scripts=ROOT/'scripts'
        step('dependencies',[python,config['pip'],'install','cryptography==50.0.2'])
        destinations={}
        for name,original in [('mcp',config['mcp_root']),('native',config['native_root'])]:
            original=Path(original);checkpoint=json.loads((original/'checkpoint.json').read_text())
            expected_runs=config['expected_native_runs'] if name=='native' else config['expected_mcp_runs']
            assert checkpoint['state']=='complete' and checkpoint['verified_runs']==expected_runs and len(checkpoint['completed'])==expected_runs//18
            collection=ROOT/'collected'/name;collection.mkdir(parents=True)
            (collection/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
            for batch,expected in checkpoint['completed'].items():
                target=collection/batch;target.mkdir();artifacts=original/expected['attempt_path']/'artifacts'
                assert hashlib.sha256((artifacts/'manifests/runs.jsonl').read_bytes()).hexdigest()==expected['manifest_sha256']
                (target/'artifacts').symlink_to(artifacts,target_is_directory=True)
            destinations[name]=collection
        resultroot=ROOT/'outputs';resultroot.mkdir()
        rf_bundle=resultroot/'rf_bundle';seq_bundle=resultroot/'seq_bundle'
        pairs=['--cohort','mcp',str(destinations['mcp']),'--cohort','native',str(destinations['native'])]
        step('verify_and_prepare_rf',[python,str(scripts/'evaluate_real_instrumentation.py'),'--prepare',*pairs,'--output',str(rf_bundle)])
        step('verify_and_prepare_sequences',[python,str(scripts/'evaluate_real_sequences.py'),'--prepare','--base-bundle',str(rf_bundle/'input.json'),*pairs,'--output',str(seq_bundle)])
        jobs=[('rf','evaluate_real_instrumentation.py',rf_bundle/'input.json',[]),
              ('knn256','evaluate_real_sequences.py',seq_bundle/'input.json',['--max-length','256']),
              ('knn512','evaluate_real_sequences.py',seq_bundle/'input.json',['--max-length','512']),
              ('cnn','evaluate_real_cnn.py',seq_bundle/'input.json',[])]
        reports=[]
        for name,script,bundle,extra in jobs:
            step(name,[python,str(scripts/script),'--bundle',str(bundle),'--output',str(resultroot/name),'--cpus','6',*extra])
            report=json.loads((resultroot/name/'report.json').read_text());reports.append({'job':name,'fits':report['fits'],'matched_rows':report['matched_rows'],'elapsed_s':report['elapsed_s']})
        total=sum(r['fits'] for r in reports);assert total==3276 and all(r['matched_rows']==config['expected_matched_rows'] for r in reports)
        (ROOT/'requirements.freeze.txt').write_text(subprocess.check_output([python,config['pip'],'freeze'],text=True))
        summary={'state':'complete','paper_results_claimed':False,'native_rows':config['expected_native_runs'],'mcp_rows':config['expected_mcp_runs'],'matched_rows':config['expected_matched_rows'],'total_fits':total,'jobs':reports,
                 'limitations':['Three-class exact-intent execution, no free-form planning or full 15-class claim.','No CL/provider capture or live shaping.','Separate protocols; no pooled training or test refitting.']}
        (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));persist('packing',fits=total)
        with tarfile.open(ROOT/'results.tar.gz','w:gz') as archive:
            for name in ('outputs','source.tar.gz','config.json','summary.json','requirements.freeze.txt','followup.py','common.py','dispatch.sh','source_versions.json','worker.log'):archive.add(ROOT/name,arcname=name)
        persist('complete',fits=total,archive_sha256=hashlib.sha256((ROOT/'results.tar.gz').read_bytes()).hexdigest())
    except Exception as exc:
        persist('failed',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
