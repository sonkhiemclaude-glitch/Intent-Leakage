"""Exploratory fixed-distance sequence attacks on signed real pilot traces."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_real_instrumentation import split_sessions


def prepare(bundle_path,cohorts,output):
    from intent_leakage.features.extraction import sequence_representation
    bundle=json.loads(bundle_path.read_text());byid={r['run_id']:r for r in bundle['rows']};found=set();lengths=[]
    for cohort,folder in cohorts:
        root=Path(folder);checkpoint=json.loads((root/'checkpoint.json').read_text())
        for task,expected in checkpoint['completed'].items():
            artifacts=root/task/'artifacts';manifest=artifacts/'manifests/runs.jsonl'
            assert hashlib.sha256(manifest.read_bytes()).hexdigest()==expected['manifest_sha256']
            for line in manifest.read_text().splitlines():
                r=json.loads(line)['record'];row=byid[r['run_id']];assert row['cohort']==cohort
                trace=artifacts/r['trace_path'];assert hashlib.sha256(trace.read_bytes()).hexdigest()==r['trace_sha256']
                events=[json.loads(line) for line in trace.read_text().splitlines()];seq={}
                for view,channel in [('ct','CT'),('ca','CA'),('ctca',None)]:
                    representation=sequence_representation(events,channel=channel).as_matrix()
                    seq[view]=representation.tolist();lengths.append(len(representation))
                row['sequences']=seq;found.add(r['run_id'])
    assert found==set(byid)
    result={'kind':'exploratory_real_sequences','paper_results_claimed':False,'base_input_sha256':hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        'rows':list(byid.values()),'sequence_max_length':256,'fixed_size_scale':16384,'max_observed_length':max(lengths),
        'empty_sequences':sum(v==0 for v in lengths),'truncated_sequences':sum(v>256 for v in lengths),'sequence_count':len(lengths)}
    output.mkdir(parents=True,exist_ok=False);(output/'input.json').write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps({k:result[k] for k in ('sequence_count','max_observed_length','truncated_sequences')}))


def fit_one(spec,rows,output,max_length=256):
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    source,dest,scenario,view,target,seed,control,metric=spec
    sessions={(r['session'],r['topology']) for r in rows}
    if scenario=='session_holdout':train_ids,test_ids=split_sessions(sessions,str(seed))
    else:
        origin='pilot_two_switch_eight_host' if scenario=='two_to_four' else 'pilot_four_ring_eight_host'
        train_ids={s for s,t in sessions if t==origin};test_ids={s for s,t in sessions if t!=origin}
    train=[r for r in rows if r['cohort']==source and r['session'] in train_ids]
    test=[r for r in rows if r['cohort']==dest and r['session'] in test_ids]
    assert train and test and not {r['session'] for r in train}&{r['session'] for r in test}
    assert not {r['run_id'] for r in train}&{r['run_id'] for r in test}
    intervals=[p[1] for r in train for p in r['sequences'][view][:max_length] if p[1]>0]
    scale=max(float(np.quantile(intervals,.95)),1e-6) if intervals else 1.0
    def encode(data):
        values=np.zeros((len(data),max_length,2))
        for i,row in enumerate(data):
            seq=np.asarray(row['sequences'][view][:max_length],dtype=float)
            if not len(seq):continue
            values[i,:len(seq),0]=seq[:,0]/16384
            values[i,:len(seq),1]=seq[:,1]/scale
        assert np.isfinite(values).all();return values.reshape(len(data),-1)
    def label(r):return r['class_id'] if target=='class' else str(r['scope_q']) if target=='scope' else r['class_id']+':'+str(r['scope_q'])
    y=np.array([label(r) for r in train]);truth=[label(r) for r in test]
    if control:np.random.default_rng(seed+10000).shuffle(y)
    model=KNeighborsClassifier(n_neighbors=3,weights='distance',metric=metric,algorithm='brute',n_jobs=1)
    model.fit(encode(train),y);prediction=model.predict(encode(test))
    identity='-'.join(map(str,spec));model_spec={'type':'exact_sequence_knn','max_length':max_length,'size_scale':16384,'iat_scale_train_only':scale,
        'view':view,'metric':metric,'n_neighbors':3,'weights':'distance','training_run_ids':[r['run_id'] for r in train],
        'training_labels':y.tolist(),'reconstruct_from_input_bundle':True}
    path=output/'models'/(identity+'.json');path.write_text(json.dumps(model_spec))
    predpath=output/'predictions'/(identity+'.json');predpath.write_text(json.dumps([{'run_id':r['run_id'],'truth':t,'prediction':str(p)} for r,t,p in zip(test,truth,prediction)]))
    return {'id':identity,'source':source,'destination':dest,'scenario':scenario,'view':view,'target':target,'seed':seed,
        'metric':metric,'permuted_training_labels':control,'train_rows':len(train),'test_rows':len(test),
        'train_sessions':sorted(train_ids),'test_sessions':sorted(test_ids),'iat_scale_train_only':scale,
        'accuracy':accuracy_score(truth,prediction),'balanced_accuracy':balanced_accuracy_score(truth,prediction),
        'macro_f1':f1_score(truth,prediction,average='macro',zero_division=0),'balanced_chance':1/(9 if target=='joint' else 3),
        'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256(predpath.read_bytes()).hexdigest()}


def run(bundle_path,output,cpus,max_length=256):
    import importlib.metadata
    from joblib import Parallel,delayed,parallel_config
    bundle=json.loads(bundle_path.read_text());rows=bundle['rows']
    cohorts=sorted({r['cohort'] for r in rows});assert len(cohorts)==2
    common=set.intersection(*[{r['session'] for r in rows if r['cohort']==c} for c in cohorts]);matched=[r for r in rows if r['session'] in common]
    assert len(matched)==len(common)*36
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir();(output/'predictions').mkdir()
    specs=[(a,b,scenario,view,target,seed,False,metric) for a in cohorts for b in cohorts
        for scenario in ('session_holdout','two_to_four','four_to_two') for view in ('ct','ca','ctca')
        for target in ('class','scope','joint') for seed in range(5) for metric in ('euclidean','manhattan')]
    specs += [(a,a,'session_holdout',view,target,seed,True,metric) for a in cohorts for view in ('ct','ca','ctca')
        for target in ('class','scope','joint') for seed in range(5) for metric in ('euclidean','manhattan')]
    results=[];start=time.monotonic()
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=cpus,return_as='generator_unordered')(delayed(fit_one)(s,matched,output,max_length) for s in specs):
            results.append(result)
            with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            (output/'checkpoint.json').write_text(json.dumps({'fits_complete':len(results),'fits_target':len(specs),'elapsed_s':time.monotonic()-start}))
    report={'kind':bundle['kind'],'paper_results_claimed':False,'input_sha256':hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        'matched_rows':len(matched),'fits':len(results),'cpus':cpus,'elapsed_s':time.monotonic()-start,
        'sequence_max_length':max_length,'max_observed_sequence_length':bundle['max_observed_length'],'input_truncated_sequences':sum(len(seq)>max_length for r in rows for seq in r['sequences'].values()),'matched_truncated_sequences':sum(len(seq)>max_length for r in matched for seq in r['sequences'].values()),'empty_sequences':bundle['empty_sequences'],
        'versions':{v:importlib.metadata.version(v) for v in ('numpy','scikit-learn','joblib','scipy')},
        'limitations':[('Exploratory three-class exact-intent native/MCP traces, no shaping.' if 'native' in cohorts else 'Exploratory three-class deterministic real controller/MCP traces, no LLM or shaping.'),
            f'Fixed {max_length}-event padded sequences; size normalization fixed, interarrival normalization fitted ONLY on source training sessions.',
            'Both fixed distance metrics are reported; no winner selected using held-out results.',
            'Five overlapping split repeats are descriptive sensitivity checks, not independent confidence intervals.',
            'Two fixed co-located topologies and one physical host; no new-infrastructure generalization claim.',
            'Model artifacts are exact reconstructable KNN specifications referencing the hashed input bundle.'],
        'results':sorted(results,key=lambda r:r['id'])}
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'fits':len(results),'elapsed_s':report['elapsed_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--base-bundle',type=Path);p.add_argument('--cohort',nargs=2,action='append');p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpus',type=int,default=6);p.add_argument('--max-length',type=int,choices=(256,512),default=256);a=p.parse_args()
    if a.prepare:prepare(a.base_bundle,a.cohort,a.output)
    else:
        assert 1<=a.cpus<=8;run(a.bundle,a.output,a.cpus,a.max_length)
