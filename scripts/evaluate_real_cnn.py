"""Exploratory CPU CNN, with disjoint training/validation/test sessions."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_real_instrumentation import split_sessions


def fit_one(spec,rows,output):
    import numpy as np,torch
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    from intent_leakage.attackers.cnn import CNN1DTraceAttacker
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    source,dest,view,target,seed,control=spec
    sessions={(r['session'],r['topology']) for r in rows};train_ids,test_ids=split_sessions(sessions,str(seed))
    validation_ids=set()
    for topology in sorted({t for _,t in sessions}):
        candidates=sorted(s for s,t in sessions if t==topology and s in train_ids)
        random.Random(str(seed)+topology+'validation').shuffle(candidates);validation_ids.add(candidates[0])
    train_ids-=validation_ids
    assert not train_ids&validation_ids and not train_ids&test_ids and not validation_ids&test_ids
    train=[r for r in rows if r['cohort']==source and r['session'] in train_ids]
    validation=[r for r in rows if r['cohort']==source and r['session'] in validation_ids]
    test=[r for r in rows if r['cohort']==dest and r['session'] in test_ids]
    intervals=[p[1] for r in train for p in r['sequences'][view][:512] if p[1]>0]
    scale=max(float(np.quantile(intervals,.95)),1e-6) if intervals else 1.0
    def encode(data):
        values=np.zeros((len(data),512,2),dtype=np.float32)
        for i,r in enumerate(data):
            seq=np.asarray(r['sequences'][view][:512],dtype=np.float32)
            if not len(seq):continue
            values[i,:len(seq),0]=seq[:,0]/16384;values[i,:len(seq),1]=seq[:,1]/scale
        assert np.isfinite(values).all();return values
    def label(r):return r['class_id'] if target=='class' else str(r['scope_q']) if target=='scope' else r['class_id']+':'+str(r['scope_q'])
    y=np.array([label(r) for r in train]);yv=np.array([label(r) for r in validation]);truth=[label(r) for r in test]
    if control:
        np.random.default_rng(seed+10000).shuffle(y);np.random.default_rng(seed+20000).shuffle(yv)
    model=CNN1DTraceAttacker(hidden_channels=(16,32),epochs=20,patience=5,random_state=seed,device='cpu')
    model.fit(encode(train),y,X_validation=encode(validation),y_validation=yv)
    prediction=model.predict(encode(test));identity='-'.join(map(str,spec));weights=output/'models'/(identity+'.pt');torch.save(model.model_.state_dict(),weights)
    metadata={'classes':model.classes_.tolist(),'target':target,'view':view,'sequence_length':512,'sequence_features':2,'hidden_channels':[16,32],
        'kernel_size':3,'learning_rate':.001,'weight_decay':0.0,'batch_size':32,'max_epochs':20,'patience':5,'random_state':seed,
        'iat_scale_train_only':scale,'size_scale':16384,'train_run_ids':[r['run_id'] for r in train],'validation_run_ids':[r['run_id'] for r in validation],
        'training_history':list(model.training_history_),'state_dict_sha256':hashlib.sha256(weights.read_bytes()).hexdigest()}
    meta=output/'models'/(identity+'.json');meta.write_text(json.dumps(metadata))
    pred=output/'predictions'/(identity+'.json');pred.write_text(json.dumps([{'run_id':r['run_id'],'truth':t,'prediction':str(p)} for r,t,p in zip(test,truth,prediction)]))
    return {'id':identity,'source':source,'destination':dest,'view':view,'target':target,'seed':seed,'permuted_labels':control,
        'train_rows':len(train),'validation_rows':len(validation),'test_rows':len(test),'train_sessions':sorted(train_ids),
        'validation_sessions':sorted(validation_ids),'test_sessions':sorted(test_ids),'epochs_run':len(model.training_history_),
        'accuracy':accuracy_score(truth,prediction),'balanced_accuracy':balanced_accuracy_score(truth,prediction),
        'macro_f1':f1_score(truth,prediction,average='macro',zero_division=0),'balanced_chance':1/(9 if target=='joint' else 3),
        'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'model_metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
        'predictions_sha256':hashlib.sha256(pred.read_bytes()).hexdigest()}


def run(bundle_path,output,cpus):
    import importlib.metadata
    from joblib import Parallel,delayed,parallel_config
    bundle=json.loads(bundle_path.read_text());rows=bundle['rows'];cohorts=sorted({r['cohort'] for r in rows});assert len(cohorts)==2
    common=set.intersection(*[{r['session'] for r in rows if r['cohort']==c} for c in cohorts]);matched=[r for r in rows if r['session'] in common]
    assert len(matched)==36*len(common)
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir();(output/'predictions').mkdir()
    specs=[(a,b,v,t,s,False) for a in cohorts for b in cohorts for v in ('ct','ca','ctca') for t in ('class','scope','joint') for s in range(3)]
    specs += [(a,a,v,'class',s,True) for a in cohorts for v in ('ct','ca','ctca') for s in range(3)]
    start=time.monotonic();results=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=cpus,return_as='generator_unordered')(delayed(fit_one)(s,matched,output) for s in specs):
            results.append(result)
            with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            (output/'checkpoint.json').write_text(json.dumps({'fits_complete':len(results),'fits_target':len(specs),'elapsed_s':time.monotonic()-start}))
    report={'kind':'exploratory_real_cpu_cnn','paper_results_claimed':False,'input_sha256':hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        'matched_rows':len(matched),'fits':len(results),'cpus':cpus,'elapsed_s':time.monotonic()-start,
        'versions':{v:importlib.metadata.version(v) for v in ('numpy','scikit-learn','joblib','torch')},
        'limitations':[('Exploratory three-class exact-intent native/MCP traces; no shaping or full 15-class grid.' if 'native' in cohorts else 'Exploratory three-class deterministic controller/MCP traces; no LLM, shaping or full 15-class grid.'),
            'Training, validation and test session seeds are disjoint; input normalization uses training only; early stopping uses validation only.',
            'Fixed small architecture and 20-epoch ceiling are engineering settings, not a validation-selected search or mature CNN benchmark.',
            '512-record padding; no topology-transfer CNN case; RF and kNN report those separately.',
            'Three overlapping repeats are sensitivity checks, not independent confidence intervals.',
            'Label controls permute source training and validation labels, leaving test truth untouched.'],
        'results':sorted(results,key=lambda r:r['id'])}
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'fits':len(results),'elapsed_s':report['elapsed_s']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpus',type=int,default=6);a=p.parse_args();assert 1<=a.cpus<=8;run(a.bundle,a.output,a.cpus)
