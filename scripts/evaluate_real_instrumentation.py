"""Exploratory session-held-out RF evaluation; never confirmatory paper results."""
import argparse,csv,hashlib,json,math,random,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def split_sessions(sessions,seed):
    train,test=set(),set()
    for topology in sorted({v[1] for v in sessions}):
        ids=sorted(v[0] for v in sessions if v[1]==topology)
        random.Random(seed+topology).shuffle(ids)
        n=max(1,round(len(ids)*.3));test.update(ids[:n]);train.update(ids[n:])
    assert train and test and not train&test
    return train,test


def prepare(output,cohorts):
    from intent_leakage.data.capture_pipeline import build_pilot_features
    rows=[];sources=[];seen=set();columns=None
    for name,folder in cohorts:
        root=Path(folder);checkpoint=json.loads((root/'checkpoint.json').read_text())
        for batch,expected in sorted(checkpoint['completed'].items()):
            artifacts=root/batch/'artifacts';protocol=json.loads((artifacts/'protocol.json').read_text())
            revision=protocol.get('protocol_revision')
            assert revision=={'direct':'dataplane_convergence_v2','mcp':'mcp_stdio_convergence_v1','native':'bedrock_mcp_convergence_v3'}[name]
            status=json.loads((artifacts/'status.json').read_text());assert status['success'] and all(status['host_network_unchanged'].values())
            assert hashlib.sha256((artifacts/'manifests/runs.jsonl').read_bytes()).hexdigest()==expected['manifest_sha256']
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)/'features.csv'
                report=build_pilot_features(artifacts,public_key_sha256=expected['public_key_sha256'],output=path)
                if columns is None:columns=report['feature_columns']
                assert columns==report['feature_columns'] and len(columns)==57
                for row in csv.DictReader(path.open()):
                    assert row['run_id'] not in seen;seen.add(row['run_id'])
                    values={k:float(row[k]) for k in columns};assert all(math.isfinite(v) for v in values.values())
                    rows.append({'run_id':row['run_id'],'class_id':row['class_id'],'scope_q':int(row['scope_q']),
                        'cohort':name,'session':str(protocol['seed']),'topology':protocol['topology']['name'],**values})
            sources.append({'cohort':name,'batch':batch,'revision':revision,'session':str(protocol['seed']),
                'manifest_sha256':expected['manifest_sha256'],'public_key_sha256':expected['public_key_sha256']})
    assert len({r['cohort'] for r in rows})==2
    output.mkdir(parents=True,exist_ok=False)
    bundle={'kind':'exploratory_real_instrumentation','paper_results_claimed':False,'features':columns,'rows':rows,'sources':sources,'cohorts':sorted({r['cohort'] for r in rows})}
    (output/'input.json').write_text(json.dumps(bundle,separators=(',',':')))
    print(json.dumps({'rows':len(rows),'features':len(columns),'input_sha256':hashlib.sha256((output/'input.json').read_bytes()).hexdigest()}))


def fit_one(spec,rows,features,output):
    import joblib,numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
    source,dest,scenario,view,target,seed,control=spec
    sessions={(r['session'],r['topology']) for r in rows}
    if scenario=='session_holdout':train_ids,test_ids=split_sessions(sessions,str(seed))
    else:
        origin='pilot_two_switch_eight_host' if scenario=='two_to_four' else 'pilot_four_ring_eight_host'
        train_ids={s for s,t in sessions if t==origin};test_ids={s for s,t in sessions if t!=origin}
    train=[r for r in rows if r['cohort']==source and r['session'] in train_ids]
    test=[r for r in rows if r['cohort']==dest and r['session'] in test_ids]
    assert train and test and not {r['session'] for r in train}&{r['session'] for r in test}
    assert not {r['run_id'] for r in train}&{r['run_id'] for r in test}
    cols=[v for v in features if v.startswith(view+'__')];assert len(cols)==19
    def label(r):return r['class_id'] if target=='class' else str(r['scope_q']) if target=='scope' else r['class_id']+':'+str(r['scope_q'])
    y=np.array([label(r) for r in train]);truth=[label(r) for r in test]
    if control:np.random.default_rng(seed+10000).shuffle(y)
    model=RandomForestClassifier(n_estimators=128,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=1)
    model.fit([[r[c] for c in cols] for r in train],y)
    prediction=model.predict([[r[c] for c in cols] for r in test])
    identity='-'.join(map(str,spec));path=output/'models'/(identity+'.joblib');joblib.dump(model,path,compress=0)
    predictions=[{'run_id':r['run_id'],'session':r['session'],'truth':t,'prediction':str(p)} for r,t,p in zip(test,truth,prediction)]
    predpath=output/'predictions'/(identity+'.json');predpath.write_text(json.dumps(predictions))
    return {'id':identity,'source':source,'destination':dest,'scenario':scenario,'view':view,'target':target,'seed':seed,
        'permuted_training_labels':control,'train_rows':len(train),'test_rows':len(test),'train_sessions':sorted(train_ids),'test_sessions':sorted(test_ids),
        'features':cols,'accuracy':accuracy_score(truth,prediction),'balanced_accuracy':balanced_accuracy_score(truth,prediction),
        'macro_f1':f1_score(truth,prediction,average='macro',zero_division=0),'balanced_chance':1/(9 if target=='joint' else 3),
        'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'predictions_sha256':hashlib.sha256(predpath.read_bytes()).hexdigest()}


def run(bundle_path,output,cpus):
    import importlib.metadata,os
    from joblib import Parallel,delayed,parallel_config
    bundle=json.loads(bundle_path.read_text());features=bundle['features'];rows=bundle['rows']
    assert len(features)==57 and all(c.startswith(('ct__','ca__','ctca__')) for c in features)
    # Match topology/order seeds across protocols; exclude v1 batches and their MCP counterparts.
    cohorts=sorted({r['cohort'] for r in rows});assert len(cohorts)==2
    common=set.intersection(*[{r['session'] for r in rows if r['cohort']==c} for c in cohorts])
    matched=[r for r in rows if r['session'] in common]
    assert len(matched)==2*18*len(common)
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir();(output/'predictions').mkdir()
    specs=[(a,b,scenario,view,target,seed,False) for a in cohorts for b in cohorts
        for scenario in ('session_holdout','two_to_four','four_to_two') for view in ('ct','ca','ctca')
        for target in ('class','scope','joint') for seed in range(5)]
    specs += [(a,a,'session_holdout',view,target,seed,True) for a in cohorts
        for view in ('ct','ca','ctca') for target in ('class','scope','joint') for seed in range(5)]
    start=time.monotonic();results=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        for result in Parallel(n_jobs=cpus,return_as='generator_unordered')(delayed(fit_one)(s,matched,features,output) for s in specs):
            results.append(result)
            with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            (output/'checkpoint.json').write_text(json.dumps({'fits_complete':len(results),'fits_target':len(specs),'elapsed_s':time.monotonic()-start}))
    report={'kind':bundle['kind'],'paper_results_claimed':False,'input_sha256':hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        'input_rows':len(rows),'matched_rows':len(matched),'common_sessions':sorted(common),'fits':len(results),'cpus':cpus,'elapsed_s':time.monotonic()-start,
        'versions':{v:importlib.metadata.version(v) for v in ('numpy','scikit-learn','joblib','scipy')},
        'limitations':[('Exploratory three-class exact-intent native/MCP traces; no shaping.' if 'native' in cohorts else 'Exploratory three-class deterministic controller/MCP traces; no LLM or shaping.'),
            'Two fixed co-located topologies, one host; session holdout is not new-infrastructure generalization.',
            'Models fixed before scoring, no test-set tuning; scope and class labels, IDs, seeds and timestamps excluded from inputs.',
            'Five overlapping split repeats are sensitivity checks, not independent confidence intervals.',
            'Cross-protocol transfer is reported separately; different protocols are never pooled for fitting.'],
        'results':sorted(results,key=lambda r:r['id'])}
    (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('fits','matched_rows','elapsed_s')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--cohort',nargs=2,action='append');p.add_argument('--bundle',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpus',type=int,default=6);a=p.parse_args()
    if a.prepare:prepare(a.output,a.cohort)
    else:
        assert 1<=a.cpus<=8;run(a.bundle,a.output,a.cpus)
