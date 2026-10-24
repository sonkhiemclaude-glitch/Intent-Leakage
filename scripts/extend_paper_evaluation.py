"""Post-hoc offline analyses of the frozen matched corpus; no external services.

Fixed settings are written before fitting. Reuses collection sessions: all results
remain exploratory. Permutation p-values require within-block exchangeability.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time,tarfile,statistics
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def labels(r,target):return r['class_id'] if target=='class' else str(r['scope_q']) if target=='scope' else r['class_id']+':'+str(r['scope_q'])

def split_sessions(rows,seed):
 import random
 tr=set();te=set()
 for top in sorted({r['topology'] for r in rows}):
  ids=sorted({r['session'] for r in rows if r['topology']==top});random.Random(str(seed)+top).shuffle(ids);n=max(1,round(len(ids)*.3));te.update(ids[:n]);tr.update(ids[n:])
 return tr,te

def feature_prefix(seq,limit):
 from intent_leakage.features.extraction import extract_channel_features
 t=0.;events=[]
 for signed,gap in seq[:limit]:
  t+=gap;events.append({'timestamp_s':t,'length_bytes':int(abs(signed)),'direction':1 if signed>0 else -1,'channel':'CA'})
 return extract_channel_features(events,channel='CA')

def fit_task(spec,rows,features,prefix,output,store):
 import numpy as np,joblib
 from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
 from sklearn.linear_model import LogisticRegression
 from sklearn.svm import SVC
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler,LabelEncoder
 from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score
 kind,cohort,target,view,seed,budget,method=spec
 rr=[r for r in rows if r['cohort']==cohort];all_sessions=sorted({r['session'] for r in rr})
 if kind=='loso':te={str(seed)};tr=set(all_sessions)-te;modelseed=4100+all_sessions.index(str(seed))
 else:tr,te=split_sessions(rr,seed);modelseed=int(seed)
 if kind=='learning':
  import random
  selected=set()
  for top in sorted({r['topology'] for r in rr}):
   ids=sorted({r['session'] for r in rr if r['topology']==top and r['session'] in tr});random.Random('learning-'+str(seed)+'-'+top).shuffle(ids);selected.update(ids[:int(budget)//2])
  assert len(selected)==int(budget);tr=selected
 train=[r for r in rr if r['session'] in tr];test=[r for r in rr if r['session'] in te]
 assert tr and te and not tr&te
 assert not {r['run_id'] for r in train}&{r['run_id'] for r in test}
 y=[labels(r,target) for r in train];truth=[labels(r,target) for r in test]
 if kind=='permutation':
  # Shuffle BOTH training and test labels within(session,scope); keep cell counts.
  assigned={r['run_id']:r['class_id'] for r in rr};rng=np.random.default_rng(20261024+int(seed));groups={}
  for r in rr:groups.setdefault((r['session'],r['scope_q']),[]).append(r)
  for key,group in sorted(groups.items()):
   original=[r['class_id'] for r in group];shuffled=rng.permutation(original).tolist();assert sorted(original)==sorted(shuffled)
   for r,v in zip(group,shuffled):assigned[r['run_id']]=v
  y=[assigned[r['run_id']] for r in train];truth=[assigned[r['run_id']] for r in test];modelseed=0
  # Fixed seed-zero session split for every null replicate.
  tr,te=split_sessions(rr,0);train=[r for r in rr if r['session'] in tr];test=[r for r in rr if r['session'] in te]
  y=[assigned[r['run_id']] for r in train];truth=[assigned[r['run_id']] for r in test]
 cols=[c for c in features if c.startswith(view+'__')];assert len(cols)==19
 def matrix(data):
  if kind=='prefix':return [[prefix[(r['run_id'],int(budget))][c.split('__',1)[1]] for c in cols] for r in data]
  return [[r[c] for c in cols] for r in data]
 models={
  'RF':lambda:RandomForestClassifier(n_estimators=128,min_samples_leaf=2,class_weight='balanced',random_state=modelseed,n_jobs=1),
  'ExtraTrees':lambda:ExtraTreesClassifier(n_estimators=128,min_samples_leaf=2,class_weight='balanced',random_state=modelseed,n_jobs=1),
  'Logistic':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=modelseed)),
  'RBF-SVM':lambda:make_pipeline(StandardScaler(),SVC(C=1,gamma='scale',class_weight='balanced',random_state=modelseed)),
  'HistGB':lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.1,l2_regularization=1,random_state=modelseed),
 }
 encoder=LabelEncoder();encoded=encoder.fit_transform(np.asarray(y,dtype=str));model=models[method]();t=time.monotonic();model.fit(matrix(train),encoded);fit_s=time.monotonic()-t;pred=encoder.inverse_transform(model.predict(matrix(test)).astype(int));identity='__'.join(map(str,spec))
 predpath=output/'predictions'/(identity+'.json');predrows=[{'run_id':r['run_id'],'session':r['session'],'topology':r['topology'],'class_id':r['class_id'],'scope_q':r['scope_q'],'truth':str(a),'prediction':str(b)} for r,a,b in zip(test,truth,pred)];predpath.write_text(json.dumps(predrows,separators=(',',':')))
 result={'id':identity,'kind':kind,'cohort':cohort,'target':target,'view':view,'seed':seed,'budget':budget,'method':method,'train_sessions':sorted(tr),'test_sessions':sorted(te),'train_rows':len(train),'test_rows':len(test),'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro',zero_division=0)),'fit_seconds':fit_s,'predictions_sha256':sha(predpath),'training_labels_sha256':hashlib.sha256(json.dumps(y,separators=(',',':')).encode()).hexdigest(),'both_sides_labels_permuted':kind=='permutation','features':cols}
 if kind=='prefix':result['test_trace_ended_by_budget_fraction']=sum(len(r['sequences']['ca'])<=int(budget) for r in test)/len(test)
 if kind=='loso' and store:
  path=output/'models'/(identity+'.joblib');joblib.dump(model,path,compress=3);result['model_sha256']=sha(path)
 return result

def main():
 import numpy as np,importlib.metadata
 from joblib import Parallel,delayed,parallel_config
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpus',type=int,default=2);p.add_argument('--permutations',type=int,default=199);p.add_argument('--store-oof-models',action='store_true');a=p.parse_args();assert 1<=a.cpus<=4 and a.permutations>=99
 output=a.output;output.mkdir(parents=True,exist_ok=False)
 for f in ('predictions','models'):(output/f).mkdir()
 with tarfile.open(a.archive,'r:gz') as t:
  base=json.load(t.extractfile('outputs/rf_bundle/input.json'));seq=json.load(t.extractfile('outputs/seq_bundle/input.json'))
 cohorts=sorted({r['cohort'] for r in base['rows']});assert cohorts==['mcp','native'];common=set.intersection(*[{r['session'] for r in base['rows'] if r['cohort']==c} for c in cohorts]);rows=[r for r in seq['rows'] if r['session'] in common];assert len(rows)==504 and len(common)==14
 features=base['features'];prefix={};budgets=[4,8,16,32,64,128,256,512]
 for r in rows:
  for n in budgets:prefix[(r['run_id'],n)]=feature_prefix(r['sequences']['ca'],n)
 # Reconstructed complete features should agree up to floating-point timestamp precision.
 full_errors=[]
 for r in rows:
  x=feature_prefix(r['sequences']['ca'],100000)
  for k,v in x.items():
   if not np.isclose(v,r['ca__'+k],rtol=1e-4,atol=2e-5):full_errors.append((r['run_id'],k,float(v),r['ca__'+k]))
 assert not full_errors,full_errors[:5]
 specs=[]
 for c in cohorts:
  specs += [('loso',c,t,v,s,0,'RF') for t in ('class','scope','joint') for v in ('ct','ca','ctca') for s in sorted(common)]
  specs += [('learning',c,t,'ca',s,n,'RF') for t in ('class','scope') for s in range(5) for n in (2,4,6,8,10)]
  specs += [('prefix',c,t,'ca',s,n,'RF') for t in ('class','scope') for s in range(5) for n in budgets]
  specs += [('baseline',c,t,'ca',s,0,m) for t in ('class','scope') for s in range(5) for m in ('RF','ExtraTrees','Logistic','RBF-SVM','HistGB')]
  specs += [('permutation',c,'class','ca',s,0,'RF') for s in range(a.permutations)]
 protocol={'kind':'posthoc_offline_extension','confirmatory':False,'input_archive_sha256':sha(a.archive),'script_sha256':sha(__file__),'matched_rows':len(rows),'sessions_per_cohort':len(common),'planned_fits':len(specs),'counts':{k:sum(s[0]==k for s in specs) for k in ('loso','learning','prefix','baseline','permutation')},'forest':{'trees':128,'min_samples_leaf':2,'class_weight':'balanced'},'fixed_baselines':{'ExtraTrees':'128 trees,minleaf2,balanced','Logistic':'train-only StandardScaler,C1,maxiter3000,balanced','RBF-SVM':'train-only StandardScaler,C1,gamma-scale,balanced','HistGB':'100 iterations,15 leaves,lr0.1,l2=1'},'permutation':f'Both train and test labels shuffled within(session,scope),fixed seed-zero split,{a.permutations} replicates per cohort; conditional exchangeability required; no population/causal claim','prefix':'At most B assurance records OR trace end; offline partial-observation diagnostic,not guaranteed early online inference','learning':'Even topology-stratified nested session subsets;same four held-out sessions at every budget for each split seed','execution_cpus':a.cpus,'new_network_or_model_requests':0,'models_retained':'LOSO only' if a.store_oof_models else 'none; refit from code/input','versions':{k:importlib.metadata.version(k) for k in ('numpy','scikit-learn','scipy','joblib')},'created_epoch':time.time()}
 (output/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[];started=time.monotonic()
 with parallel_config(backend='loky',inner_max_num_threads=1):
  for r in Parallel(n_jobs=a.cpus,return_as='generator_unordered')(delayed(fit_task)(s,rows,features,prefix,output,a.store_oof_models) for s in specs):
   results.append(r)
   with (output/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
   if len(results)%25==0 or len(results)==len(specs):
    d={'completed':len(results),'total':len(specs),'elapsed_s':time.monotonic()-started};(output/'checkpoint.json').write_text(json.dumps(d));print(json.dumps(d),flush=True)
 report={**protocol,'fits':len(results),'elapsed_s':time.monotonic()-started,'results':sorted(results,key=lambda r:r['id'])};(output/'report.json').write_text(json.dumps(report,indent=2))
 (output/'checksums.json').write_text(json.dumps({str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'},indent=2))
if __name__=='__main__':main()
