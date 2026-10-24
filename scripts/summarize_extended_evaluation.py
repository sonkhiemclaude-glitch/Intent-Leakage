"""Verify post-hoc fit artifacts and summarize independent-held-session predictions.

Bootstrap ranges are conditional summaries of fixed OOF predictions, not a
population guarantee or a replacement for independently collected testbeds.
"""
from pathlib import Path
import argparse,json,hashlib,tarfile,statistics
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def metrics(rows):
 t=[r['truth'] for r in rows];g=[r['prediction'] for r in rows];classes=sorted(set(t));f=[]
 for c in sorted(set(t)|set(g)):
  tp=sum(a==b==c for a,b in zip(t,g));fp=sum(a!=c and b==c for a,b in zip(t,g));fn=sum(a==c and b!=c for a,b in zip(t,g));f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return {'accuracy':sum(a==b for a,b in zip(t,g))/len(t),'balanced_accuracy':statistics.mean(sum(a==b==c for a,b in zip(t,g))/t.count(c) for c in classes),'macro_f1':statistics.mean(f)}

def summarize(evaluation,archive,engineering,output,reload_models=False):
 report=json.loads((evaluation/'report.json').read_text());assert sha(archive)==report['input_archive_sha256'];checks=json.loads((evaluation/'checksums.json').read_text())
 for name,h in checks.items():assert sha(evaluation/name)==h,name
 with tarfile.open(archive,'r:gz') as t:bundle=json.load(t.extractfile('outputs/seq_bundle/input.json'))
 cohorts=('mcp','native');common=set.intersection(*[{r['session'] for r in bundle['rows'] if r['cohort']==c} for c in cohorts]);rows=[r for r in bundle['rows'] if r['session'] in common];lookup={r['run_id']:r for r in rows};preds={};reloaded=0
 for r in report['results']:
  p=evaluation/'predictions'/(r['id']+'.json');assert sha(p)==r['predictions_sha256'];v=json.loads(p.read_text());assert len(v)==r['test_rows'];assert not set(r['train_sessions'])&set(r['test_sessions'])
  for k,x in metrics(v).items():assert abs(x-r[k])<1e-12,(r['id'],k)
  for q in v:
   original=lookup[q['run_id']];assert original['cohort']==r['cohort'] and q['session'] in r['test_sessions'];assert q['class_id']==original['class_id'] and q['scope_q']==original['scope_q']
   if r['kind']!='permutation':
    truth=original['class_id'] if r['target']=='class' else str(original['scope_q']) if r['target']=='scope' else original['class_id']+':'+str(original['scope_q']);assert q['truth']==truth
  preds[r['id']]=v
  if 'model_sha256' in r:
   modelpath=evaluation/'models'/(r['id']+'.joblib');assert sha(modelpath)==r['model_sha256']
   if reload_models:
    import joblib
    model=joblib.load(modelpath);training=[q for q in rows if q['cohort']==r['cohort'] and q['session'] in r['train_sessions']]
    def label(q):return q['class_id'] if r['target']=='class' else str(q['scope_q']) if r['target']=='scope' else q['class_id']+':'+str(q['scope_q'])
    names=sorted({label(q) for q in training});values=model.predict([[lookup[q['run_id']][c] for c in r['features']] for q in v]);assert [names[int(x)] for x in values]==[q['prediction'] for q in v];reloaded+=1
 assert len(report['results'])==1010
 result={'all_results_exploratory':True,'new_collection_runs':0,'verified_fits':len(report['results']),'reloaded_oof_models':reloaded,'input_archive_sha256':report['input_archive_sha256'],'protocol':report['counts'],'loso':[],'learning':[],'prefix':[],'baselines':[],'permutation':[],'scope_by_operation':[],'scope_by_operation_all_views':[]}
 for c in cohorts:
  for target in ('class','scope','joint'):
   for view in ('ct','ca','ctca'):
    fits=[r for r in report['results'] if r['kind']=='loso' and r['cohort']==c and r['target']==target and r['view']==view];assert len(fits)==14
    predictions=[q for r in fits for q in preds[r['id']]];assert len(predictions)==252 and len({q['run_id'] for q in predictions})==252
    assert {q['run_id'] for q in predictions}=={q['run_id'] for q in rows if q['cohort']==c}
    groups={top:[r['balanced_accuracy'] for r in fits if lookup[preds[r['id']][0]['run_id']]['topology']==top] for top in sorted({r['topology'] for r in rows})};assert all(len(v)==7 for v in groups.values())
    rng=np.random.default_rng(24062026);boot=np.mean(np.concatenate([rng.choice(x,(5000,7),replace=True) for x in groups.values()],axis=1),axis=1)
    result['loso'].append({'cohort':c,'target':target,'view':view,**metrics(predictions),'conditional_session_bootstrap_95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'session_min':min(r['balanced_accuracy'] for r in fits),'session_max':max(r['balanced_accuracy'] for r in fits),'test_runs':252,'session_folds':14})
    if target=='scope':
     for cl in ('I01','I02','I15'):
      item={'cohort':c,'view':view,'class_id':cl,**metrics([q for q in predictions if q['class_id']==cl]),'test_runs':84};result['scope_by_operation_all_views'].append(item)
      if view=='ca':result['scope_by_operation'].append(item)
  for kind,key,groupkeys in [('learning','learning',['target','budget']),('prefix','prefix',['target','budget']),('baseline','baselines',['target','method'])]:
   fits=[r for r in report['results'] if r['kind']==kind and r['cohort']==c]
   for group in sorted({tuple(r[k] for k in groupkeys) for r in fits}):
    subset=[r for r in fits if tuple(r[k] for k in groupkeys)==group];values=[r['balanced_accuracy'] for r in subset];record={'cohort':c,**dict(zip(groupkeys,group)),'mean':statistics.mean(values),'min':min(values),'max':max(values),'values':values,'overlapping_splits':len(subset),'macro_f1_mean':statistics.mean(r['macro_f1'] for r in subset)}
    if kind=='prefix':record['all_trace_ended_by_budget_fraction']=statistics.mean(len(q['sequences']['ca'])<=record['budget'] for q in rows if q['cohort']==c)
    result[key].append(record)
  null=[r for r in report['results'] if r['kind']=='permutation' and r['cohort']==c];assert len(null)==199
  rr=[q for q in rows if q['cohort']==c]
  for r in null:
   rng=np.random.default_rng(20261024+int(r['seed']));assigned={};groups={}
   for q in rr:groups.setdefault((q['session'],q['scope_q']),[]).append(q)
   for key,group in sorted(groups.items()):
    shuffled=rng.permutation([q['class_id'] for q in group]).tolist()
    for q,x in zip(group,shuffled):assigned[q['run_id']]=x
   assert all(assigned[q['run_id']]==q['truth'] for q in preds[r['id']])
  observed=next(r for r in report['results'] if r['kind']=='baseline' and r['cohort']==c and r['target']=='class' and r['method']=='RF' and r['seed']==0)['balanced_accuracy'];values=[r['balanced_accuracy'] for r in null]
  result['permutation'].append({'cohort':c,'observed':observed,'null_mean':statistics.mean(values),'null_95th':float(np.quantile(values,.95)),'null_max':max(values),'conditional_permutation_p':(1+sum(x>=observed for x in values))/(len(values)+1),'null_values':values,'replicates':len(values),'assumption':'Within(session,scope) label exchangeability; fixed seed-zero session split; both train and test labels shuffled'})
 eng=json.loads(engineering.read_text());lateness=[x for r in eng['rows'] for x in r['cell_write_lateness_ms']];result['engineering']={'run_count':len(eng['rows']),'write_count':len(lateness),'lateness_quantiles_ms':{str(q):float(np.quantile(lateness,q)) for q in (.5,.95,.99,1)},'response_body_slot_budget_occupancy_range':[min(r['response_body_bytes']/(256*(4096-36)) for r in eng['rows']),max(r['response_body_bytes']/(256*(4096-36)) for r in eng['rows'])]}
 output.write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--engineering',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reload-models',action='store_true');a=p.parse_args();r=summarize(a.evaluation,a.archive,a.engineering,a.output,a.reload_models);print(json.dumps({'verified_fits':r['verified_fits'],'reloaded_oof_models':r['reloaded_oof_models'],'loso':r['loso'],'permutation':[{k:v for k,v in q.items() if k!='null_values'} for q in r['permutation']]},indent=2))
