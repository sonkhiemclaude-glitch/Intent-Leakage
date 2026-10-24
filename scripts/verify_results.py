"""Verify published artifacts and independently recompute stored prediction metrics.

No model loading, fitting, credentials, controller access or network requests.
"""
from pathlib import Path
import hashlib,json,statistics,tarfile
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def scores(rows):
 truth=[str(r['truth']) for r in rows];pred=[str(r['prediction']) for r in rows]
 labels=sorted(set(truth));ba=statistics.mean(sum(t==g==k for t,g in zip(truth,pred))/truth.count(k) for k in labels)
 f1=[]
 for k in sorted(set(truth)|set(pred)):
  tp=sum(t==g==k for t,g in zip(truth,pred));fp=sum(t!=k and g==k for t,g in zip(truth,pred));fn=sum(t==k and g!=k for t,g in zip(truth,pred));f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return {'accuracy':sum(t==g for t,g in zip(truth,pred))/len(truth),'balanced_accuracy':ba,'macro_f1':statistics.mean(f1)}

def load_archive(path):
 hashes={};objects={}
 with tarfile.open(path,'r|gz') as archive:
  for m in archive:
   if not m.isfile():continue
   data=archive.extractfile(m).read();hashes[m.name]=hashlib.sha256(data).hexdigest()
   if m.name.endswith('.json') and ('/models/' not in m.name):objects[m.name]=json.loads(data)
 return hashes,objects

def main():
 for line in (DATA/'SHA256SUMS.txt').read_text().splitlines():
  expected,name=line.split('  ',1);p=(DATA/name).resolve();assert p.is_relative_to(DATA.resolve());assert digest(p)==expected,name
 provenance=json.loads((DATA/'provenance.json').read_text())
 assert digest(DATA/'analysis_results.tar.gz')==provenance['analysis_archive_sha256']
 hashes,obj=load_archive(DATA/'analysis_results.tar.gz');verified=0;summaries=[]
 for method,extension in [('rf','joblib'),('knn256','json'),('knn512','json'),('cnn','pt')]:
  report=obj[f'outputs/{method}/report.json'];bundle='rf_bundle' if method=='rf' else 'seq_bundle'
  assert report['input_sha256']==hashes[f'outputs/{bundle}/input.json']
  for r in report['results']:
   base=f'outputs/{method}/';pred=base+'predictions/'+r['id']+'.json';model=base+'models/'+r['id']+'.'+extension
   assert hashes[pred]==r['predictions_sha256'];assert hashes[model]==r['model_sha256']
   if 'model_metadata_sha256' in r:assert hashes[base+'models/'+r['id']+'.json']==r['model_metadata_sha256']
   assert not set(r['train_sessions'])&set(r['test_sessions'])
   if method=='cnn':assert not set(r['validation_sessions'])&(set(r['train_sessions'])|set(r['test_sessions']))
   assert len(obj[pred])==r['test_rows']
   for k,v in scores(obj[pred]).items():assert abs(v-r[k])<1e-12,(method,r['id'],k)
   verified+=1
  for cohort in ('mcp','native'):
   for view in ('ct','ca','ctca'):
    for target in ('class','scope'):
     rows=[r for r in report['results'] if r['source']==r['destination']==cohort and r['view']==view and r['target']==target and r.get('scenario','session_holdout')=='session_holdout' and not r.get('permuted_training_labels',r.get('permuted_labels',False)) and r.get('metric','euclidean')=='euclidean']
     if rows:summaries.append({'method':method,'cohort':cohort,'view':view,'target':target,'mean_balanced_accuracy':statistics.mean(r['balanced_accuracy'] for r in rows),'min':min(r['balanced_accuracy'] for r in rows),'max':max(r['balanced_accuracy'] for r in rows),'overlapping_splits':len(rows)})
 assert verified==3276
 assert digest(DATA/'ablation_results.tar.gz')==provenance['ablation_archive_sha256']
 ah,ao=load_archive(DATA/'ablation_results.tar.gz');ar=ao['ablations/report.json'];av=0
 for r in ar['results']:
  pred='ablations/predictions/'+r['id']+'.json';model='ablations/models/'+r['id']+'.joblib'
  assert ah[pred]==r['predictions_sha256'];assert ah[model]==r['model_sha256'];assert not set(r['train_sessions'])&set(r['test_sessions'])
  assert len(ao[pred])==r['test_rows'];assert abs(scores(ao[pred])['balanced_accuracy']-r['balanced_accuracy'])<1e-12;av+=1
 assert av==216
 # Session-complete/post-hoc extension: verify all archived fits and predictions
 # without deserializing the fitted estimators.
 extension_path=DATA/'posthoc_extension_results.tar.gz'
 assert digest(extension_path)==provenance['posthoc_extension_archive_sha256']
 eh,eo=load_archive(extension_path);er=eo['evaluation/report.json'];protocol=eo['evaluation/protocol.json']
 for name,h in eo['evaluation/checksums.json'].items():assert eh['evaluation/'+name]==h,name
 assert er['input_archive_sha256']==provenance['analysis_archive_sha256']
 assert er['script_sha256']==provenance['posthoc_extension_script_sha256']
 assert er['fits']==1010 and len(er['results'])==1010 and er['new_network_or_model_requests']==0
 seqrows=obj['outputs/seq_bundle/input.json']['rows'];lookup={r['run_id']:r for r in seqrows}
 ev=0
 for r in er['results']:
  predpath='evaluation/predictions/'+r['id']+'.json';assert eh[predpath]==r['predictions_sha256']
  assert not set(r['train_sessions'])&set(r['test_sessions'])
  predictions=eo[predpath];assert len(predictions)==r['test_rows']
  for q in predictions:
   original=lookup[q['run_id']];assert original['cohort']==r['cohort'] and q['session'] in r['test_sessions']
   assert q['class_id']==original['class_id'] and q['scope_q']==original['scope_q']
   if r['kind']!='permutation':
    truth=original['class_id'] if r['target']=='class' else str(original['scope_q']) if r['target']=='scope' else original['class_id']+':'+str(original['scope_q'])
    assert q['truth']==truth
  for k,v in scores(predictions).items():assert abs(v-r[k])<1e-12,(r['id'],k)
  if 'model_sha256' in r:
   model='evaluation/models/'+r['id']+'.joblib';assert eh[model]==r['model_sha256']
  ev+=1
 assert ev==1010
 summary=json.loads((DATA/'posthoc_extension_summary.json').read_text())
 model_reload=json.loads((DATA/'posthoc_extension_model_reload.json').read_text())
 assert summary['verified_fits']==1010 and summary['new_collection_runs']==0
 assert summary['input_archive_sha256']==provenance['analysis_archive_sha256']
 assert model_reload['reloaded_oof_models']==252 and model_reload['all_reloaded_predictions_match_saved_predictions']
 engineering=json.loads((DATA/'engineering_runs.json').read_text());assert len(engineering['rows'])==8
 writes=[]
 for row in engineering['rows']:
  assert row['controller_baseline_preserved'] and row['post_read_reachability']
  writes.extend(row['cell_write_lateness_ms']);folder=DATA/'readonly'/row['batch']/row['run_id'];records=[json.loads(x) for x in (folder/'tls_records.jsonl').read_text().splitlines()]
  assert len(records)==528
 assert len(writes)==4096
 result={'verified_analysis_fits':verified,'verified_ablation_fits':av,'verified_posthoc_extension_fits':ev,'matched_rows':504,'new_extension_collection_runs':0,'reloaded_extension_loso_models':252,'engineering_runs':8,'engineering_cell_writes':len(writes),'cell_write_lateness_ms':{'median':statistics.median(writes),'max':max(writes)},'all_results_exploratory':True,'summary':summaries,'ablation_summary':ar['summary']}
 out=ROOT/'results';out.mkdir(exist_ok=True);(out/'verified_metrics.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('summary','ablation_summary')},indent=2))
if __name__=='__main__':main()
