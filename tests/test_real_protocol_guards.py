import pandas as pd
import pytest
from intent_leakage.experiments.protocol_guards import validate_real_attacker_protocol
from intent_leakage.splits.grouped import GroupLeakageError


def frame():
 return pd.DataFrame([{'intent_instance_id':f'i{k}','phrase_family_id':split,'topology_id':top,'model_id':model,'split':split,'ct__total_bytes':100} for k,(split,top,model) in enumerate([('train','TA','modelA'),('validation','TA','modelA'),('test','TA','modelA'),('test','TB','modelA'),('test','TA','modelB')])])
def check(f,views=None):
 return validate_real_attacker_protocol(f,feature_columns_by_view=views or {'CT':['ct__total_bytes']},split_column='split',core_topology='TA',core_model='modelA')

def test_frozen_core_and_single_factor_transfer_are_accepted():
 result=check(frame());assert result['core_model']=='modelA' and len(result['split_assignment_sha256'])==64

def test_phrase_and_instance_leakage_fail_before_model_fitting():
 for column in ('phrase_family_id','intent_instance_id'):
  f=frame();f.loc[1,column]=f.loc[0,column]
  with pytest.raises(GroupLeakageError):check(f)

def test_transfer_refit_and_joint_transfer_are_rejected():
 f=frame();f.loc[3,'split']='validation';f.loc[3,'phrase_family_id']='transfer-validation'
 with pytest.raises(ValueError,match='test'):check(f)
 f=frame();f.loc[3,'model_id']='modelB'
 with pytest.raises(ValueError,match='exactly one'):check(f)

@pytest.mark.parametrize('views',[{'CA':['ct__total_bytes']},{'CT':['ct__class_id']},{'CL':['class_id']},{'unknown':['ct__total_bytes']}])
def test_view_crossing_and_label_aliases_are_rejected(views):
 with pytest.raises(ValueError,match='observation view'):check(frame(),views)

def test_pilot_and_missing_identity_metadata_are_rejected():
 f=frame();f.loc[0,'split']='instrumentation_only'
 with pytest.raises(ValueError,match='splits'):check(f)
 f=frame();f.loc[0,'intent_instance_id']=None
 with pytest.raises(ValueError,match='empty'):check(f)


def test_real_training_and_prediction_persist_protocol(tmp_path):
 import joblib
 from intent_leakage.experiments.train_attackers import train_grouped_random_forest_attackers_from_csv,predict_grouped_random_forest_attackers_from_csv
 records=[]
 for split,n in [('train',8),('validation',2),('test',2)]:
  for i in range(n):
   records.append({'intent_instance_id':f'{split}-{i}','phrase_family_id':split,'topology_id':'TA','model_id':'modelA','split':split,'synthetic':False,'class_id':'I01' if i%2 else 'I02','scope_q':1,'ct__total_bytes':i%2})
 p=tmp_path/'features.csv';pd.DataFrame(records).to_csv(p,index=False);model=tmp_path/'model.joblib'
 train_grouped_random_forest_attackers_from_csv(p,model,feature_columns_by_view={'CT':['ct__total_bytes']},group_columns=(),split_column='split',n_estimators=5,core_topology='TA',core_model='modelA')
 assert joblib.load(model)['protocol_guard']['core_topology']=='TA'
 predict_grouped_random_forest_attackers_from_csv(p,model,tmp_path/'predictions.csv')
 assert len(pd.read_csv(tmp_path/'predictions.csv'))==2
