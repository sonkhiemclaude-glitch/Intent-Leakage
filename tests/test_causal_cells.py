import pytest
from intent_leakage.shaping.causal_cells import PlannedReply,run_logical_session
from intent_leakage.shaping.application_fit import ApplicationSample,fit_candidates,select_on_validation,score_candidate


def test_exact_fragment_causality_without_wall_clock():
    result=run_logical_session([(0.,b'x'*500)],lambda _:PlannedReply(b'y'*300,.037),cell_bytes=128,rounds=30,period_s=.01)
    assert result['success'] and len(result['writes'])==60
    handler=result['handlers'][0];response=result['responses'][0]
    assert handler['request_complete_s']==pytest.approx(.05)
    assert handler['completed_s']==pytest.approx(.087)
    assert response['received_s']>=handler['completed_s']
    assert response['payload']==b'y'*300
    assert not result['live_timing_measured']


def test_late_dependency_overflow_never_extends_envelope():
    result=run_logical_session([(0,b'command')],lambda _:PlannedReply(b'answer',1),rounds=10,period_s=.01)
    assert not result['success'] and not result['responses']
    assert result['termination_s']==.1 and len(result['writes'])==20


def test_dummies_do_not_invoke_planner():
    def forbidden(_):raise AssertionError('dummy reached application')
    result=run_logical_session([],forbidden,rounds=8)
    assert result['success'] and all(v['dummy'] for v in result['writes'])


def test_logical_fifo_and_explicit_stop():
    result=run_logical_session([(0,b'first'),(0,b'second')],lambda p:PlannedReply(p,.02),rounds=30)
    assert result['success'] and [r['payload'] for r in result['responses']]==[b'first',b'second']
    assert result['handlers'][1]['started_s']>=result['handlers'][0]['completed_s']
    stopped=run_logical_session([(0,b'request')],lambda _:PlannedReply(b'reply',1),rounds=30,stop_at=.02)
    assert not stopped['success'] and 'explicit_stop' in stopped['failures']


def test_train_only_fit_does_not_change_after_test_overflow():
    train=[ApplicationSample('a','train',b'a'*200,b'b'*400,.02)]
    candidates=fit_candidates(train,cell_sizes=(128,512),periods=(.01,),margin=1)
    assert len(candidates)==2
    for c in candidates:
        assert not score_candidate(train[0],cell_bytes=c['cell_bytes'],period_s=c['period_s'],rounds=c['rounds'])['overflow']
        assert not c['deployable']
    before=repr(candidates)
    unseen=ApplicationSample('unseen','test',b'a'*200,b'b'*60000,1)
    assert score_candidate(unseen,cell_bytes=128,period_s=.01,rounds=candidates[0]['rounds'])['overflow']
    assert repr(candidates)==before
    with pytest.raises(ValueError):fit_candidates([unseen])
    with pytest.raises(ValueError):select_on_validation(candidates,[unseen],byte_budget=99999,duration_budget_s=10)


def test_validation_budget_infeasibility_and_no_identity_overlap():
    train=[ApplicationSample('a','train',b'a',b'b',0)]
    candidates=fit_candidates(train,cell_sizes=(128,),periods=(.01,))
    val=[ApplicationSample('b','validation',b'a',b'b',0)]
    assert select_on_validation(candidates,val,byte_budget=1,duration_budget_s=1)['selected_candidate'] is None
    chosen=select_on_validation(candidates,val,byte_budget=9999,duration_budget_s=1)
    assert chosen['selected_candidate']==0 and not chosen['test_data_used']
    with pytest.raises(ValueError):select_on_validation(candidates,[ApplicationSample('a','validation',b'a',b'b',0)],byte_budget=9999,duration_budget_s=1)
