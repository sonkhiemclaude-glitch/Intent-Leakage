import json
import pytest
from intent_leakage.adapters.owned_flow_transaction import FixtureFlowBackend,OwnedFlowTransaction,TransactionError,OnosPilotFlowBackend
from intent_leakage.adapters.base import ExternalExecutionDisabledError
HOST='00:00:00:00:00:01/None'


def test_read_mutate_verify_rollback_and_persistent_no_replay(tmp_path):
    backend=FixtureFlowBackend();before=backend.read();path=tmp_path/'journal.json'
    tx=OwnedFlowTransaction(backend,path);applied=tx.execute('a'*64,[HOST])
    assert applied['controller_verified'] and not applied['independent_dataplane_verified']
    assert not applied['formal_I01_semantics'] and len(backend.read())==3
    assert OwnedFlowTransaction(backend,path).execute('a'*64,[HOST])['state']=='applied' and backend.posts==1
    assert tx.rollback()['state']=='cleaned' and backend.read()==before
    tx.rollback();assert len(backend.deletes)==2
    assert OwnedFlowTransaction(backend,path).execute('a'*64,[HOST])['state']=='cleaned' and backend.posts==1


def test_lost_ack_does_not_retry_or_sweep_unknown_owned_rules(tmp_path):
    backend=FixtureFlowBackend();backend.fail_after_install=True;path=tmp_path/'j.json';tx=OwnedFlowTransaction(backend,path)
    with pytest.raises(TimeoutError):tx.execute('b'*64,[HOST])
    assert json.loads(path.read_text())['state']=='uncertain' and backend.posts==1
    with pytest.raises(TransactionError):tx.execute('b'*64,[HOST])
    with pytest.raises(TransactionError):tx.rollback()
    assert not backend.deletes and len(backend.read())==3


def test_foreign_and_changed_owned_state_are_never_deleted(tmp_path):
    backend=FixtureFlowBackend();tx=OwnedFlowTransaction(backend,tmp_path/'j.json');s=tx.execute('c'*64,[HOST])
    key=tuple(s['owned'][0]['identity']);backend.flows[key]['priority']=10
    with pytest.raises(TransactionError):tx.rollback()
    assert not backend.deletes and ('of:0000000000000001','1') in backend.flows


def test_journal_identity_collision_and_external_gate(tmp_path):
    backend=FixtureFlowBackend();tx=OwnedFlowTransaction(backend,tmp_path/'j.json');tx.execute('d'*64,[HOST])
    with pytest.raises(TransactionError):tx.execute('e'*64,[HOST])
    with pytest.raises(ExternalExecutionDisabledError):OnosPilotFlowBackend(object())
    assert backend.posts==1


def test_busy_journal_fails_before_mutation(tmp_path):
    backend=FixtureFlowBackend();tx=OwnedFlowTransaction(backend,tmp_path/'j.json')
    with tx._lock():
        with pytest.raises(TransactionError,match='busy'):tx.execute('f'*64,[HOST])
    assert backend.posts==0
