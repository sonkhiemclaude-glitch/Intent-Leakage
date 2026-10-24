"""Exercise the real REST serializer against a LOCAL HTTPS API double only."""
import json
import pytest
from test_mcp_stdio import controller,HOST,DEVICE
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.owned_flow_transaction import OnosPilotFlowBackend,OwnedFlowTransaction,FixtureFlowBackend
pytestmark=pytest.mark.realtime


def test_journaled_controller_binding_and_body_observer_over_local_https(controller,tmp_path):
    config,state,_=controller
    foreign=next(iter(FixtureFlowBackend().flows.values()));foreign['id']='999';state['flows']['999']=foreign
    observed=[]
    adapter=OnosRestAdapter(config['ct_origin'],username=config['username'],password=config['password'],ca_file=config['ca_file'],
                            allow_external=True,timeout_s=2,application_observer=observed.append)
    tx=OwnedFlowTransaction(OnosPilotFlowBackend(adapter,allow_external=True),tmp_path/'https_journal.json')
    result=tx.execute('a'*64,[HOST]);assert result['controller_verified'] and not result['independent_dataplane_verified']
    cached=tx.execute('a'*64,[HOST]);assert cached['replayed_from_journal'] and not cached['current_state_reverified']
    assert tx.rollback()['state']=='cleaned' and state['flows']=={'999':foreign}
    assert not adapter._owned and not adapter.application_observation_errors
    assert sum(method=='POST' for method,_,_ in state['calls'])==1
    assert sum(method=='DELETE' for method,_,_ in state['calls'])==2
    assert any(s.method=='POST' and s.request_body for s in observed)
    metadata=json.dumps([s.metadata() for s in observed])
    assert 'Authorization' not in metadata and 'Basic ' not in metadata


def test_real_http_body_capture_roundtrip_and_rollback(controller,tmp_path):
    from intent_leakage.adapters.application_capture import ApplicationCapture,verify_capture
    config,state,_=controller
    capture=ApplicationCapture(tmp_path/'bodies','https01')
    adapter=OnosRestAdapter(config['ct_origin'],username=config['username'],password=config['password'],ca_file=config['ca_file'],
                            allow_external=True,timeout_s=2,application_observer=capture.observer('CT'))
    tx=OwnedFlowTransaction(OnosPilotFlowBackend(adapter,allow_external=True),tmp_path/'journal.json')
    capture.phase='active';tx.execute('b'*64,[HOST])
    capture.phase='cleanup';assert tx.rollback()['state']=='cleaned'
    receipt=capture.finish(run_success=True,observer_errors=adapter.application_observation_errors)
    manifest=verify_capture(capture.root,expected_sha256=receipt['manifest_sha256'])
    assert any(r['method']=='POST' and r['phase']=='active' for r in manifest['exchanges'])
    assert sum(r['method']=='DELETE' and r['phase']=='cleanup' for r in manifest['exchanges'])==2
    assert not state['flows'] and not adapter._owned
