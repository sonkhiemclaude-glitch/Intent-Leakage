"""Real HTTPS + AF_UNIX mTLS; API double, not independent ONOS/dataplane."""
import json
import pytest
from test_mcp_stdio import controller
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.readonly_controller_bridge import ReadOnlyControllerBridge
from intent_leakage.adapters.local_shaped_transport import LocalTransportProfile
from intent_leakage.experiments.local_transport_lab import authenticated_socketpair,make_endpoints
pytestmark=pytest.mark.realtime


@pytest.mark.parametrize("tcp",[False,True])
def test_large_https_body_survives_fixed_cells_exactly_without_dummy_reads(controller,tcp):
    config,state,_=controller;state['flows']['999']={'id':'999','appId':'foreign','blob':'x'*70000}
    original=json.loads(json.dumps(state['flows']));observed=[]
    adapter=OnosRestAdapter(config['ca_origin'],username=config['username'],password=config['password'],ca_file=config['ca_file'],allow_external=True,timeout_s=.5,application_observer=observed.append)
    bridge=ReadOnlyControllerBridge(adapter,allow_external=True,max_response_bytes=131072)
    profile=LocalTransportProfile(channel='CA',cell_bytes=4096,rounds=128,period_s=.01,max_message_bytes=131072,max_pending=1,max_lateness_s=.1,io_timeout_s=.5)
    with authenticated_socketpair(loopback_tcp=tcp) as pair:
        agent,endpoint=make_endpoints(pair,profile=profile,lab=bridge,lead_s=.5,allow_loopback_tcp=tcp)
        try:
            result=agent.submit(b'{"op":"read","resource":"flows"}');agent.start();endpoint.start()
            raw=result.result(timeout=4)
            metrics=[agent.wait(4),endpoint.wait(4)]
            assert all(m['success'] for m in metrics),metrics
            assert all(m['transport_kind']==('ipv4_loopback' if tcp else 'unix') for m in metrics)
            assert raw==observed[0].response_body and len(raw)>65536
            assert json.loads(raw)=={'flows':list(original.values())}
            assert all(len(m['application_cells_sent'])==128 for m in metrics)
            assert len(bridge.calls)==1 and len(state['calls'])==1 and state['calls'][0][0]=='GET'
            assert state['flows']==original and not adapter._owned
        finally:
            agent.close();endpoint.close();agent.wait(2);endpoint.wait(2)


def test_loopback_tcp_requires_explicit_endpoint_opt_in():
    from intent_leakage.adapters.local_shaped_transport import LocalShapedEndpoint,TransportError
    import time
    with authenticated_socketpair(loopback_tcp=True) as pair:
        with pytest.raises(TransportError):
            LocalShapedEndpoint(pair[0],role='agent',profile=LocalTransportProfile(),start_at=time.monotonic()+1,session_id=b'x'*16)


def test_readonly_qualification_runner_reports_exact_body_without_credentials(controller):
    from intent_leakage.experiments.readonly_bridge_lab import run_readonly_bridge
    config,state,_=controller;state['flows']['999']={'id':'999','blob':'a'*70000}
    private={'origin':config['ca_origin'],'username':config['username'],'password':'fixture','ca_file':config['ca_file'],'timeout_s':.5}
    profile=LocalTransportProfile(channel='CA',cell_bytes=4096,rounds=128,period_s=.01,max_message_bytes=131072,max_pending=1,max_messages=1,max_lateness_s=.1,io_timeout_s=.5)
    report=run_readonly_bridge(private,profile=profile,allow_external=True)
    assert report['success'] and report['exact_https_body_reassembled']
    assert report['response_body_bytes']>65536 and report['completed_http_reads']==1
    assert 'password' not in json.dumps(report) and not report['remote_mutations_supported']
    assert not report['wire_cost_measured'] and len(state['calls'])==1
