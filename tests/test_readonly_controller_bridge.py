import json,time
import pytest
from intent_leakage.adapters.readonly_controller_bridge import ReadOnlyControllerBridge
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.local_shaped_transport import RequestContext,TransportCancelled,EnvelopeOverflow
from intent_leakage.adapters.base import ExternalExecutionDisabledError


def binding(raw=b'{"flows":[]}'):
    a=object.__new__(OnosRestAdapter);a.application_observation_errors=[];calls=[]
    def read(resource):calls.append(resource);return raw
    a.read_application_body=read
    return ReadOnlyControllerBridge(a,allow_external=True),calls


@pytest.mark.parametrize('payload',[b'{"op":"write","resource":"flows"}',b'{"op":"read","resource":"hosts"}',b'{"op":"read","resource":"flows","url":"other"}',b'{"op":"write","op":"read","resource":"flows"}',b'[]'])
def test_mutations_unknown_resources_and_ambiguous_requests_never_reach_controller(payload):
    bridge,calls=binding()
    with pytest.raises(ValueError):bridge.handle(payload,RequestContext())
    assert not calls


def test_exact_body_not_reserialized_and_default_external_gate():
    bridge,calls=binding(b'{ "flows" : [] }\n')
    assert bridge.handle(b'{"op":"read","resource":"flows"}',RequestContext())==b'{ "flows" : [] }\n'
    assert calls==['flows'] and bridge.calls[0]['outcome']=='returned'
    with pytest.raises(ExternalExecutionDisabledError):ReadOnlyControllerBridge(bridge.adapter)


def test_expired_context_prevents_io_and_late_result_is_not_published():
    bridge,calls=binding()
    with pytest.raises(TransportCancelled):bridge.handle(b'{"op":"read","resource":"flows"}',RequestContext(deadline=time.monotonic()-1))
    assert not calls
    context=RequestContext()
    def cancel(_):context.cancel();return b'{}'
    bridge.adapter.read_application_body=cancel
    with pytest.raises(TransportCancelled):bridge.handle(b'{"op":"read","resource":"flows"}',context)
    assert bridge.calls[-1]['outcome']=='failed'


def test_large_body_rejected_without_retry_or_dynamic_limit_change():
    bridge,calls=binding(b'x'*131073)
    with pytest.raises(EnvelopeOverflow):bridge.handle(b'{"op":"read","resource":"flows"}',RequestContext())
    assert calls==['flows'] and bridge.max_response_bytes==131072
