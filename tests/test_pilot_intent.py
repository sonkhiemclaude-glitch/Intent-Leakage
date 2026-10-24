import pytest
from intent_leakage.orchestration.pilot_intent import IntentBoundTools


class Mcp:
    revision='test'
    def __init__(self):self.calls=[]
    def list_tools(self):return []
    def call_tool(self,name,args):
        self.calls.append((name,args))
        return {'flows':[{'appId':'other'},{'appId':'org.intentleakage.pilot','state':'ADDED'}]}


def test_bound_intent_rejects_wrong_mutation_before_call():
    m=Mcp();bound=IntentBoundTools(m,[('quarantine_hosts',{'host_ids':['target']})])
    with pytest.raises(ValueError):bound.call_tool('quarantine_hosts',{'host_ids':['other']})
    with pytest.raises(ValueError):bound.call_tool('release_owned',{})
    assert m.calls==[] and not bound.complete


def test_flow_projection_is_explicit_and_no_raw_flows_sent_to_model():
    m=Mcp();bound=IntentBoundTools(m,[('read_flows',{})]);r=bound.call_tool('read_flows',{})
    assert r['total_flows']==2 and r['owned_added_flows']==1 and 'flows' not in r
    assert r['projection']=='counts_only_v1' and len(r['full_result_sha256'])==64
    assert bound.complete
    with pytest.raises(ValueError):bound.call_tool('read_flows',{})
    assert len(m.calls)==1
