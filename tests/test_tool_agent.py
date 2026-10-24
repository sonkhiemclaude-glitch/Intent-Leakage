"""Native Converse protocol with an SDK double; no paid model calls."""
import copy

import pytest

from intent_leakage.adapters.base import ExternalExecutionDisabledError
from intent_leakage.adapters.bedrock import BedrockAdapter, BedrockConfig
from intent_leakage.adapters.mcp_onos_tools import list_tools
from intent_leakage.orchestration.tool_agent import (
    AgentBudget, AgentExecutionError, BedrockOnosToolAgent,
)


def tool_call(identity='call1', name='read_flows', arguments=None):
    return {'toolUse': {'toolUseId': identity, 'name': name, 'input': arguments or {}}}


def response(*blocks, stop='tool_use'):
    return {'stopReason': stop, 'output': {'message': {'role': 'assistant', 'content': list(blocks)}},
            'usage': {'inputTokens': 10, 'outputTokens': 20}}


class Mcp:
    revision='fixture'
    def __init__(self):self.calls=[];self.error=False
    def list_tools(self):return list_tools()
    def call_tool(self,name,arguments):
        self.calls.append((name,arguments))
        if self.error:raise TimeoutError('unknown controller outcome')
        return {'flows':[]}


class Client:
    def __init__(self,replies):self.replies=iter(replies);self.calls=[];self.after_call=lambda:None
    def converse(self,**request):
        self.calls.append(copy.deepcopy(request));self.after_call();return next(self.replies)
    def close(self):pass


def model(client,**changes):
    from dataclasses import replace
    config=BedrockConfig('amazon.nova-micro-v1:0','us-east-1',False,{'temperature':0},128,
                         {'total_max_attempts':1},'reuse')
    return BedrockAdapter(replace(config,**changes),allow_external=True,client_factory=lambda:client)


def agent(client,mcp=None,**kwargs):
    return BedrockOnosToolAgent(model(client),mcp or Mcp(),system_prompt='Frozen lab tool-use prompt.',
                               allow_external=True,**kwargs)


def test_native_tool_result_roundtrip_preserves_ids_and_does_not_claim_fulfillment():
    client=Client([response(tool_call()),response({'text':'Completed.'},stop='end_turn')]);mcp=Mcp()
    result=agent(client,mcp).run('Read the current flow state.')
    assert result['state']=='model_completed' and result['fulfillment_verified'] is False
    assert result['model_requests']==2 and result['tool_calls']==1
    assert mcp.calls==[('read_flows',{})]
    assert client.calls[1]['messages'][-1]=={'role':'user','content':[
        {'toolResult':{'toolUseId':'call1','content':[{'json':{'flows':[]}}]}}]}
    assert len(client.calls[0]['toolConfig']['tools'])==6
    assert client.calls[0]['inferenceConfig']['maxTokens']==128


def test_invalid_second_tool_prevents_first_mutation_in_same_response():
    valid=tool_call('first','quarantine_hosts',{'host_ids':['00:00:00:00:00:01/None']})
    client=Client([response(valid,tool_call('second','run_shell',{'cmd':'unregistered'}))]);mcp=Mcp()
    with pytest.raises(AgentExecutionError) as caught:agent(client,mcp).run('Act on the lab host.')
    assert not mcp.calls and caught.value.record['tool_calls']==0


def test_duplicate_call_id_cannot_replay_an_action_across_rounds():
    client=Client([response(tool_call()),response(tool_call())]);mcp=Mcp()
    with pytest.raises(AgentExecutionError):agent(client,mcp).run('Read flow state.')
    assert len(mcp.calls)==1 and len(client.calls)==2


def test_model_request_budget_stops_infinite_tool_loop_without_extra_action():
    client=Client([response(tool_call('a')),response(tool_call('b'))]);mcp=Mcp()
    with pytest.raises(AgentExecutionError) as caught:
        agent(client,mcp,budget=AgentBudget(max_model_requests=2)).run('Read flow state.')
    assert len(client.calls)==2 and len(mcp.calls)==1
    assert caught.value.record['model_requests']==2


def test_tool_budget_is_checked_for_whole_response_before_actions():
    client=Client([response(tool_call('a'),tool_call('b'))]);mcp=Mcp()
    with pytest.raises(AgentExecutionError):agent(client,mcp,budget=AgentBudget(max_tool_calls=1)).run('Read.')
    assert mcp.calls==[]


def test_mutation_timeout_is_not_retried_and_retains_unknown_outcome():
    client=Client([response(tool_call('a','quarantine_hosts',{'host_ids':['00:00:00:00:00:01/None']}))])
    mcp=Mcp();mcp.error=True
    with pytest.raises(AgentExecutionError) as caught:agent(client,mcp).run('Quarantine lab host.')
    assert len(mcp.calls)==1 and len(client.calls)==1
    assert caught.value.record['tool_events'][0]['outcome']=='unknown_after_error'


def test_truncated_tool_response_never_actuates():
    client=Client([response(tool_call(),stop='max_tokens')]);mcp=Mcp()
    with pytest.raises(AgentExecutionError):agent(client,mcp).run('Read.')
    assert not mcp.calls


def test_late_model_response_cannot_pass_elapsed_deadline():
    ticks=[0.];client=Client([response({'text':'Done'},stop='end_turn')]);mcp=Mcp()
    client.after_call=lambda:ticks.__setitem__(0,6.)
    with pytest.raises(AgentExecutionError) as caught:
        agent(client,mcp,budget=AgentBudget(max_elapsed_s=5),clock=lambda:ticks[0]).run('Read.')
    assert caught.value.record['elapsed_s']==6 and not mcp.calls


def test_oversized_request_is_stopped_before_any_paid_model_invocation():
    client=Client([])
    with pytest.raises(AgentExecutionError) as caught:agent(client).run('x'*70000)
    assert not client.calls and caught.value.record['model_requests']==0


def test_oversized_tool_result_is_not_sent_back_to_the_model():
    client=Client([response(tool_call())]);mcp=Mcp()
    def large_result(name,arguments):mcp.calls.append((name,arguments));return {'payload':'x'*40000}
    mcp.call_tool=large_result
    with pytest.raises(AgentExecutionError) as caught:agent(client,mcp).run('Read.')
    assert len(client.calls)==1 and len(mcp.calls)==1
    assert caught.value.record['tool_events'][0]['outcome']=='result_over_budget'


def test_gate_precedes_tool_discovery():
    with pytest.raises(ExternalExecutionDisabledError):
        BedrockOnosToolAgent(None,None,system_prompt='x')


@pytest.mark.parametrize('changes',[{'streaming':True},{'retry_policy':{'total_max_attempts':2}}])
def test_hidden_api_attempts_and_unimplemented_streaming_are_rejected(changes):
    with pytest.raises(ValueError):
        BedrockOnosToolAgent(model(Client([]),**changes),Mcp(),system_prompt='x',allow_external=True)


def test_fixed_sequence_stops_before_extra_model_query():
    from intent_leakage.orchestration.pilot_intent import IntentBoundTools
    mcp=Mcp();bound=IntentBoundTools(mcp,[('read_flows',{})])
    # The second model response is intentionally an unwanted extra tool call.
    client=Client([response(tool_call()),response(tool_call('extra'))])
    result=agent(client,bound,stop_after_required_tools=True).run('Read flows once.')
    assert result['state']=='required_tools_completed'
    assert result['completion_policy']=='required_tool_sequence'
    assert result['fulfillment_verified'] is False
    assert len(client.calls)==1 and len(mcp.calls)==1


def test_fixed_sequence_rejects_early_final_and_unbound_adapter():
    from intent_leakage.orchestration.pilot_intent import IntentBoundTools
    with pytest.raises(ValueError):agent(Client([]),stop_after_required_tools=True)
    bound=IntentBoundTools(Mcp(),[('read_flows',{})])
    with pytest.raises(AgentExecutionError) as exc:
        agent(Client([response({'text':'DONE'},stop='end_turn')]),bound,stop_after_required_tools=True).run('Read once.')
    assert exc.value.record['state']=='failed' and not bound.complete
