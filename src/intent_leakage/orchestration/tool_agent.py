"""Budgeted Converse tool-use loop for the frozen three-class ONOS MCP pilot.

Model completion is not independent proof of intent fulfillment. The caller's
run lifecycle must verify dataplane postconditions and cleanup, including after
any exception. Wall deadlines are checked at bounded SDK/RPC boundaries; a
deployment supervisor must enforce the outer hard deadline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
import time

from ..adapters.base import require_external_execution
from ..adapters.bedrock import BedrockAdapter
from ..adapters.mcp import McpToolAdapter
from ..adapters.mcp_onos_tools import TOOL_SCHEMA_SHA256, validate_call
from ..utils.hashing import stable_digest
from .pilot_intent import IntentBoundTools


@dataclass(frozen=True)
class AgentBudget:
    max_model_requests: int = 4
    max_tool_calls: int = 12
    max_elapsed_s: float = 120
    max_request_bytes: int = 65536
    max_tool_result_bytes: int = 32768

    def validate(self):
        if type(self.max_model_requests) is not int or not 1 <= self.max_model_requests <= 16:
            raise ValueError('Model request budget must be between 1 and 16')
        if type(self.max_tool_calls) is not int or not 1 <= self.max_tool_calls <= 32:
            raise ValueError('Tool-call budget must be between 1 and 32')
        if isinstance(self.max_elapsed_s, bool) or not isinstance(self.max_elapsed_s, (int, float)) or not math.isfinite(self.max_elapsed_s) or self.max_elapsed_s <= 0:
            raise ValueError('Elapsed-time budget must be finite and positive')
        for value in (self.max_request_bytes,self.max_tool_result_bytes):
            if type(value) is not int or not 1 <= value <= 1024*1024:
                raise ValueError('JSON byte budgets must be between 1 byte and 1 MiB')
        if self.max_tool_result_bytes > self.max_request_bytes:
            raise ValueError('Tool-result budget cannot exceed request budget')


class AgentExecutionError(RuntimeError):
    def __init__(self, message, record):
        self.record = record
        super().__init__(message)


class BedrockOnosToolAgent:
    def __init__(self, model: BedrockAdapter, mcp: McpToolAdapter, *, system_prompt: str,
                 budget: AgentBudget = AgentBudget(), allow_external: bool = False,
                 clock=time.monotonic, stop_after_required_tools: bool = False):
        require_external_execution(allow_external)
        budget.validate()
        if type(stop_after_required_tools) is not bool:
            raise ValueError('Invalid completion policy')
        if stop_after_required_tools and not isinstance(mcp, IntentBoundTools):
            raise ValueError('Required-tool completion needs a frozen intent binding')
        self.stop_after_required_tools=stop_after_required_tools
        if model.config.streaming:
            raise ValueError('This pilot loop requires non-streaming Converse')
        if model.config.retry_policy.get('total_max_attempts', 1) != 1:
            raise ValueError('The agent budget requires exactly one SDK attempt per model request')
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError('A frozen nonempty system prompt is required')
        self.model, self.mcp, self.budget = model, mcp, budget
        self.system_prompt, self.clock = system_prompt, clock
        self.tools = json.loads(json.dumps(mcp.list_tools()))
        if stable_digest(self.tools) != TOOL_SCHEMA_SHA256:
            raise ValueError('Unexpected ONOS MCP tool schema')
        self.tool_config = {'tools': [{'toolSpec': {'name': t['name'], 'description': t['description'],
                             'inputSchema': {'json': t['inputSchema']}}} for t in self.tools]}

    def run(self, prompt: str):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('A nonempty operator prompt is required')
        start = self.clock()
        messages = [{'role': 'user', 'content': [{'text': prompt}]}]
        record = {'state': 'started', 'paper_results_claimed': False,
                  'fulfillment_verified': False, 'completion_policy': 'required_tool_sequence' if self.stop_after_required_tools else 'model_end_turn', 'model_requests': 0, 'tool_calls': 0,
                  'tool_schema_sha256': TOOL_SCHEMA_SHA256, 'mcp_revision': self.mcp.revision,
                  'model_id': self.model.config.model_id, 'model_events': [], 'tool_events': [],
                  'model_config': asdict(self.model.config), 'budget': asdict(self.budget),
                  'system_prompt_sha256': stable_digest(self.system_prompt),
                  'operator_prompt_sha256': stable_digest(prompt)}
        seen_ids = set()

        def deadline():
            if self.clock()-start >= self.budget.max_elapsed_s:
                raise RuntimeError('Agent elapsed-time budget exhausted')

        try:
            for request_index in range(self.budget.max_model_requests):
                deadline()
                request = {'system': [{'text': self.system_prompt}], 'messages': messages,
                           'toolConfig': self.tool_config}
                wire = {**request, 'modelId': self.model.config.model_id,
                        'inferenceConfig': {**self.model.config.decoding_parameters,
                                            'maxTokens': self.model.config.max_response_tokens}}
                request_bytes = len(json.dumps(wire,separators=(',', ':'),allow_nan=False).encode())
                if request_bytes > self.budget.max_request_bytes:
                    raise RuntimeError('Agent model-request JSON byte budget exhausted')
                record['model_requests'] += 1
                response = self.model.invoke(request)
                event = {'stop_reason': response.get('stopReason'), 'usage': response.get('usage', {}),
                         'response_sha256': stable_digest(response),'request_bytes':request_bytes}
                record['model_events'].append(event)
                deadline()
                message = response.get('output', {}).get('message', {})
                event['assistant_message'] = json.loads(json.dumps(message))
                content = message.get('content')
                if message.get('role') != 'assistant' or not isinstance(content, list) or not content:
                    raise ValueError('Invalid Converse assistant message')
                calls = [block['toolUse'] for block in content if isinstance(block, dict) and 'toolUse' in block]
                if response.get('stopReason') == 'end_turn' and not calls:
                    if self.stop_after_required_tools and not self.mcp.complete:
                        raise ValueError('Model stopped before required tools completed')
                    record.update(state='model_completed', final_message=message)
                    break
                if response.get('stopReason') != 'tool_use' or not calls:
                    raise ValueError('Incomplete/refused model turn; no tool action accepted')
                if request_index+1 >= self.budget.max_model_requests:
                    raise RuntimeError('No model-request budget remains for a tool-result turn')
                if record['tool_calls']+len(calls) > self.budget.max_tool_calls:
                    raise RuntimeError('Agent tool-call budget exhausted')
                # Validate the whole batch before executing its first mutation.
                batch_ids = set()
                for call in calls:
                    if not isinstance(call, dict) or set(call) != {'toolUseId', 'name', 'input'}:
                        raise ValueError('Invalid Converse tool-use block')
                    identity = call['toolUseId']
                    if not isinstance(identity, str) or not re.fullmatch(r'[a-zA-Z0-9_.:-]{1,64}', identity):
                        raise ValueError('Invalid tool-use identity')
                    if identity in seen_ids or identity in batch_ids:
                        raise ValueError('Replayed/duplicate tool-use identity')
                    validate_call(call['name'], call['input'])
                    batch_ids.add(identity)
                seen_ids.update(batch_ids)
                messages.append(json.loads(json.dumps(message)))
                results = []
                for call in calls:
                    deadline()
                    record['tool_calls'] += 1
                    tool_event = {'name': call['name'], 'tool_use_id': call['toolUseId'], 'outcome': 'started',
                                  'arguments': json.loads(json.dumps(call['input']))}
                    record['tool_events'].append(tool_event)
                    try:
                        value = self.mcp.call_tool(call['name'], call['input'])
                    except Exception as exc:
                        tool_event.update(outcome='unknown_after_error', error_type=type(exc).__name__)
                        raise
                    result_bytes = len(json.dumps(value,separators=(',', ':'),allow_nan=False).encode())
                    tool_event.update(outcome='returned', result_sha256=stable_digest(value), result_bytes=result_bytes)
                    if result_bytes > self.budget.max_tool_result_bytes:
                        tool_event['outcome'] = 'result_over_budget'
                        raise RuntimeError('Agent tool-result JSON byte budget exhausted')
                    tool_event['result'] = value
                    deadline()
                    # status is omitted for cross-model compatibility; execution
                    # errors abort the run and are never repaired/replayed here.
                    results.append({'toolResult': {'toolUseId': call['toolUseId'], 'content': [{'json': value}]}})
                messages.append({'role': 'user', 'content': results})
                if self.stop_after_required_tools and self.mcp.complete:
                    record.update(state='required_tools_completed')
                    break
            else:
                raise RuntimeError('Agent model-request budget exhausted')
            record['elapsed_s'] = self.clock()-start
            return record
        except Exception as exc:
            record.update(state='failed', error_type=type(exc).__name__, elapsed_s=self.clock()-start)
            raise AgentExecutionError('Agent run failed; verify postconditions and cleanup independently', record) from exc
