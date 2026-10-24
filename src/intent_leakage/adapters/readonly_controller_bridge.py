"""Read-only real HTTPS controller binding for the local mTLS fixed-cell bridge.

No controller mutation is supported. Context checks bound admission/publication,
not the lifetime of an in-flight HTTP read. Socket timeouts are not an absolute
whole-exchange deadline; stalled/late reads are failures, never task successes.
AF_UNIX framing is co-located only; no constant TLS/IP trace claim is implied.
"""
from __future__ import annotations
import json,threading,time
from .base import require_external_execution
from .onos_rest import OnosRestAdapter
from .local_shaped_transport import EnvelopeOverflow,RequestContext


class ReadOnlyControllerBridge:
    remote_mutations_supported=False

    def __init__(self,adapter,*,allow_external=False,max_response_bytes=131072,resources=('flows',)):
        require_external_execution(allow_external)
        if not isinstance(adapter,OnosRestAdapter):raise ValueError('explicit HTTPS ONOS binding required')
        if type(max_response_bytes) is not int or not 1<=max_response_bytes<=1048576:raise ValueError('invalid response bound')
        if not resources or len(set(resources))!=len(resources) or any(r not in ('flows','hosts','devices','links') for r in resources):raise ValueError('invalid frozen resources')
        self.adapter=adapter;self.max_response_bytes=max_response_bytes;self.resources=frozenset(resources)
        self._lock=threading.Lock();self.calls=[]

    def handle(self,payload:bytes,context:RequestContext)->bytes:
        if type(payload) is not bytes or len(payload)>256:raise ValueError('invalid bounded read request')
        # Reject duplicate JSON keys, too: no ambiguous operation interpretation.
        def unique(pairs):
            result={}
            for k,v in pairs:
                if k in result:raise ValueError('duplicate request field')
                result[k]=v
            return result
        request=json.loads(payload,object_pairs_hook=unique)
        if not isinstance(request,dict) or set(request)!={'op','resource'} or request['op']!='read' or request['resource'] not in self.resources:
            raise ValueError('operation outside frozen read-only allowlist')
        with self._lock:
            context.check_active();started=time.monotonic();record={'resource':request['resource'],'started_at':started,'outcome':'inflight'};self.calls.append(record)
            try:
                body=self.adapter.read_application_body(request['resource'])
                context.check_active()
                if len(body)>self.max_response_bytes:raise EnvelopeOverflow('controller body exceeds frozen bridge limit',code='controller_response_size')
                if self.adapter.application_observation_errors:raise ValueError('incomplete application observation')
                record.update(outcome='returned',response_bytes=len(body));return body
            except Exception as exc:
                record.update(outcome='failed',error_type=type(exc).__name__);raise
            finally:record['finished_at']=time.monotonic()

    def cleanup(self):
        # There are no owned controller mutations to roll back for this binding.
        return None
