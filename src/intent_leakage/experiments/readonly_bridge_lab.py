"""Single co-located HTTPS read via loopback mTLS cells; no mutation/LLM."""
from __future__ import annotations
import hashlib,json,time
from dataclasses import asdict
from ..adapters.base import require_external_execution
from ..adapters.onos_rest import OnosRestAdapter
from ..adapters.readonly_controller_bridge import ReadOnlyControllerBridge
from ..adapters.local_shaped_transport import LocalTransportProfile
from .local_transport_lab import authenticated_socketpair,make_endpoints


def run_readonly_bridge(config,*,profile,allow_external=False,resource='flows'):
    require_external_execution(allow_external)
    if profile.channel not in ('CT','CA') or profile.max_message_bytes<131072:raise ValueError('engineering runner requires explicit >=128KiB message profile')
    if resource not in ('flows','hosts','devices','links'):raise ValueError('invalid resource')
    if set(config)!={'origin','username','password','ca_file','timeout_s'}:raise ValueError('unexpected connection configuration fields')
    if not 0<config['timeout_s']<=2:raise ValueError('bounded socket timeout must be <=2s')
    observed=[]
    adapter=OnosRestAdapter(config['origin'],username=config['username'],password=config['password'],ca_file=config['ca_file'],
                            allow_external=True,timeout_s=config['timeout_s'],application_observer=observed.append)
    handler=ReadOnlyControllerBridge(adapter,allow_external=True,max_response_bytes=profile.max_message_bytes,resources=(resource,))
    report={'purpose':'read_only_bridge_engineering_qualification','success':False,'profile':asdict(profile),'profile_sha256':profile.sha256,
            'resource':resource,'remote_mutations_supported':False,'independent_dataplane_verified':False,
            'TLS_IP_trace_constancy_verified':False,'wire_cost_measured':False,'formal_profile_selected':False,
            'socket_timeout_is_absolute_exchange_deadline':False,'metrics':[]}
    with authenticated_socketpair(loopback_tcp=True) as pair:
        report['flow_manifest']={'endpoints':[{'client_ip':pair[0].getsockname()[0],'client_port':pair[0].getsockname()[1],
                                             'server_ip':pair[0].getpeername()[0],'server_port':pair[0].getpeername()[1],'channel':profile.channel}],
                                 'capture_interface':'lo on Linux / lo0 on macOS','representation':'loopback TLS tunnel only; exclude backend HTTPS','handshake_completed_before_returning_pair':True}
        agent,bridge=make_endpoints(pair,profile=profile,lab=handler,lead_s=.5,allow_loopback_tcp=True)
        budget=profile.rounds*profile.period_s+3
        try:
            future=agent.submit(json.dumps({'op':'read','resource':resource},separators=(',',':')).encode());agent.start();bridge.start()
            body=future.result(timeout=budget)
            metrics=[agent.wait(budget),bridge.wait(budget)];report['metrics']=metrics
            if not all(m['success'] for m in metrics):raise RuntimeError('envelope did not complete')
            if len(observed)!=1 or observed[0].response_body!=body or len(handler.calls)!=1:raise RuntimeError('body/handler causality mismatch')
            report.update(success=True,response_body_bytes=len(body),response_body_sha256=hashlib.sha256(body).hexdigest(),
                          exact_https_body_reassembled=True,completed_http_reads=1)
        except Exception as exc:
            report.update(error_type=type(exc).__name__,error_code=getattr(exc,'code',None))
        finally:
            agent.close();bridge.close()
            report['metrics']=[agent.wait(3),bridge.wait(3)]
            report['handler_calls']=handler.calls
            report['application_observation_errors']=adapter.application_observation_errors
            if not all(m['success'] for m in report['metrics']):report['success']=False
    return report
