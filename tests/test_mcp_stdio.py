"""Actual stdio subprocess + local TLS API fixture, not a live ONOS experiment."""
import datetime
import io
import ipaddress
import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from intent_leakage.adapters.base import ExternalExecutionDisabledError
from intent_leakage.adapters.mcp_onos_server import serve
from intent_leakage.adapters.mcp_onos_tools import list_tools, TOOL_SCHEMA_SHA256
from intent_leakage.adapters.mcp_stdio import (
    OnosStdioMcpAdapter, McpToolExecutionError, McpProtocolError,
)
from intent_leakage.utils.hashing import stable_digest

HOST = '00:00:00:00:00:01/None'
DEVICE = 'of:0000000000000001'


@pytest.fixture
def controller(tmp_path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, 'MCP test fixture')])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=1))
            .not_valid_after(now+datetime.timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]), critical=False)
            .sign(key, hashes.SHA256()))
    certificate = tmp_path/'cert.pem';private = tmp_path/'key.pem'
    certificate.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    state = {'flows': {}, 'calls': [], 'next_id': 100, 'fail_post': False, 'fail_delete': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):pass

        def send_json(self, value, code=200):
            body = json.dumps(value).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length', str(len(body)));self.end_headers();self.wfile.write(body)

        def do_GET(self):
            state['calls'].append(('GET', self.server.server_port, self.path))
            if self.path == '/onos/v1/hosts':
                self.send_json({'hosts': [{'id': HOST, 'mac': HOST.split('/')[0], 'locations': [{'elementId': DEVICE}]}]})
            elif self.path == '/onos/v1/flows':self.send_json({'flows':list(state['flows'].values())})
            else:self.send_json({'flows':[f for key,f in state['flows'].items() if key==self.path.rsplit('/',1)[-1]]})

        def do_POST(self):
            state['calls'].append(('POST', self.server.server_port, self.path))
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if state['fail_post']:self.send_json({},500);return
            entries=[]
            for rule in payload['flows']:
                flow=str(state['next_id']);state['next_id']+=1
                state['flows'][flow]={**rule,'id':flow,'state':'ADDED'}
                entries.append({'deviceId':rule['deviceId'],'flowId':flow})
            self.send_json({'flows':entries})

        def do_DELETE(self):
            state['calls'].append(('DELETE', self.server.server_port, self.path))
            if state['fail_delete']:self.send_json({},500);return
            state['flows'].pop(self.path.rsplit('/',1)[-1],None);self.send_json({})

    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version=context.maximum_version=ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certificate,private)
    servers=[]
    for _ in range(2):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        server.socket=context.wrap_socket(server.socket,server_side=True)
        threading.Thread(target=server.serve_forever,daemon=True).start();servers.append(server)
    config={'ct_origin':f'https://127.0.0.1:{servers[0].server_port}',
            'ca_origin':f'https://127.0.0.1:{servers[1].server_port}',
            'ca_file':str(certificate),'username':'fixture','password':'fixture'}
    yield config,state,servers
    for server in servers:server.shutdown();server.server_close()


def test_external_gate_precedes_subprocess_or_certificate_access(monkeypatch):
    import intent_leakage.adapters.mcp_stdio as module
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**kw:pytest.fail('unexpected process'))
    with pytest.raises(ExternalExecutionDisabledError):
        OnosStdioMcpAdapter(ct_origin='x',ca_origin='x',ca_file='missing',username='x',password='x')


def test_real_stdio_tls_tool_roundtrip_and_eof_owned_cleanup(controller,monkeypatch):
    import intent_leakage.adapters.mcp_stdio as module
    config,state,servers=controller
    state['flows']['999']={'id':'999','appId':'another-project'}
    monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK','fixture-must-not-be-forwarded')
    original=module.subprocess.Popen
    def spawn(*args,**kwargs):
        assert not any(key.startswith('AWS') or key.startswith('SSH') for key in kwargs['env'])
        assert 'IL_MCP_ONOS_PASSWORD' in kwargs['env']
        assert kwargs.get('shell') is not True
        return original(*args,**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    with OnosStdioMcpAdapter(**config,allow_external=True) as client:
        assert stable_digest(client.list_tools())==TOOL_SCHEMA_SHA256
        assert client.call_tool('lookup_hosts',{'host_ids':[HOST]})['hosts'][0]['id']==HOST
        rules=client.call_tool('quarantine_hosts',{'host_ids':[HOST]})['rules']
        assert len(rules)==2 and len(state['flows'])==3
        result=client.call_tool('audit_policies',{'policies':[rules[0]]})
        assert len(result['policies'][0]['flows'])==1
        assert len(client.call_tool('read_flows',{})['flows'])==3
        assert all(port==servers[1].server_port for method,port,path in state['calls'] if method=='GET' and '/flows' in path)
    assert set(state['flows'])=={'999'}  # EOF cleanup never removes a foreign rule.
    assert client._process.poll()==0


def test_invalid_tool_arguments_never_reach_controller(controller):
    config,state,_=controller
    with OnosStdioMcpAdapter(**config,allow_external=True) as client:
        for name,args in [('shell',{'command':'anything'}),('read_flows',{'path':'/other'}),
                          ('quarantine_hosts',{'host_ids':[HOST,HOST]}),
                          ('quarantine_hosts',{'host_ids':[HOST],'extra':True})]:
            with pytest.raises(ValueError):client.call_tool(name,args)
        assert state['calls']==[]


def test_controller_error_is_not_retried_and_remains_an_error(controller):
    config,state,_=controller;state['fail_post']=True
    with OnosStdioMcpAdapter(**config,allow_external=True) as client:
        with pytest.raises(McpToolExecutionError):client.call_tool('quarantine_hosts',{'host_ids':[HOST]})
        assert sum(method=='POST' for method,_,_ in state['calls'])==1
        assert client.events[-1]['outcome']=='tool_error'


def test_eof_cleanup_failure_cannot_be_reported_as_success(controller):
    config,state,_=controller
    client=OnosStdioMcpAdapter(**config,allow_external=True)
    client.call_tool('quarantine_hosts',{'host_ids':[HOST]})
    state['fail_delete']=True
    with pytest.raises(McpProtocolError,match='verify cleanup'):client.close()
    assert len(state['flows'])==2 and client._process.poll()==2
    assert sum(method=='DELETE' for method,_,_ in state['calls'])==1


def test_tools_unavailable_before_initialized_notification():
    messages=[{'jsonrpc':'2.0','id':1,'method':'tools/list','params':{}},
              {'jsonrpc':'2.0','id':2,'method':'initialize','params':{'protocolVersion':'2025-06-18'}},
              {'jsonrpc':'2.0','id':3,'method':'tools/list','params':{}}]
    source=io.BytesIO(b''.join(json.dumps(x).encode()+b'\n' for x in messages));sink=io.BytesIO()
    serve(None,None,source,sink)
    responses=[json.loads(x) for x in sink.getvalue().splitlines()]
    assert 'error' in responses[0] and 'result' in responses[1] and 'error' in responses[2]


def test_jsonrpc_parse_and_invalid_envelope_have_distinct_standard_errors():
    source=io.BytesIO(b'{broken}\n'+json.dumps({'jsonrpc':'1.0','id':7,'method':'ping'}).encode()+b'\n')
    sink=io.BytesIO();serve(None,None,source,sink)
    errors=[json.loads(x)['error']['code'] for x in sink.getvalue().splitlines()]
    assert errors==[-32700,-32600]


@pytest.mark.realtime
def test_mcp_timeout_closes_session_and_does_not_reissue_mutation(controller,monkeypatch):
    import sys,time
    import intent_leakage.adapters.mcp_stdio as module
    config,_,_=controller;original=module.subprocess.Popen
    program='''import json,sys,time
from intent_leakage.adapters.mcp_onos_tools import list_tools,PROTOCOL_VERSION,SERVER_REVISION
for line in sys.stdin:
 r=json.loads(line)
 if 'id' not in r:continue
 if r['method']=='initialize':v={'protocolVersion':PROTOCOL_VERSION,'capabilities':{'tools':{}},'serverInfo':{'name':SERVER_REVISION}}
 elif r['method']=='tools/list':v={'tools':list_tools()}
 else:time.sleep(10);v={}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':v}),flush=True)
'''
    def spawn(*args,**kwargs):return original([sys.executable,'-u','-c',program],**kwargs)
    monkeypatch.setattr(module.subprocess,'Popen',spawn)
    client=OnosStdioMcpAdapter(**config,allow_external=True,timeout_s=.4,controller_timeout_s=.01)
    started=time.monotonic()
    with pytest.raises(TimeoutError):client.call_tool('quarantine_hosts',{'host_ids':[HOST]})
    assert time.monotonic()-started<3
    assert client._process.poll() is not None and client._closed
    assert len([e for e in client.events if e['method']=='tools/call'])==1
    assert client.events[-1]['outcome']=='TimeoutError'
    with pytest.raises(McpProtocolError):client.call_tool('read_flows',{})


def test_agent_sdk_double_through_actual_stdio_tls_controller_binding(controller):
    from intent_leakage.adapters.bedrock import BedrockAdapter,BedrockConfig
    from intent_leakage.orchestration.tool_agent import BedrockOnosToolAgent
    config,state,_=controller
    replies=iter([
        {'stopReason':'tool_use','output':{'message':{'role':'assistant','content':[{'toolUse':{
            'toolUseId':'native1','name':'quarantine_hosts','input':{'host_ids':[HOST]}}}]}}},
        {'stopReason':'end_turn','output':{'message':{'role':'assistant','content':[{'text':'Requested.'}]}}},
    ])
    class Client:
        def converse(self,**request):return next(replies)
        def close(self):pass
    client=Client()
    model=BedrockAdapter(BedrockConfig('amazon.nova-micro-v1:0','us-east-1',False,
        {'temperature':0},128,{'total_max_attempts':1},'reuse'),allow_external=True,client_factory=lambda:client)
    with OnosStdioMcpAdapter(**config,allow_external=True) as mcp:
        agent=BedrockOnosToolAgent(model,mcp,system_prompt='Use registered lab tools.',allow_external=True)
        result=agent.run('Quarantine 00:00:00:00:00:01/None in the fixture.')
        assert len(state['flows'])==2 and result['tool_calls']==1
        assert result['state']=='model_completed' and result['fulfillment_verified'] is False
    assert state['flows']=={}
