import json,threading,tempfile
from pathlib import Path
import pytest
from intent_leakage.adapters.bedrock import BedrockAdapter
from intent_leakage.adapters.bedrock_unix import ProviderServer,UnixBedrockAdapter,ProviderBridgeError,pilot_config


class Client:
    def __init__(self):self.calls=[]
    def converse(self,**request):
        self.calls.append(request)
        return {'stopReason':'end_turn','output':{'message':{'role':'assistant','content':[{'text':'done'}]}},'usage':{'inputTokens':10,'outputTokens':1}}
    def close(self):pass


@pytest.fixture
def short_path():
    with tempfile.TemporaryDirectory(prefix='il-ipc-',dir='/tmp') as folder:
        yield Path(folder)


def test_ipc_freezes_model_and_enforces_shared_quota(short_path):
    tmp_path=short_path
    client=Client();model=BedrockAdapter(pilot_config(),allow_external=True,client_factory=lambda:client)
    server=ProviderServer(tmp_path/'s',model,tmp_path/'stats',max_requests=1)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        proxy=UnixBedrockAdapter(tmp_path/'s',allow_external=True)
        request={'messages':[{'role':'user','content':[{'text':'check'}]}]}
        assert proxy.invoke(request)['stopReason']=='end_turn'
        with pytest.raises(ProviderBridgeError):proxy.invoke(request)
        assert len(client.calls)==1
        assert client.calls[0]['modelId']=='amazon.nova-micro-v1:0'
        assert client.calls[0]['inferenceConfig']['maxTokens']==512
        assert json.loads((tmp_path/'stats').read_text())['input_tokens']==10
        with pytest.raises(ValueError):proxy.invoke({**request,'modelId':'another'})
    finally:server.shutdown();server.server_close();thread.join();model.close()


def test_ipc_requires_external_gate(tmp_path):
    with pytest.raises(Exception):UnixBedrockAdapter(tmp_path/'s')


def test_provider_error_is_sanitized_and_retained(short_path,monkeypatch):
    class Denied(Exception):
        response={'Error':{'Code':'AccessDeniedException','Message':'invalid dummy-secret'}}
    class DeniedClient(Client):
        def converse(self,**request):raise Denied()
    monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK','dummy-secret')
    model=BedrockAdapter(pilot_config(),allow_external=True,client_factory=DeniedClient)
    server=ProviderServer(short_path/'s',model,short_path/'stats')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        proxy=UnixBedrockAdapter(short_path/'s',allow_external=True)
        with pytest.raises(ProviderBridgeError):proxy.invoke({'messages':[{'role':'user','content':[{'text':'check'}]}]})
        text=(short_path/'stats').read_text();assert 'dummy-secret' not in text
        assert json.loads(text)['last_error']['aws_code']=='AccessDeniedException'
    finally:server.shutdown();server.server_close();thread.join();model.close()
