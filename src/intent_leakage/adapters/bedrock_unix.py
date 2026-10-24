"""Bounded local IPC to a host-side Bedrock client; isolated testbed has no egress."""
from dataclasses import asdict
import argparse,json,os,socket,socketserver,time
from pathlib import Path
from .base import require_external_execution
from .bedrock import BedrockAdapter,BedrockConfig
from ..utils.hashing import stable_digest

MAX_REQUEST=68*1024
MAX_RESPONSE=512*1024


def pilot_config():
    return BedrockConfig('amazon.nova-micro-v1:0','us-east-1',False,{'temperature':0.0},512,
        {'mode':'standard','total_max_attempts':1},'reuse',connect_timeout_s=5,read_timeout_s=20)


def receive(stream,limit):
    data=stream.readline(limit+1)
    if len(data)>limit or not data.endswith(b'\n'):raise ValueError('Invalid or oversized IPC frame')
    return json.loads(data)


class ProviderBridgeError(RuntimeError):pass


class UnixBedrockAdapter:
    def __init__(self,path,*,allow_external=False):
        require_external_execution(allow_external)
        self.path=str(path);self.config=pilot_config()
    def invoke(self,request):
        # Validate allowed request fields and the frozen inference configuration.
        BedrockAdapter(self.config,allow_external=True)._request(request)
        payload=json.dumps({'request':request,'config_sha256':stable_digest(asdict(self.config))},separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(payload)>MAX_REQUEST:raise ValueError('IPC request limit exceeded')
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
            connection.settimeout(30);connection.connect(self.path);connection.sendall(payload)
            with connection.makefile('rb') as stream:response=receive(stream,MAX_RESPONSE)
        if set(response)!={'response'}:raise ProviderBridgeError(response.get('error_type','Invalid provider response'))
        return response['response']
    def close(self):pass


class ProviderServer(socketserver.UnixStreamServer):
    allow_reuse_address=False
    def __init__(self,path,model,stats,max_requests=72,max_request_bytes=1024*1024):
        self.model=model;self.stats=Path(stats);self.max_requests=max_requests;self.max_request_bytes=max_request_bytes
        self.counters=json.loads(self.stats.read_text()) if self.stats.exists() else {'requests':0,'request_bytes':0,'input_tokens':0,'output_tokens':0,'failures':0}
        super().__init__(str(path),ProviderHandler)
        os.chmod(path,0o600)
    def save(self):
        temp=self.stats.with_suffix('.tmp');temp.write_text(json.dumps(self.counters));temp.replace(self.stats)


class ProviderHandler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(35)
        try:
            frame=receive(self.rfile,MAX_REQUEST)
            if set(frame)!={'request','config_sha256'} or frame['config_sha256']!=stable_digest(asdict(self.server.model.config)):
                raise ValueError('Unexpected provider configuration')
            self.server.model._request(frame['request'])
            size=len(json.dumps(frame,separators=(',',':'),allow_nan=False).encode())
            counters=self.server.counters
            if counters['requests']>=self.server.max_requests or counters['request_bytes']+size>self.server.max_request_bytes:
                raise RuntimeError('Provider quota exhausted')
            counters['requests']+=1;counters['request_bytes']+=size;self.server.save()
            result=self.server.model.invoke(frame['request'])
            counters['input_tokens']+=result.get('usage',{}).get('inputTokens',0)
            counters['output_tokens']+=result.get('usage',{}).get('outputTokens',0);self.server.save()
            payload=json.dumps({'response':result},separators=(',',':'),allow_nan=False).encode()+b'\n'
            if len(payload)>MAX_RESPONSE:raise ValueError('IPC response limit exceeded')
        except Exception as exc:
            self.server.counters['failures']+=1
            detail=getattr(exc,'response',{}).get('Error',{})
            message=str(detail.get('Message',''))[:1000]
            secret=os.environ.get('AWS_BEARER_TOKEN_BEDROCK','')
            if secret:message=message.replace(secret,'<redacted>')
            self.server.counters['last_error']={'error_type':type(exc).__name__,'aws_code':detail.get('Code'),'aws_message':message}
            self.server.save()
            # Never echo SDK error text, headers, request or credentials.
            payload=json.dumps({'error_type':type(exc).__name__}).encode()+b'\n'
        try:self.wfile.write(payload)
        except (OSError,TimeoutError):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True);p.add_argument('--stats',type=Path,required=True);a=p.parse_args()
    token=a.credential_file.read_text().strip()
    if not token:raise ValueError('Empty provider credential')
    os.environ['AWS_BEARER_TOKEN_BEDROCK']=token
    model=BedrockAdapter(pilot_config(),allow_external=True)
    try:
        with ProviderServer(a.socket,model,a.stats) as server:server.serve_forever(poll_interval=.2)
    finally:
        model.close();a.socket.unlink(missing_ok=True)


if __name__=='__main__':main()
