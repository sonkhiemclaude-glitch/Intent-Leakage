"""Spawn-isolated LOCAL pacing comparison; no controller/EC2/network service.

Session/pacing code and application handler can occupy separate processes.
All child processes are bounded and owned by this experiment, never global
process names. Local state only; no arbitrary remote mutation support.
"""
from __future__ import annotations
import base64,json,math,multiprocessing as mp,os,signal,time
from .local_transport_lab import LocalPolicyLab,successful_case
from ..adapters.local_shaped_transport import TransportCancelled,TransportError


def burn_cpu(seconds):
    deadline=time.thread_time()+seconds; value=1
    while time.thread_time()<deadline:value=(value*1664525+1013904223)&0xffffffff


class CpuPolicyLab(LocalPolicyLab):
    def __init__(self,cpu_s):super().__init__();self.cpu_s=cpu_s
    def handle(self,payload,context):
        burn_cpu(self.cpu_s)
        return super().handle(payload,context)


def _backend_worker(pipe,cancel,commit_lock,cpu_s):
    lab=LocalPolicyLab()
    class SharedContext:
        def __init__(self,deadline):self.deadline=deadline
        def commit(self,action):
            with commit_lock:
                if cancel.is_set() or time.monotonic()>=self.deadline:raise TransportCancelled('local backend deadline/cancellation')
                return action()
    pipe.send_bytes(json.dumps({'ready':True,'pid':os.getpid(),'state':lab.state}).encode())
    try:
        while True:
            request=json.loads(pipe.recv_bytes(2*1024*1024));rid=request['id'];command=request['command']
            try:
                if command=='handle':
                    payload=base64.b64decode(request['payload'],validate=True)
                    if len(payload)>65536 or not math.isfinite(request['deadline']):raise ValueError('invalid local IPC request')
                    started=time.monotonic();burn_cpu(cpu_s)
                    answer=lab.handle(payload,SharedContext(request['deadline']))
                    result={'id':rid,'ok':True,'payload':base64.b64encode(answer).decode(),'handler_cpu_window_s':time.monotonic()-started}
                elif command=='cleanup':
                    lab.cleanup();result={'id':rid,'ok':True,'state':lab.state,'calls':lab.calls,'cleanup_count':lab.cleanup_count}
                else:raise ValueError('unregistered IPC command')
            except Exception as exc:result={'id':rid,'ok':False,'error_type':type(exc).__name__}
            pipe.send_bytes(json.dumps(result).encode())
            if command=='cleanup':break
    finally:pipe.close()


class ProcessPolicyLab:
    def __init__(self,cpu_s=0.,timeout_s=5.):
        ctx=mp.get_context('spawn');self._parent,child=ctx.Pipe();self._cancel=ctx.Event();self._commit_lock=ctx.RLock()
        import threading
        self._rpc_lock=threading.RLock();self._timeout=timeout_s;self._serial=0;self._closed=False;self._broken=False
        self._process=ctx.Process(target=_backend_worker,args=(child,self._cancel,self._commit_lock,cpu_s))
        self._process.start();child.close()
        if not self._parent.poll(15):self._kill();raise TimeoutError('local backend startup')
        ready=json.loads(self._parent.recv_bytes(2*1024*1024));self.pid=ready['pid'];self.state=ready['state'];self.calls=[];self.cleanup_count=0

    def _kill(self):
        if self._process.is_alive():self._process.terminate()
        self._process.join(2)
        if self._process.is_alive():self._process.kill();self._process.join(2)
        self._parent.close();self._closed=True

    def _rpc(self,command,**fields):
        with self._rpc_lock:
            if self._closed or self._broken:raise TransportError('local backend unavailable')
            self._serial+=1;rid=self._serial
            try:
                self._parent.send_bytes(json.dumps({'id':rid,'command':command,**fields}).encode())
                if not self._parent.poll(self._timeout):raise TimeoutError('local backend RPC deadline')
                response=json.loads(self._parent.recv_bytes(2*1024*1024))
                if response.get('id')!=rid or not response.get('ok'):raise TransportError('local backend rejected operation')
                return response
            except Exception:
                self._broken=True;raise

    def handle(self,payload,context):
        if len(payload)>65536:raise ValueError('IPC application bound')
        deadline=context.deadline
        response=self._rpc('handle',payload=base64.b64encode(payload).decode(),deadline=deadline)
        answer=base64.b64decode(response['payload'],validate=True)
        return context.commit(lambda:answer)

    def cleanup(self):
        if self._closed:return
        # Prevent later commits BEFORE waiting for a slow IPC handler.
        with self._commit_lock:self._cancel.set()
        try:
            result=self._rpc('cleanup');self.state=result['state'];self.calls=result['calls'];self.cleanup_count=result['cleanup_count']
            self._process.join(2)
            if self._process.is_alive():raise TimeoutError('backend shutdown')
        finally:self._kill()


def _session_worker(send,mode,spec):
    os.setsid();backend=None
    try:
        send.send_bytes(json.dumps({'kind':'started','pid':os.getpid(),'pgid':os.getpgrp()}).encode())
        backend=ProcessPolicyLab(spec['handler_cpu_s']) if mode=='process_handler' else CpuPolicyLab(spec['handler_cpu_s'])
        result=successful_case(cell_bytes=spec['cell_bytes'],rounds=spec['rounds'],period_s=spec['period_s'],
                               value='x'*spec['value_bytes'],lab=backend,lead_s=spec['lead_s'])
        result.update(success=True,kind='result',pacing_pid=os.getpid(),handler_pid=backend.pid if isinstance(backend,ProcessPolicyLab) else os.getpid())
    except Exception as exc:
        result={'kind':'result','success':False,'error_type':type(exc).__name__,'metrics':getattr(exc,'metrics',[]),'pacing_pid':os.getpid()}
    finally:
        if isinstance(backend,ProcessPolicyLab) and not backend._closed:
            try:backend.cleanup()
            except Exception:result['backend_cleanup_unverified']=True
    try:send.send_bytes(json.dumps(result).encode())
    finally:send.close()


def _load_worker(stop):
    while not stop.is_set():burn_cpu(.02)


def isolated_case(mode,spec,*,loaded=False,timeout_s=30):
    if mode not in ('thread_handler','process_handler'):raise ValueError('invalid comparison mode')
    ctx=mp.get_context('spawn');receive,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=_session_worker,args=(send,mode,spec));load=None;stop=ctx.Event()
    if loaded:load=ctx.Process(target=_load_worker,args=(stop,));load.start()
    process.start();send.close();started=time.monotonic();group_confirmed=False;result=None
    try:
        while time.monotonic()-started<timeout_s:
            if receive.poll(.1):
                message=json.loads(receive.recv_bytes(4*1024*1024))
                if message.get('kind')=='started':group_confirmed=message['pgid']==message['pid']==process.pid
                else:result=message;break
            if not process.is_alive() and not receive.poll():break
        if result is None:result={'success':False,'error_type':'SessionDeadlineOrExit','metrics':[]}
    finally:
        stop.set()
        if load:
            load.join(2)
            if load.is_alive():load.terminate();load.join(2)
        process.join(2)
        if process.is_alive():
            if group_confirmed and os.getpgid(process.pid)==process.pid:os.killpg(process.pid,signal.SIGTERM)
            else:process.terminate()
            process.join(2)
            if process.is_alive():
                if group_confirmed and os.getpgid(process.pid)==process.pid:os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.join(2)
        receive.close()
    result.update(mode=mode,loaded=loaded,specification=spec,parent_elapsed_s=time.monotonic()-started)
    return result
