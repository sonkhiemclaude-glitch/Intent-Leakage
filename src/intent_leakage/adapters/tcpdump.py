"""Bounded, idempotently stopped tcpdump capture with explicit readiness.

Runs only after the external execution gate. Immediate mode avoids losing short
transactions in the libpcap buffer. No shell, global process cleanup or interface
configuration is performed. Caller owns the lifecycle and capture namespace.
"""
from __future__ import annotations
import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from .base import require_external_execution
from ..orchestration.capture import CaptureArtifact, CaptureSpec

@dataclass(frozen=True)
class TcpdumpResult:
    artifact: CaptureArtifact
    packets_captured: int | None
    packets_dropped_by_kernel: int | None
    exit_code: int
    stderr: str

class TcpdumpCapture:
    def __init__(self, *, allow_external: bool=False, executable='tcpdump', readiness_timeout_s=5, drain_s=.2):
        require_external_execution(allow_external=allow_external)
        if readiness_timeout_s<=0 or drain_s<0:raise ValueError('Invalid capture timing')
        self.executable=executable;self.readiness_timeout_s=readiness_timeout_s;self.drain_s=drain_s
        self.process=None;self.output=None;self._prefix=b'';self._result=None

    def start(self,spec: CaptureSpec,output: str|Path):
        spec.validate()
        if self.process is not None or self._result is not None:raise RuntimeError('Capture instances are single use')
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True);self.output=output
        self.process=subprocess.Popen([self.executable,'--immediate-mode','-U','-s','0','-i',spec.interface,'-w',str(output),spec.capture_filter],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        deadline=time.monotonic()+self.readiness_timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stderr,selectors.EVENT_READ)
            while b'listening on' not in self._prefix:
                if self.process.poll() is not None:raise RuntimeError('Capture exited before readiness')
                if time.monotonic()>=deadline:raise TimeoutError('Capture readiness timed out')
                if selector.select(min(.1,max(0,deadline-time.monotonic()))):
                    data=os.read(self.process.stderr.fileno(),4096)
                    if not data:raise RuntimeError('Capture stderr closed before readiness')
                    self._prefix+=data

    def stop_and_store(self) -> TcpdumpResult:
        if self._result is not None:return self._result
        if self.process is None:raise RuntimeError('Capture was never started')
        if self.process.poll() is None:
            time.sleep(self.drain_s)
            try:self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:pass
        try:_,tail=self.process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill();_,tail=self.process.communicate(timeout=5)
        text=(self._prefix+tail).decode(errors='replace')
        def count(label):
            match=re.search(r'(\d+) '+label,text);return int(match.group(1)) if match else None
        self._result=TcpdumpResult(CaptureArtifact.from_file(self.output),count('packets captured'),count('packets dropped by kernel'),self.process.returncode,text)
        return self._result
