"""Bounded local stdio binding to this repository's registered ONOS MCP server.

No executable or shell command is accepted from model output or configuration.
Optional MCP resources/sampling are not advertised. A timed-out mutation is never
replayed as a tool call: the caller must fail the run and independently verify
cleanup. Compensating EOF cleanup can repeat owned DELETE requests.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading
import time

from .mcp import McpToolAdapter
from .mcp_onos_tools import (
    PROTOCOL_VERSION, SERVER_REVISION, TOOL_SCHEMA_SHA256, validate_call,
)
from ..utils.hashing import stable_digest


class McpProtocolError(RuntimeError):
    pass


class McpToolExecutionError(RuntimeError):
    pass


class OnosStdioMcpAdapter(McpToolAdapter):
    MAX_FRAME_BYTES = 1024*1024

    def __init__(self, *, ct_origin: str, ca_origin: str, ca_file: str,
                 username: str, password: str, timeout_s: float = 30,
                 controller_timeout_s: float = 2, allow_external: bool = False):
        super().__init__(revision=SERVER_REVISION, allow_external=allow_external)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (timeout_s, controller_timeout_s)):
            raise ValueError('Finite positive timeouts are required')
        if timeout_s < 9*controller_timeout_s:
            raise ValueError('RPC timeout must cover bounded owned-rule cleanup')
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._next_id = 0
        self._buffer = b''
        self._closed = False
        self.events = []
        self._stderr = tempfile.TemporaryFile()
        config = {'ct_origin': ct_origin, 'ca_origin': ca_origin, 'ca_file': str(Path(ca_file).resolve()),
                  'username': username, 'controller_timeout_s': controller_timeout_s}
        # Do not forward Bedrock/AWS credentials or SSH-related environment.
        environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LC_ALL': 'C',
                       'PYTHONPATH': str(Path(__file__).resolve().parents[2]),
                       'IL_MCP_ONOS_CONFIG': json.dumps(config), 'IL_MCP_ONOS_PASSWORD': password}
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-m', 'intent_leakage.adapters.mcp_onos_server'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                env=environment, start_new_session=True,
            )
        except BaseException:
            self._stderr.close()
            raise
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        try:
            initialized = self._rpc('initialize', {'protocolVersion': PROTOCOL_VERSION,
                'capabilities': {}, 'clientInfo': {'name': 'intent-leakage', 'version': '1'}})
            if initialized.get('protocolVersion') != PROTOCOL_VERSION or initialized.get('serverInfo', {}).get('name') != SERVER_REVISION:
                raise McpProtocolError('Unexpected MCP protocol/server revision')
            if 'tools' not in initialized.get('capabilities', {}):
                raise McpProtocolError('MCP server has no tools capability')
            self._notification('notifications/initialized')
            self._tools = self._rpc('tools/list', {})['tools']
            if stable_digest(self._tools) != TOOL_SCHEMA_SHA256:
                raise McpProtocolError('Frozen MCP tool schema mismatch')
        except BaseException:
            try:self.close()
            except Exception:pass
            raise

    def _write(self, message, deadline):
        encoded = json.dumps(message, separators=(',', ':'), allow_nan=False).encode()+b'\n'
        if len(encoded) > self.MAX_FRAME_BYTES:
            raise McpProtocolError('Request exceeds MCP frame limit')
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdin, selectors.EVENT_WRITE)
            while encoded:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('MCP write deadline exceeded')
                try:count = os.write(self._process.stdin.fileno(), encoded[:16384])
                except BlockingIOError:continue
                encoded = encoded[count:]

    def _read(self, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b'\n' not in self._buffer:
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('MCP response deadline exceeded')
                try:chunk = os.read(self._process.stdout.fileno(), 65536)
                except BlockingIOError:continue
                if not chunk:
                    raise McpProtocolError('MCP server closed its output')
                self._buffer += chunk
                if len(self._buffer) > self.MAX_FRAME_BYTES:
                    raise McpProtocolError('Response exceeds MCP frame limit')
            line, self._buffer = self._buffer.split(b'\n', 1)
        try:return json.loads(line)
        except (ValueError, UnicodeError) as exc:raise McpProtocolError('Invalid MCP response') from exc

    def _notification(self, method):
        self._write({'jsonrpc': '2.0', 'method': method}, time.monotonic()+self.timeout_s)

    def _rpc(self, method, params):
        with self._lock:
            if self._closed:
                raise McpProtocolError('MCP session is closed')
            if self._next_id >= 512:
                raise McpProtocolError('MCP session request budget exhausted')
            self._next_id += 1
            start = time.monotonic()
            event = {'method': method, 'request_id': self._next_id, 'outcome': 'pending'}
            if method == 'tools/call':event['tool_name'] = params.get('name')
            self.events.append(event)
            try:
                self._write({'jsonrpc': '2.0', 'id': self._next_id, 'method': method, 'params': params}, start+self.timeout_s)
                response = self._read(start+self.timeout_s)
                event['response_sha256'] = stable_digest(response)
                if not isinstance(response, dict) or response.get('jsonrpc') != '2.0' or type(response.get('id')) is not int or response['id'] != self._next_id:
                    raise McpProtocolError('MCP response identity mismatch')
                if 'error' in response or not isinstance(response.get('result'), dict):
                    raise McpProtocolError('MCP request failed')
                event['outcome'] = 'tool_error' if response['result'].get('isError') else 'result'
                return response['result']
            except BaseException as exc:
                event['outcome'] = type(exc).__name__
                raise
            finally:
                event['elapsed_s'] = time.monotonic()-start

    def list_tools(self):
        return json.loads(json.dumps(self._tools))

    def call_tool(self, name, arguments):
        validate_call(name, arguments)
        try:
            result = self._rpc('tools/call', {'name': name, 'arguments': arguments})
        except BaseException:
            # An interrupted mutation has an unknown outcome. Close the session;
            # never accept a late response as the next request's answer.
            try:self.close()
            except Exception:pass
            raise
        if result.get('isError') is not False:
            raise McpToolExecutionError('Controller tool reported an execution error')
        if not isinstance(result.get('structuredContent'), dict):
            raise McpProtocolError('Expected structured controller result')
        return result['structuredContent']

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()  # EOF triggers server-owned rule cleanup.
            try:code = self._process.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:self._process.kill();self._process.wait(timeout=2)
                raise McpProtocolError('MCP server cleanup deadline exceeded')
            if code:
                raise McpProtocolError('MCP server failed; independently verify cleanup')
        finally:
            self._process.stdout.close();self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
