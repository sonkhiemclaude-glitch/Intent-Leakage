"""Registered local stdio MCP server. stdout contains JSON-RPC messages only."""
from __future__ import annotations

import json
import os
import sys

from .mcp_onos_tools import PROTOCOL_VERSION, SERVER_REVISION, dispatch, list_tools, validate_call
from .onos_rest import OnosRestAdapter

MAX_FRAME_BYTES = 1024*1024


def serve(ct, ca, source, sink):
    initialized = False
    ready = False
    for line in iter(lambda: source.readline(MAX_FRAME_BYTES+2), b''):
        request_id = None
        error_code, error_message = -32600, 'Invalid JSON-RPC request'
        try:
            if len(line) > MAX_FRAME_BYTES or not line.endswith(b'\n'):
                raise ValueError('Invalid MCP frame')
            error_code, error_message = -32700, 'Parse error'
            request = json.loads(line)
            error_code, error_message = -32600, 'Invalid JSON-RPC request'
            if not isinstance(request, dict) or request.get('jsonrpc') != '2.0':
                raise ValueError('Invalid JSON-RPC envelope')
            request_id = request.get('id')
            method = request.get('method')
            params = request.get('params', {})
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ValueError('Invalid method/parameters')
            error_code, error_message = -32602, 'Invalid MCP parameters'
            if method == 'notifications/initialized' and 'id' not in request:
                ready = initialized
                continue
            if 'id' not in request:
                continue  # Notifications have no response.
            if isinstance(request_id, bool) or not isinstance(request_id, (str, int)):
                raise ValueError('Invalid request ID')
            if method == 'initialize' and not initialized:
                if not isinstance(params.get('protocolVersion'), str):
                    raise ValueError('Protocol version required')
                result = {'protocolVersion': PROTOCOL_VERSION, 'capabilities': {'tools': {}},
                          'serverInfo': {'name': SERVER_REVISION, 'version': '1'}}
                initialized = True
            elif method == 'ping':
                result = {}
            elif not ready:
                raise ValueError('MCP session is not initialized')
            elif method == 'tools/list' and set(params) <= {'_meta'}:
                result = {'tools': list_tools()}
            elif method == 'tools/call' and 'name' in params and set(params) <= {'name', 'arguments', '_meta'}:
                name, arguments = params['name'], params.get('arguments', {})
                validate_call(name, arguments)
                try:
                    value = dispatch(ct, ca, name, arguments)
                    result = {'content': [{'type': 'text', 'text': json.dumps(value)}],
                              'structuredContent': value, 'isError': False}
                except Exception as exc:
                    result = {'content': [{'type': 'text', 'text': 'Controller tool failed: '+type(exc).__name__}],
                              'isError': True}
            else:
                response = {'jsonrpc': '2.0', 'id': request_id,
                            'error': {'code': -32601, 'message': 'Method not available'}}
                sink.write(json.dumps(response).encode()+b'\n');sink.flush()
                continue
            response = {'jsonrpc': '2.0', 'id': request_id, 'result': result}
        except (ValueError, TypeError, KeyError):
            response = {'jsonrpc': '2.0', 'id': request_id,
                        'error': {'code': error_code, 'message': error_message}}
        encoded = json.dumps(response, separators=(',', ':')).encode()+b'\n'
        if len(encoded) > MAX_FRAME_BYTES:
            encoded = json.dumps({'jsonrpc': '2.0', 'id': request_id,
                                  'error': {'code': -32603, 'message': 'Response exceeds frame limit'}}).encode()+b'\n'
        sink.write(encoded);sink.flush()


def main():
    # This module is the only production executable registered by the client.
    # Passwords arrive in a private child environment, never in process arguments.
    config = json.loads(os.environ['IL_MCP_ONOS_CONFIG'])
    ct = ca = None
    try:
        common = {'username': config['username'], 'password': os.environ['IL_MCP_ONOS_PASSWORD'],
                  'ca_file': config['ca_file'], 'timeout_s': config['controller_timeout_s'], 'allow_external': True}
        ct = OnosRestAdapter(config['ct_origin'], **common)
        ca = OnosRestAdapter(config['ca_origin'], **common)
        serve(ct, ca, sys.stdin.buffer, sys.stdout.buffer)
    finally:
        if ct is not None:
            ct.cleanup({'kind': 'remove_owned_rules'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('MCP server stopped: '+type(exc).__name__, file=sys.stderr)
        raise SystemExit(2)
