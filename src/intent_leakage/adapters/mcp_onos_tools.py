"""Frozen, narrow controller tools for the three-class instrumentation pilot.

This is not an implementation of the other twelve paper classes. The stdio
protocol is pinned to MCP 2025-06-18, without optional sampling/resources.
"""
from __future__ import annotations

import copy
import re

from ..utils.hashing import stable_digest

PROTOCOL_VERSION = '2025-06-18'
SERVER_REVISION = 'intent-leakage-onos-stdio-v1'
HOST_PATTERN = r'00:00:00:00:00:[0-9A-Fa-f]{2}/None'
HOSTS = {'type': 'array', 'minItems': 1, 'maxItems': 4, 'uniqueItems': True,
         'items': {'type': 'string', 'pattern': '^'+HOST_PATTERN+'$'}}


def tool(name, description, properties, required, readonly):
    return {'name': name, 'description': description,
            'inputSchema': {'type': 'object', 'properties': properties,
                            'required': required, 'additionalProperties': False},
            'annotations': {'readOnlyHint': readonly, 'openWorldHint': False}}


TOOLS = [
    tool('lookup_hosts', 'Read known lab hosts; provide 1, 2 or 4 host IDs.',
         {'host_ids': HOSTS}, ['host_ids'], True),
    tool('quarantine_hosts', 'Install owned ETH_SRC/ETH_DST drop rules for 1, 2 or 4 lab hosts. This pilot does not preserve a separate management VLAN.',
         {'host_ids': HOSTS}, ['host_ids'], False),
    tool('release_owned', 'Remove only rules acknowledged to this server session.', {}, [], False),
    tool('read_control_flows', 'Read flow state through the control endpoint for bounded installation checks.', {}, [], True),
    tool('read_flows', 'Read current controller flows through the assurance endpoint.', {}, [], True),
    tool('audit_policies', 'Read 1, 2 or 4 specified existing flow policies once, without mutation.',
         {'policies': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'uniqueItems': True,
                       'items': {'type': 'object', 'properties': {
                           'device_id': {'type': 'string', 'pattern': '^of:[0-9a-f]{16}$'},
                           'flow_id': {'type': 'string', 'pattern': '^-?[0-9]+$'}},
                           'required': ['device_id', 'flow_id'], 'additionalProperties': False}}},
         ['policies'], True),
]
TOOL_SCHEMA_SHA256 = stable_digest(TOOLS)


def list_tools():
    return copy.deepcopy(TOOLS)


def validate_call(name, arguments):
    if not isinstance(arguments, dict):
        raise ValueError('Tool arguments must be an object')
    if name in ('lookup_hosts', 'quarantine_hosts'):
        values = arguments.get('host_ids')
        if set(arguments) != {'host_ids'} or not isinstance(values, list) or len(values) not in (1, 2, 4):
            raise ValueError('Expected 1, 2 or 4 lab host IDs')
        if any(not isinstance(v, str) or not re.fullmatch(HOST_PATTERN, v) for v in values) or len(set(values)) != len(values):
            raise ValueError('Invalid or duplicate lab host IDs')
    elif name == 'audit_policies':
        values = arguments.get('policies')
        if set(arguments) != {'policies'} or not isinstance(values, list) or len(values) not in (1, 2, 4):
            raise ValueError('Expected 1, 2 or 4 policy identities')
        identities = []
        for value in values:
            if not isinstance(value, dict) or set(value) != {'device_id', 'flow_id'}:
                raise ValueError('Invalid policy identity')
            if not isinstance(value['device_id'], str) or not re.fullmatch(r'of:[0-9a-f]{16}', value['device_id']):
                raise ValueError('Invalid device ID')
            if not isinstance(value['flow_id'], str) or not re.fullmatch(r'-?[0-9]+', value['flow_id']):
                raise ValueError('Invalid flow ID')
            identities.append((value['device_id'], value['flow_id']))
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate policy identity')
    elif name in ('read_flows', 'read_control_flows', 'release_owned'):
        if arguments:
            raise ValueError('This tool accepts no arguments')
    else:
        raise ValueError('Unregistered controller tool')


def dispatch(ct, ca, name, arguments):
    validate_call(name, arguments)
    if name == 'lookup_hosts':
        return {'hosts': ct.lookup_hosts(arguments['host_ids'])}
    if name == 'quarantine_hosts':
        return ct.apply_operation({'kind': name, 'host_ids': arguments['host_ids']})
    if name == 'release_owned':
        ct.cleanup({'kind': 'remove_owned_rules'})
        return {'delete_requests_acknowledged': True, 'dataplane_verified': False}
    if name == 'read_control_flows':
        return ct.assurance_read({'kind': 'flows'})
    if name == 'read_flows':
        return ca.assurance_read({'kind': 'flows'})
    return {'policies': [ca.assurance_read({'kind': 'flow', **p}) for p in arguments['policies']]}
