"""Narrow ONOS 2.7 REST binding for live host-quarantine instrumentation.

Only owned drop rules may be removed. Slice/service and demand semantics remain
unprovisioned: this adapter does not manufacture controller applications for them.
"""
from __future__ import annotations
import base64
import json
import re
import ssl
import time
import urllib.request
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, quote
from .base import AdapterNotProvisionedError
from .onos import OnosControllerAdapter
from .application_observation import ApplicationExchange


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('ONOS redirects are not permitted')


class OnosRestAdapter(OnosControllerAdapter):
    def __init__(self, base_url: str, *, username: str, password: str,
                 ca_file: str, allow_external: bool = False, timeout_s: float = 10, application_observer=None) -> None:
        super().__init__(allow_external=allow_external)
        url = urlsplit(base_url)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError('A fixed HTTPS origin without credentials/path is required')
        if timeout_s <= 0:
            raise ValueError('timeout_s must be positive')
        self.base_url = base_url.rstrip('/') + '/onos/v1/'
        self.timeout_s = timeout_s
        self.application_observer = application_observer
        self.application_observation_errors = []
        context = ssl.create_default_context(cafile=ca_file)
        context.minimum_version = context.maximum_version = ssl.TLSVersion.TLSv1_2
        self._opener = urllib.request.build_opener(_NoRedirect(), urllib.request.HTTPSHandler(context=context))
        self._auth = 'Basic ' + base64.b64encode(f'{username}:{password}'.encode()).decode()
        self._owned: set[tuple[str, str]] = set()

    def request(self, method: str, path: str, payload=None, *, _return_body=False):
        # Paths are generated internally, not read from model/tool output.
        if not re.fullmatch(r'(devices|hosts|links|flows)(/[A-Za-z0-9:%._-]+)*', path):
            raise ValueError('Unregistered ONOS resource')
        data = None if payload is None else json.dumps(payload, separators=(',', ':')).encode()
        headers={'Authorization':self._auth, 'Accept':'application/json', 'Connection':'close'}
        if data is not None:headers['Content-Type']='application/json'
        req = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
        started = time.monotonic()
        with self._opener.open(req, timeout=self.timeout_s) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError('Controller response exceeds frozen size limit')
        result = json.loads(raw) if raw else {}
        if self.application_observer is not None:
            try:
                self.application_observer(ApplicationExchange(method,path,data or b'',raw,time.monotonic()-started))
            except Exception as exc:
                # Never lose acknowledged flow ownership or replay a completed
                # mutation because an instrumentation callback failed. The run
                # must check this explicit incomplete-capture flag separately.
                self.application_observation_errors.append(type(exc).__name__)
        return raw if _return_body else result

    def read_application_body(self, resource: str) -> bytes:
        """Exact JSON body of a read; excludes HTTP headers and TLS framing."""
        if resource not in ('flows','hosts','devices','links'):
            raise ValueError('unregistered body-read resource')
        return self.request('GET',resource,_return_body=True)

    def discover_topology(self):
        return {k:self.request('GET', k)[k] for k in ('devices','hosts','links')}

    def lookup_hosts(self, host_ids: Sequence[str]):
        wanted = set(host_ids)
        hosts = self.request('GET','hosts')['hosts']
        found = [h for h in hosts if h['id'] in wanted]
        if len(found) != len(wanted):
            raise ValueError('Unknown target host')
        return found

    def lookup_paths(self, demands):
        raise AdapterNotProvisionedError('Demand semantics/topologies must be frozen first')

    def apply_operation(self, operation: Mapping[str, Any]):
        if operation.get('kind') != 'quarantine_hosts' or set(operation) != {'kind','host_ids'}:
            raise ValueError('Operation is outside the implemented allowlist')
        hosts = self.lookup_hosts(operation['host_ids'])
        if not hosts or len(hosts) > 4:
            raise ValueError('Pilot requires one to four known hosts')
        rules=[]
        for h in hosts:
            for loc in h['locations']:
                for kind in ('ETH_SRC','ETH_DST'):
                    rules.append({'deviceId':loc['elementId'],'priority':50000,
                        'isPermanent':True,'timeout':0,'appId':'org.intentleakage.pilot',
                        'selector':{'criteria':[{'type':kind,'mac':h['mac']}]},
                        'treatment':{'instructions':[]}})
        result=self.request('POST','flows',{'flows':rules})
        entries=[(str(f['deviceId']),str(f['flowId'])) for f in result['flows']]
        # Preserve any acknowledged IDs even if an incomplete response fails validation.
        self._owned.update(entries)
        if len(entries)!=len(rules):
            raise ValueError('Controller did not acknowledge every rule')
        return {'rules':[{'device_id':d,'flow_id':f} for d,f in entries]}

    def assurance_read(self, query):
        if query == {'kind':'flows'}:
            return self.request('GET','flows')
        if set(query) == {'kind','device_id','flow_id'} and query['kind'] == 'flow':
            device, flow = str(query['device_id']), str(query['flow_id'])
            if not re.fullmatch(r'of:[0-9a-f]{16}',device) or not re.fullmatch(r'-?[0-9]+',flow):
                raise ValueError('Invalid flow identity')
            return self.request('GET','flows/'+device+'/'+flow)
        raise ValueError('Unregistered assurance query')

    def cleanup(self, inverse_operation):
        if inverse_operation != {'kind':'remove_owned_rules'}:
            raise ValueError('Only this adapter instance\'s rules may be removed')
        for device, flow in sorted(self._owned):
            self.request('DELETE','flows/'+quote(device,safe=':')+'/'+flow)
            self._owned.remove((device,flow))
