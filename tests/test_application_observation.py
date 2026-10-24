import json
from intent_leakage.adapters.onos_rest import OnosRestAdapter


class Reply:
    def __init__(self,raw):self.raw=raw
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,n):return self.raw


class Opener:
    def __init__(self,raw):self.raw=raw;self.calls=0
    def open(self,request,timeout):self.calls+=1;return Reply(self.raw)


def fixture(observer):
    # Avoid any socket/TLS creation; exercise the existing request serializer.
    adapter=object.__new__(OnosRestAdapter);adapter.base_url='https://fixture.invalid/onos/v1/'
    adapter.timeout_s=1;adapter._auth='Basic PRIVATE_NOT_TO_BE_RECORDED';adapter.application_observer=observer
    adapter.application_observation_errors=[];adapter._opener=Opener(b'{"flows":[]}');return adapter


def test_exact_application_bodies_are_not_tls_or_auth_headers():
    observed=[];adapter=fixture(observed.append)
    result=adapter.request('POST','flows',{'flows':[]})
    assert result=={'flows':[]} and len(observed)==1
    sample=observed[0];assert sample.request_body==b'{"flows":[]}' and sample.response_body==b'{"flows":[]}'
    assert 'PRIVATE_NOT_TO_BE_RECORDED' not in json.dumps(sample.metadata())
    assert sample.metadata()['representation'].startswith('HTTP application bodies')


def test_observer_failure_flags_capture_without_losing_ack_or_replaying():
    def broken(_):raise ValueError('injected')
    adapter=fixture(broken)
    assert adapter.request('POST','flows',{'flows':[]})=={'flows':[]}
    assert adapter._opener.calls==1 and adapter.application_observation_errors==['ValueError']
