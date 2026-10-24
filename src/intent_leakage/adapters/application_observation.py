"""Opt-in privileged application-body observations; NEVER TLS record sizes.

Bodies stay in memory unless the trusted caller explicitly handles them. Public
metadata contains only lengths/digests and no Authorization/header values.
"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ApplicationExchange:
    method:str
    resource:str
    request_body:bytes
    response_body:bytes
    elapsed_s:float
    def metadata(self):
        return {'method':self.method,'resource':self.resource,'request_body_bytes':len(self.request_body),
                'response_body_bytes':len(self.response_body),'request_body_sha256':hashlib.sha256(self.request_body).hexdigest(),
                'response_body_sha256':hashlib.sha256(self.response_body).hexdigest(),'unshaped_exchange_elapsed_s':self.elapsed_s,
                'representation':'HTTP application bodies, excluding headers/TLS/IP overhead'}
