"""Exact operation binding and explicit counts projection for a native pilot."""
import time
from ..utils.hashing import stable_digest


class IntentBoundTools:
    def __init__(self,mcp,expected):
        self.mcp=mcp;self.expected=expected;self.results=[];self.finished=[];self.revision=mcp.revision
    def list_tools(self):return self.mcp.list_tools()
    def call_tool(self,name,arguments):
        index=len(self.results)
        if index>=len(self.expected) or (name,arguments)!=self.expected[index]:
            raise ValueError('Tool call does not match the frozen operator intent')
        raw=self.mcp.call_tool(name,arguments)
        self.results.append(raw);self.finished.append((name,time.monotonic()))
        if name=='read_flows':
            flows=raw['flows']
            return {'projection':'counts_only_v1','total_flows':len(flows),
                'owned_added_flows':sum(f.get('appId')=='org.intentleakage.pilot' and f.get('state')=='ADDED' for f in flows),
                'full_result_sha256':stable_digest(raw)}
        return raw
    @property
    def complete(self):return len(self.results)==len(self.expected)
