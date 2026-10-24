"""Reproducible official-client interoperability probe against a local TLS fixture.

Uses a separate SDK environment; no Bedrock, EC2, real credentials or ONOS calls.
Requires the project's dev/capture dependencies and mcp==1.26.0 in --sdk-python.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
PROBE='''import asyncio,os,json,sys,importlib.metadata
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from intent_leakage.adapters.mcp_onos_tools import TOOL_SCHEMA_SHA256,list_tools
async def main():
 assert importlib.metadata.version('mcp')=='1.26.0'
 config=json.loads(os.environ['MCP_TEST_CONFIG'])
 params=StdioServerParameters(command=sys.executable,args=['-m','intent_leakage.adapters.mcp_onos_server'],env={'PYTHONPATH':os.environ['PYTHONPATH'],'IL_MCP_ONOS_CONFIG':json.dumps({k:v for k,v in config.items() if k!='password'}),'IL_MCP_ONOS_PASSWORD':config['password']})
 async with stdio_client(params) as (read,write):
  async with ClientSession(read,write) as session:
   initialized=await session.initialize();tools=await session.list_tools()
   expected={t['name']:t for t in list_tools()}
   assert {t.name for t in tools.tools}==set(expected)
   for t in tools.tools:assert t.inputSchema==expected[t.name]['inputSchema'] and t.description==expected[t.name]['description']
   args={'host_ids':['00:00:00:00:00:01/None']}
   hosts=await session.call_tool('lookup_hosts',args)
   assert not hosts.isError and hosts.structuredContent['hosts']
   applied=await session.call_tool('quarantine_hosts',args)
   assert not applied.isError and len(applied.structuredContent['rules'])==2
   flows=await session.call_tool('read_flows')
   assert not flows.isError and len(flows.structuredContent['flows'])==3
   audited=await session.call_tool('audit_policies',{'policies':[applied.structuredContent['rules'][0]]})
   assert not audited.isError and len(audited.structuredContent['policies'][0]['flows'])==1
   released=await session.call_tool('release_owned')
   assert not released.isError and released.structuredContent=={'delete_requests_acknowledged':True,'dataplane_verified':False}
   control=await session.call_tool('read_control_flows')
   assert not control.isError and len(control.structuredContent['flows'])==1
   applied=await session.call_tool('quarantine_hosts',args)
   assert not applied.isError and len(applied.structuredContent['rules'])==2
   print(json.dumps({'sdk_version':importlib.metadata.version('mcp'),'negotiated_protocol':initialized.protocolVersion,'tools':sorted(expected),'tool_schema_sha256':TOOL_SCHEMA_SHA256,'successful_calls':7,'paper_results_claimed':False,'backend':'local TLS API fixture, not ONOS'}),flush=True)
asyncio.run(asyncio.wait_for(main(),30))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sdk-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    # Preserve the venv launcher path: resolving its symlink selects the base
    # interpreter and loses the SDK environment's site-packages.
    args=parser.parse_args();sdk=args.sdk_python.absolute();out=args.output.resolve()
    assert sdk.is_file()
    out.mkdir(parents=True,exist_ok=False)
    probe=out/'sdk_probe.py';probe.write_text(PROBE)
    sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('fixture',ROOT/'tests/test_mcp_stdio.py')
    fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
    with tempfile.TemporaryDirectory(prefix='intent-mcp-api-fixture-') as temporary:
        fixture=fixture_module.controller.__wrapped__(Path(temporary));config,state,servers=next(fixture)
        config['controller_timeout_s']=2
        state['flows']['999']={'id':'999','appId':'another-project'}
        try:
            environment={'PATH':os.environ.get('PATH','/usr/bin:/bin'),
                         'PYTHONPATH':str(ROOT/'src'),'MCP_TEST_CONFIG':json.dumps(config)}
            process=subprocess.run([str(sdk),str(probe)],capture_output=True,text=True,timeout=45,env=environment)
            (out/'stdout.txt').write_text(process.stdout);(out/'stderr.txt').write_text(process.stderr)
            assert process.returncode==0, 'SDK interoperability failed; inspect the saved fixture logs'
            assert set(state['flows'])=={'999'},'EOF cleanup did not preserve the foreign rule'
            report=json.loads(process.stdout)
            report.update(eof_cleanup_verified=True,foreign_rule_preserved=True,
                          bedrock_requests=0,fixture_only=True)
        finally:
            try:next(fixture)
            except StopIteration:pass
    freeze=subprocess.run([str(sdk),'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'requirements.txt').write_text(freeze.stdout)
    sources=list((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'tests/test_mcp_stdio.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
        for source in sources:archive.add(source,arcname=str(source.relative_to(ROOT)))
    (out/'report.json').write_text(json.dumps(report,indent=2))
    (ROOT/'outputs/mcp_stdio_interop_latest.json').write_text(json.dumps({'output':str(out),'report_sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest()},indent=2))
    print(json.dumps({k:report[k] for k in ('sdk_version','negotiated_protocol','successful_calls','eof_cleanup_verified','fixture_only')}))


if __name__=='__main__':
    main()
