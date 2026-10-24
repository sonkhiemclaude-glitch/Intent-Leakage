"""Linux-only engineering run: bounded PCAP + HTTPS read via loopback TLS cells."""
import argparse,hashlib,json,os,subprocess,uuid
from pathlib import Path
from intent_leakage.adapters.base import require_external_execution
from intent_leakage.adapters.local_shaped_transport import LocalTransportProfile
from intent_leakage.adapters.tshark import TsharkTlsParser
from intent_leakage.adapters.onos_rest import OnosRestAdapter
from intent_leakage.adapters.owned_flow_transaction import fingerprint,identity
from intent_leakage.experiments.readonly_bridge_lab import run_readonly_bridge
from intent_leakage.experiments.bounded_loopback_capture import capture_loopback_call,_save


def packet_summary(text,server_port):
    totals={'agent_to_bridge':{'packets':0,'ip_bytes':0,'captured_frame_bytes':0},'bridge_to_agent':{'packets':0,'ip_bytes':0,'captured_frame_bytes':0}}
    for line in text.splitlines():
        fields=line.split('\t')
        if len(fields)!=4:raise ValueError('unexpected IP accounting fields')
        sport,dport,ip_size,frame_size=map(int,fields)
        if dport==server_port:side='agent_to_bridge'
        elif sport==server_port:side='bridge_to_agent'
        else:raise ValueError('backend packet outside selected tunnel')
        totals[side]['packets']+=1;totals[side]['ip_bytes']+=ip_size;totals[side]['captured_frame_bytes']+=frame_size
    if any(v['packets']==0 for v in totals.values()):raise ValueError('missing tunnel direction')
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-external',action='store_true');args=p.parse_args()
    require_external_execution(args.allow_external)
    config=json.loads(args.config.read_text());profile=LocalTransportProfile(**json.loads(args.profile.read_text()))
    if profile.rounds*profile.period_s+10>120:raise ValueError('profile too long for capture wrapper')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False);report={'state':'incomplete','success':False,'formal_experiment':False,'remote_mutations_supported':False}
    _save(args.output/'result.json',report)
    try:
        baseline_adapter=OnosRestAdapter(config['origin'],username=config['username'],password=config['password'],ca_file=config['ca_file'],timeout_s=config['timeout_s'],allow_external=True)
        def snapshot():
            flows=json.loads(baseline_adapter.read_application_body('flows'))['flows']
            return sorted((identity(flow),fingerprint(flow)) for flow in flows)
        before=snapshot()
        _save(args.output/'controller_before.json',before)
        report['controller_baseline_preserved']=False
        result=capture_loopback_call(lambda:run_readonly_bridge(config,profile=profile,allow_external=True),args.output/'capture',allow_external=True,max_s=profile.rounds*profile.period_s+10)
        if not result['success']:raise RuntimeError('capture or read qualification failed')
        after=snapshot()
        _save(args.output/'controller_after.json',after)
        if before!=after:raise RuntimeError('controller rule baseline changed')
        report['controller_baseline_preserved']=True
        report['controller_baseline_sha256']=hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
        read=result['call_result'];manifest={**read['flow_manifest'],'trace_id':uuid.uuid4().hex}
        events=TsharkTlsParser(timeout_s=30).parse(args.output/'capture/capture.pcap',flow_manifest=manifest)
        if {e['direction'] for e in events}!={-1,1} or len({e['flow_id'] for e in events})!=1:raise ValueError('missing/ambiguous tunnel TLS records')
        port=manifest['endpoints'][0]['server_port']
        command=['tshark','-n','-r',str(args.output/'capture/capture.pcap'),'-Y',f'ip.src==127.0.0.1 && ip.dst==127.0.0.1 && tcp.port=={port}','-T','fields']
        for field in ('tcp.srcport','tcp.dstport','ip.len','frame.len'):command+=['-e',field]
        packets=subprocess.run(command,capture_output=True,text=True,check=True,timeout=30)
        accounting=packet_summary(packets.stdout,port)
        _save(args.output/'flow_manifest.json',manifest)
        fd=os.open(args.output/'tls_records.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:
            for event in events:f.write(json.dumps(event)+'\n')
        report.update(tls_parser_revision='tls_record_metadata_v2_opaque_type',state='read_only_capture_qualified',success=True,selected_tunnel_verified=True,tls_records=len(events),
                      tls_record_bytes_by_direction={str(d):sum(e['length_bytes'] for e in events if e['direction']==d) for d in (-1,1)},
                      loopback_packet_accounting=accounting,packet_accounting_includes_handshake_ack_and_retransmissions=True,
                      physical_link_wire_cost_measured=False,pcap_sha256=hashlib.sha256((args.output/'capture/capture.pcap').read_bytes()).hexdigest(),
                      response_body_sha256=read['response_body_sha256'],response_body_bytes=read['response_body_bytes'])
    except Exception as exc:report.update(state='failed',error_type=type(exc).__name__)
    _save(args.output/'result.json',report);print(json.dumps(report,indent=2))
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
