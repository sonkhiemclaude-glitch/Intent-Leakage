"""Explicit external read-only qualification; config is private, never reported."""
import argparse,json,os
from pathlib import Path
from intent_leakage.adapters.base import require_external_execution
from intent_leakage.adapters.local_shaped_transport import LocalTransportProfile
from intent_leakage.experiments.readonly_bridge_lab import run_readonly_bridge


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--profile',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-external',action='store_true');args=p.parse_args()
    require_external_execution(args.allow_external)
    config=json.loads(args.config.read_text());profile=LocalTransportProfile(**json.loads(args.profile.read_text()))
    # Reserve a new output before any external read. Never overwrite old results.
    fd=os.open(args.output,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump({'state':'incomplete'},f);f.flush();os.fsync(f.fileno())
        report=run_readonly_bridge(config,profile=profile,allow_external=True)
        f.seek(0);json.dump(report,f,indent=2);f.truncate();f.flush();os.fsync(f.fileno())
    print(json.dumps({k:report.get(k) for k in ('success','response_body_bytes','error_type','formal_profile_selected')}))
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
