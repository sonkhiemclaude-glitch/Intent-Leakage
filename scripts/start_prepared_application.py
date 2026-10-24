"""Inspect locally by default; --run launches the bounded T4 qualification.

No Bedrock calls. Source changes after preparation are refused, before SSH.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def validate_source(manifest_path):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest['protocol_revision']!='application_bodies_convergence_v1':raise ValueError('unexpected protocol')
    for folder in manifest.get('packaged_roots',[]):
        actual={str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*.py')}
        expected={p for p in manifest['source_sha256'] if p.startswith(folder+'/') and p.endswith('.py')}
        if actual!=expected:raise ValueError('prepared source file set changed: '+folder)
    for relative,expected in manifest['source_sha256'].items():
        path=(ROOT/relative).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('prepared source changed: '+relative)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Actually connect and deploy to T4; omit for local validation only')
    parser.add_argument('--window-minutes',type=int,default=90)
    parser.add_argument('--manifest',type=Path,default=ROOT/'deployment/application_pilot/prepared_source.json')
    args=parser.parse_args()
    if not 30<=args.window_minutes<=120:parser.error('window must be 30..120 minutes')
    manifest=validate_source(args.manifest)
    if not args.run:
        print(json.dumps({'state':'local_sources_verified_no_deployment','files':len(manifest['source_sha256']),
                          'plan':manifest['plan'],'ssh_connections':0,'aws_requests':0},indent=2));return
    # Dates are generated at launch so an overnight preparation cannot silently expire.
    subprocess.run([sys.executable,str(ROOT/'scripts/deploy_overnight_queue.py'),'--pilot-variant','application',
                    '--start-batch','1','--end-batch','2','--deadline-epoch',str(time.time()+args.window_minutes*60)],cwd=ROOT,check=True)


if __name__=='__main__':main()
