"""Bounded owned tcpdump lifecycle, Linux loopback only.

Capture duration is bounded, not arbitrary callback/network execution. Run the
caller itself in a separately time-limited process. Capture success alone is
not verification of selected tunnel records or of privacy/cost claims.
"""
import json,os,re,selectors,signal,subprocess,sys,threading,time
from pathlib import Path
from ..adapters.base import require_external_execution


def _save(path,data):
    temp=path.with_suffix('.next');fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def capture_loopback_call(call,output,*,allow_external=False,max_s=30):
    require_external_execution(allow_external)
    if sys.platform!='linux':raise ValueError('Linux isolated namespace required')
    if not .1<=max_s<=120:raise ValueError('invalid bounded capture duration')
    out=Path(output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
    report={'success':False,'state':'incomplete','selected_tunnel_verified':False,'max_capture_s':max_s,
            'filter':'tcp and host127.0.0.1','interface':'lo','call_result':None}
    _save(out/'capture.json',report);process=None;timer=None;stderr=b'';expired=threading.Event()
    start=time.monotonic()
    try:
        process=subprocess.Popen(['tcpdump','--immediate-mode','-Z','root','-p','-U','-s','0','-i','lo','-w',str(out/'capture.pcap'),'tcp','and','host','127.0.0.1'],
                                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        def stop_at_deadline():
            expired.set()
            if process.poll() is None:
                try:process.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
        timer=threading.Timer(max_s,stop_at_deadline);timer.daemon=True;timer.start()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr,selectors.EVENT_READ)
            while b'listening on' not in stderr:
                if expired.is_set() or process.poll() is not None or time.monotonic()-start>min(5,max_s):raise RuntimeError('capture readiness failed')
                if selector.select(.05):
                    chunk=os.read(process.stderr.fileno(),4096)
                    if not chunk:raise RuntimeError('capture ended before readiness')
                    stderr+=chunk
                    if len(stderr)>16384:raise RuntimeError('capture diagnostics bound exceeded')
        report['ready_after_s']=time.monotonic()-start
        report['call_result']=call()
    except Exception as exc:report['error_type']=type(exc).__name__
    finally:
        if timer:timer.cancel()
        if process:
            if process.poll() is None:
                try:process.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
            try:_,tail=process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();_,tail=process.communicate(timeout=2);report['forced_kill']=True
            stderr+=tail;report['capture_exit_code']=process.returncode
        report['elapsed_s']=time.monotonic()-start;report['capture_deadline_expired']=expired.is_set()
        # These diagnostics contain tcpdump statistics, never captured packet payloads.
        (out/'tcpdump.stderr').write_bytes(stderr);os.chmod(out/'tcpdump.stderr',0o600)
        text=stderr.decode(errors='replace');drops=re.search(r'(\d+) packets? dropped by kernel',text);packets=re.search(r'(\d+) packets? captured',text)
        report['kernel_drops']=int(drops.group(1)) if drops else None
        report['packets_captured']=int(packets.group(1)) if packets else None
        pcap=out/'capture.pcap';valid_file=pcap.exists() and pcap.stat().st_size>=24
        if pcap.exists():os.chmod(pcap,0o600)
        result=report['call_result']
        report['success']=bool(isinstance(result,dict) and result.get('success') is True and not report.get('error_type') and not report.get('forced_kill') and not expired.is_set()
                               and report.get('capture_exit_code')==0 and report['kernel_drops']==0 and report['packets_captured'] and valid_file)
        report['state']='captured_requires_tunnel_verification' if report['success'] else 'failed'
        _save(out/'capture.json',report)
    return report
