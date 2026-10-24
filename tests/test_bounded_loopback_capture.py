"""Owned subprocess fault tests with a FAKE tcpdump; not packet evidence."""
import json,subprocess,sys,time
import pytest
import intent_leakage.experiments.bounded_loopback_capture as module


def fake(monkeypatch,tmp_path,*,drops=0,ready=True):
    script=tmp_path/'fake_tcpdump.py'
    script.write_text('import sys,signal,time\nfrom pathlib import Path\n'+
        "Path(sys.argv[sys.argv.index('-w')+1]).write_bytes(b'x'*24)\n"+
        f"def stop(*_):\n print('1 packet captured\\n{drops} packets dropped by kernel',file=sys.stderr,flush=True)\n sys.exit(0)\n"+
        "signal.signal(signal.SIGINT,stop)\n"+
        ("print('listening on lo',file=sys.stderr,flush=True)\n" if ready else 'sys.exit(0)\n')+
        'while True:time.sleep(.01)\n')
    original=subprocess.Popen
    monkeypatch.setattr(module.sys,'platform','linux')
    monkeypatch.setattr(module.subprocess,'Popen',lambda argv,**kw:original([sys.executable,str(script),*argv[1:]],**kw))


def test_ready_subprocess_stopped_and_evidence_kept_without_overwrite(monkeypatch,tmp_path):
    fake(monkeypatch,tmp_path);out=tmp_path/'out'
    result=module.capture_loopback_call(lambda:{'success':True},out,allow_external=True,max_s=2)
    assert result['success'] and not result['selected_tunnel_verified'] and result['kernel_drops']==0
    assert json.loads((out/'capture.json').read_text())==result
    with pytest.raises(FileExistsError):module.capture_loopback_call(lambda:pytest.fail('replayed'),out,allow_external=True)


@pytest.mark.parametrize('drops,ready',[(1,True),(0,False)])
def test_drops_and_readiness_failures_never_count_as_success(monkeypatch,tmp_path,drops,ready):
    fake(monkeypatch,tmp_path,drops=drops,ready=ready);calls=[]
    result=module.capture_loopback_call(lambda:calls.append(1) or {'success':True},tmp_path/'out',allow_external=True,max_s=2)
    assert not result['success']
    if not ready:assert not calls


def test_callback_failure_retained_and_owned_capture_terminated(monkeypatch,tmp_path):
    fake(monkeypatch,tmp_path)
    def fail():raise ValueError('private payload not to be logged')
    result=module.capture_loopback_call(fail,tmp_path/'out',allow_external=True,max_s=2)
    assert not result['success'] and result['error_type']=='ValueError' and result['capture_exit_code']==0
    assert 'private payload' not in json.dumps(result)


def test_capture_deadline_expires_without_claiming_callback_cancellation(monkeypatch,tmp_path):
    fake(monkeypatch,tmp_path)
    def slow():time.sleep(.5);return {'success':True}
    result=module.capture_loopback_call(slow,tmp_path/'out',allow_external=True,max_s=.2)
    assert not result['success'] and result['capture_deadline_expired']
