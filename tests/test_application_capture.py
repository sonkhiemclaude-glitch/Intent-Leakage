import json,stat
import pytest
from intent_leakage.adapters.application_capture import ApplicationCapture,verify_capture
from intent_leakage.adapters.application_observation import ApplicationExchange


def exchange():return ApplicationExchange('POST','flows',b'{"flows":[]}',b'{"ok":true}',.02)


def test_capture_durable_exact_bytes_permissions_and_integrity(tmp_path):
    c=ApplicationCapture(tmp_path,'run01');c.phase='active';c.observer('CT')(exchange())
    with pytest.raises(ValueError):verify_capture(c.root)
    receipt=c.finish(run_success=True);m=verify_capture(c.root,expected_sha256=receipt['manifest_sha256'])
    assert m['exchanges'][0]['phase']=='active' and not m['eligible_for_paper_test']
    assert (c.root/'000000.request.bin').read_bytes()==exchange().request_body
    assert stat.S_IMODE((c.root/'000000.request.bin').stat().st_mode)==0o600
    (c.root/'000000.response.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):verify_capture(c.root)


def test_failed_quota_or_observer_cannot_be_marked_complete(tmp_path):
    c=ApplicationCapture(tmp_path,'run02',max_exchanges=1);observe=c.observer('CA');observe(exchange())
    with pytest.raises(ValueError,match='quota'):observe(exchange())
    assert not c.finish(run_success=True)['complete']
    with pytest.raises(ValueError,match='incomplete'):verify_capture(c.root)
    d=ApplicationCapture(tmp_path,'run03');d.observer('CT')(exchange())
    assert not d.finish(run_success=True,observer_errors=['OSError'])['complete']


def test_existing_run_never_overwritten_or_path_escape(tmp_path):
    ApplicationCapture(tmp_path,'run04')
    with pytest.raises(FileExistsError):ApplicationCapture(tmp_path,'run04')
    with pytest.raises(ValueError):ApplicationCapture(tmp_path,'../escape')


def test_unfinished_or_empty_run_not_eligible(tmp_path):
    c=ApplicationCapture(tmp_path,'run05');assert not c.finish(run_success=True)['complete']
    with pytest.raises(ValueError):c.observer('CT')(exchange())


def test_buffering_has_no_body_disk_io_until_finish_and_disk_failure_blocks_success(tmp_path,monkeypatch):
    import intent_leakage.adapters.application_capture as module
    c=ApplicationCapture(tmp_path,'run06');c.observer('CT')(exchange())
    assert not list(c.root.glob('*.bin'))
    original=module._write_new
    def fail_body(path,data):
        if path.suffix=='.bin':raise OSError('injected full disk')
        return original(path,data)
    monkeypatch.setattr(module,'_write_new',fail_body)
    assert not c.finish(run_success=True)['complete']
    with pytest.raises(ValueError,match='incomplete'):verify_capture(c.root)
