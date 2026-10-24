import hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('start_prepared_application',ROOT/'scripts/start_prepared_application.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_frozen_source_changes_refused_before_deployment(tmp_path,monkeypatch):
    source=tmp_path/'worker.py';source.write_text('original')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'protocol_revision':'application_bodies_convergence_v1',
         'source_sha256':{'worker.py':hashlib.sha256(source.read_bytes()).hexdigest()},'plan':{}}))
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    assert runner.validate_source(manifest)['plan']=={}
    source.write_text('changed')
    with pytest.raises(ValueError,match='source changed'):runner.validate_source(manifest)


def test_source_manifest_path_escape_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path/'project');runner.ROOT.mkdir()
    outside=tmp_path/'outside';outside.write_text('data')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'protocol_revision':'application_bodies_convergence_v1',
       'source_sha256':{'../outside':hashlib.sha256(outside.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError):runner.validate_source(manifest)
