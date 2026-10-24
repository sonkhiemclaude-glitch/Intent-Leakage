import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('capture_cli',Path(__file__).resolve().parents[1]/'scripts/qualify_captured_readonly_bridge.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_both_tunnel_directions_count_full_ip_packets_including_ack_only_frames():
    result=module.packet_summary('40000\t45000\t60\t74\n45000\t40000\t60\t74\n40000\t45000\t1500\t1514\n',45000)
    assert result['agent_to_bridge']=={'packets':2,'ip_bytes':1560,'captured_frame_bytes':1588}
    assert result['bridge_to_agent']['ip_bytes']==60


@pytest.mark.parametrize('text',['1\t2\t60\t74','40000\t45000\t60\t74','bad fields'])
def test_backend_only_or_missing_direction_or_malformed_rows_are_rejected(text):
    with pytest.raises(ValueError):module.packet_summary(text,45000)
