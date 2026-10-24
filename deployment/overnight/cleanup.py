"""Remove only owned resources; disable completed queues to avoid future boot races."""
import json,time
from pathlib import Path
from common import cleanup,command
root=Path(__file__).resolve().parent
cleanup()
try:state=json.loads((root/'checkpoint.json').read_text()).get('state')
except (OSError,ValueError):state=None
if state in ('complete','failed','deadline_reached'):
 unit=root.name.lower()+'.service'
 if unit.startswith('intent-leakage-night-'):
  p=command(['sudo','-n','systemctl','disable',unit])
  (root/'terminal_service_cleanup.json').write_text(json.dumps({'state':state,'disable_returncode':p.returncode,'epoch':time.time()}))

# Only the current queue's separate provider credential may be deleted.
try:
 config=json.loads((root/'queue.json').read_text())
 if config.get('protocol_revision') in ('bedrock_mcp_convergence_v1','bedrock_mcp_convergence_v2','bedrock_mcp_convergence_v3'):(root/'.bedrock-token').unlink(missing_ok=True)
except (OSError,ValueError):pass
