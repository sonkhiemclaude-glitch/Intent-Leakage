LIVE NATIVE BEDROCK -> MCP STDIO -> ONOS INSTRUMENTATION

Protocol: bedrock_mcp_convergence_v3; three classes I01/I02/I15, q=1/2/4,
two repetitions per cell and batch. No shaping; not the full 15-class paper grid.
Frozen model: amazon.nova-micro-v1:0, us-east-1, temperature 0, non-streaming,
512 output tokens/request, exactly one SDK attempt (no hidden retries).
Per row: at most 4 requests, 2 tools, 90 seconds at bounded RPC/SDK boundaries,
64 KiB request JSON and 32 KiB tool result. Per batch provider quota: 72 requests
and 1 MiB cumulative request JSON. Counts are engineering bounds, not exact bills.

The network-none ONOS/Mininet namespace is retained. A host-side SDK process
exposes one private AF_UNIX socket in a short /tmp directory; only that socket is
mounted read-only into the worker. The separate provider credential is mode 0600
on the host, outside source/results and the socket bind mount; cleanup removes
it after the service exits. The original .env and SSH key are never uploaded.
The MCP server's environment does not receive the provider credential.

An exact-intent wrapper rejects any unexpected tool or arguments BEFORE the MCP
call. read_flows results are explicitly projected to counts_only_v1 for the model,
with the full result hash retained. This avoids sending large reactive-flow dumps
to the model. The complete TLS assurance response is still captured for CA.
This is a constrained native execution pilot, not free-form planning validation.

After the frozen required tool sequence completes, the driver separately waits for flow installation,
checks CA state and held-out ICMP reachability, then verifies cleanup. Model
completion alone never establishes fulfillment. ICMP checks stay outside capture.
CT/CA loopback TLS is captured; the provider CL connection is not captured.
Signed records include all native messages, tool arguments/results/projection,
usage and config; batch verification reconciles usage with the provider counters.

The systemd queue checkpoints verified batches, retains failed/interrupted
attempts, stops at the first execution/verification failure, yields for foreign
workloads and owns only exact labelled container names. It is independent of the
personal Mac/SSH session. It does not shut down the shared EC2 instance.

Revision v1 used 256 output tokens/request. After five successful uncommitted
rows, its first q=4 audit request returned provider ModelErrorException before
any model tool action. AWS documents this exception for maxTokens exhaustion.
Revision v2 increases only the output ceiling to 512; v1 evidence is retained
and is not counted in verified batches. Model/retry/request/tool limits remain.

Revision v2 produced 108 verified rows, then stopped on an extra read_flows call
after an I15 audit was already done. The bound rejected that call before MCP
execution; cleanup passed. Revision v3 ends at the required-tools boundary,
without making a post-result model request. Its state is required_tools_completed,
not model_completed; fulfillment remains independently checked. This optional
policy requires the trusted exact IntentBoundTools wrapper and does not change
the default agent loop. V2/V3 rows are separate statistical cohorts. Remaining
batches 007..020 target252 V3 rows; the six V2 batches are not rerun or pooled.
Automatic follow-up uses only V3 versus its 14 matched MCP session seeds (504 rows).
