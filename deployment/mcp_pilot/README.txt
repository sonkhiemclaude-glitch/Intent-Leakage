Three-class MCP controller instrumentation pilot

pilot.py is a separately frozen variant of the repaired controller pilot. It
routes EVERY controller mutation, poll and assurance query through the registered
local stdio MCP server; it does not directly call the ONOS REST adapter. Separate
CT/CA HTTPS origins remain endpoint-separable. Parent/child are local processes
inside the isolated Mininet container, not a remotely exposed MCP HTTP server.
The control-plane backend of the MCP server is the same narrow ONOS adapter.

The repository's stdlib MCP implementation advertises tools/ping only and pins
protocol 2025-06-18. Six frozen tools cover host lookup/quarantine, owned-rule
release, CT polling, CA flow reads and policy audit. It is not a complete MCP SDK
or implementation of all fifteen paper classes. Runtime needs no new package or
Docker image. A separate client interoperability test with official Python MCP
SDK 1.26.0 passed against the local TLS API fixture; see
outputs/mcp_stdio_interop_latest.json and its pinned requirements/source snapshot. That fixture test is not an ONOS experiment.

The constructor's explicit external gate precedes process/network/file access.
Only the fixed internal server module can launch. No shell command comes from
configuration or a model. The child receives a minimal environment with ONOS
credentials, without AWS/Bedrock credentials. Only acknowledged rule IDs can be
cleaned on EOF. A failed POST with unknown outcome still needs independent reset
of the isolated testbed; EOF cleanup is not a full network-state guarantee.
MCP timeouts close the session without replaying the tool call. Compensation may
repeat owned DELETE requests. Frozen tool schemas are hashed at initialization.

Dataplane checks keep the v2 positive-convergence policy and strict quarantine
check. Signed per-run metadata additionally preserves every MCP RPC method/name,
request ID, elapsed time, outcome and response hash, outside attacker features.
The verifier requires valid MCP metadata plus the same full nine-cell batch,
zero capture drops, signatures/hashes and unchanged host-network snapshots.
The protocol revision is mcp_stdio_convergence_v1, kept separate from direct REST.

A bounded BedrockOnosToolAgent implementation is also available for the next
integration phase. It uses native Converse toolUse/toolResult messages, validates
an entire tool batch before mutation, rejects reused tool IDs, truncation,
unregistered operations, hidden SDK retries and exhausted budgets. It preserves
failed/unknown tool outcomes and never equates model completion with fulfillment.
Default per-run bounds are 4 model requests, 12 tool calls, 120 seconds checked
at SDK/RPC boundaries, 64 KiB compact JSON per model request and 32 KiB per
tool result. These are engineering defaults, not paper constants or exact token/
billing caps. Hard cancellation belongs to the deployment supervisor.
The model path passed SDK-double tests. An initial live probe was denied; after
key replacement, a 128-token run executed both tools but truncated its final
response. A bounded 256-token Nova Micro run then completed three model requests
and two native MCP tool calls against a local TLS fixture, correctly counted all
three flows, and verified cleanup preserving the foreign rule. This is not ONOS
or data-plane validation. See outputs/bedrock_native_tools_fixture_latest.json.
The deterministic MCP pilot does NOT call this agent or any LLM.

After the direct queue is complete and only if at least five minutes remain in
the assigned window, deploy ONE separate batch with --pilot-variant mcp,
--start-batch 4 --end-batch 4 and the unchanged --deadline-epoch. Use
--previous-receipt to pin that bound. Do not overlap testbeds, change the running
direct queue or extend the exclusive window. One failure stops the pilot.
A new deployment also has a maximum eight-minute batch timeout.
No EC2 stop/terminate or message to another person is performed by these scripts.

Real T4 result: 18/18 passed at 07:19:36 VN on October 21, 2026. 1,866 TLS
records, 338 all-phase MCP RPCs, 204 dataplane checks; no LLM/shaping. The
full feature/signature/hash report is at
outputs/intent-leakage-night-20261021T001618Z/batch_004/analysis/summary.json.
All services are disabled and containers removed; EC2 itself is not stopped.
