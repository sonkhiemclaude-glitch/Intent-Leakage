# Linux testbed components

These are experiment components for an isolated Linux laboratory, not a remote-machine launcher. Supply your own controller, topology, configuration and optional provider credentials. The public release contains no SSH target, private key or host-access inventory.

`ec2_testbed/` and `live_pilot/` contain the container build/runtime components; `mcp_pilot/` and `native_pilot/` implement the deterministic and native tool loops. `application_pilot/` collects bounded application bodies. `readonly_bridge/` declares the read-only engineering profile. `overnight/verify_batch.py` verifies signed instrumentation batches without credentials; `common.py` contains the tested ownership and scheduling guards.

The measured matched workload has three operations and scopes 1/2/4. Fresh controller/topology sessions, explicit OpenFlow 1.3, phase-separated TLS1.2 endpoints and bounded dataplane checks are important experimental settings. The separate read-only tunnel uses loopback mTLS1.3 with fixed 4096-byte cells, 256 rounds and a10ms period. Do not treat it as a completed state-changing defense.

Use project-specific containers and an exclusive laboratory window. Global network cleanup, host firewall changes and shared-container pruning are not part of the measurement procedure. `scripts/start_prepared_application.py` retains a frozen-source validator; its historical remote launcher is intentionally outside this portable release, so provision the Linux components directly instead of invoking its `--run` path.
