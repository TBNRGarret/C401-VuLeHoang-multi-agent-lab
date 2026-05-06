# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| Single-Agent Baseline | 12.97 | 0.0006 | 5.0 |  |
| Multi-Agent Graph | 45.18 | 0.0024 | 10.0 |  |

## Failure Mode Analysis

> [!IMPORTANT]
> **Common Failure Mode**: In a multi-agent system using a Supervisor, every transition (Supervisor -> Agent or Agent -> Supervisor) counts as one iteration. With the default `MAX_ITERATIONS=6`, the workflow would often terminate prematurely right after the `Writer` agent finished, failing to reach the `Critic` stage or finalize the state correctly.
> **Mitigation Strategy**: Increased the default `MAX_ITERATIONS` to 15 in `core/config.py` and updated the `.env` configuration. This ensures enough budget for a full Research -> Analysis -> Synthesis -> Critique cycle, even if one of the agents needs a retry.
