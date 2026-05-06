"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown.

    TODO(student): Add richer analysis, examples, screenshots, and trace links.
    """

    lines = ["# Benchmark Report", "", "| Run | Latency (s) | Cost (USD) | Quality | Notes |", "|---|---:|---:|---:|---|"]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")
    report = "\n".join(lines) + "\n\n"
    report += "## Failure Mode Analysis\n\n"
    report += "> [!IMPORTANT]\n"
    report += "> **Common Failure Mode**: In a multi-agent system using a Supervisor, every transition (Supervisor -> Agent or Agent -> Supervisor) counts as one iteration. With the default `MAX_ITERATIONS=6`, the workflow would often terminate prematurely right after the `Writer` agent finished, failing to reach the `Critic` stage or finalize the state correctly.\n"
    report += "> **Mitigation Strategy**: Increased the default `MAX_ITERATIONS` to 15 in `core/config.py` and updated the `.env` configuration. This ensures enough budget for a full Research -> Analysis -> Synthesis -> Critique cycle, even if one of the agents needs a retry.\n"
    
    return report
