"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and return a placeholder metric object.

    TODO(student): Add quality scoring, estimated token cost, citation coverage, and error rate.
    """

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    
    # Calculate total cost from agent traces
    total_cost = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost")
        total_cost += cost if cost is not None else 0.0
    for event in state.trace:
        if event["name"] == "baseline":
            cost = event["payload"].get("cost_usd")
            total_cost += cost if cost is not None else 0.0
            
    # Improved quality score: Check for length AND presence of citations [Title](URL)
    quality_score = 0.0
    if state.final_answer:
        # Base score for length (up to 5 points)
        length_score = min(5.0, len(state.final_answer) / 200.0)
        
        # Citation score: check for markdown links (up to 5 points)
        import re
        links = re.findall(r"\[.*?\]\(https?://.*?\)", state.final_answer)
        citation_score = min(5.0, len(links) * 1.5) # ~3+ citations for full score
        
        quality_score = length_score + citation_score
        
    metrics = BenchmarkMetrics(run_name=run_name, latency_seconds=latency, estimated_cost_usd=total_cost, quality_score=quality_score)
    return state, metrics
