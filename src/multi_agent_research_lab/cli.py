"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline placeholder."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    state.final_answer = (
        "Baseline skeleton response. TODO(student): replace this with a real single-agent "
        "implementation and record latency/cost/quality metrics."
    )
    
    # 1. Search
    search_client = SearchClient()
    sources = search_client.search(query)
    sources_text = "\n".join([f"- [{s.title}]({s.url}): {s.snippet}" for s in sources])
    
    # 2. LLM Call
    llm_client = LLMClient()
    system_prompt = "You are a professional Researcher and Writer. Answer the query based on the search results."
    user_prompt = f"Query: {query}\n\nSearch Results:\n{sources_text}"
    
    llm_response = llm_client.complete(system_prompt, user_prompt)
    state.final_answer = llm_response.content
    state.add_trace_event("baseline", {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd})
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = "Research GraphRAG state-of-the-art and write a 500-word summary",
) -> None:
    """Run both baseline and multi-agent and generate a benchmark report."""
    from multi_agent_research_lab.evaluation.benchmark import run_benchmark
    from multi_agent_research_lab.evaluation.report import render_markdown_report
    import os
    
    _init()
    console.print("[bold green]Running Single-Agent Baseline...[/bold green]")
    def run_baseline(q: str):
        state = ResearchState(request=ResearchQuery(query=q))
        search_client = SearchClient()
        sources = search_client.search(q)
        sources_text = "\n".join([f"- [{s.title}]({s.url}): {s.snippet}" for s in sources])
        
        llm_client = LLMClient()
        system_prompt = "You are a professional Researcher and Writer. Answer the query based on the search results."
        user_prompt = f"Query: {q}\n\nSearch Results:\n{sources_text}"
        
        llm_response = llm_client.complete(system_prompt, user_prompt)
        state.final_answer = llm_response.content
        state.add_trace_event("baseline", {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd})
        return state

    baseline_state, baseline_metrics = run_benchmark("Single-Agent Baseline", query, run_baseline)
    
    console.print("[bold blue]Running Multi-Agent Workflow...[/bold blue]")
    def run_multi(q: str):
        state = ResearchState(request=ResearchQuery(query=q))
        workflow = MultiAgentWorkflow()
        return workflow.run(state)
        
    multi_state, multi_metrics = run_benchmark("Multi-Agent Graph", query, run_multi)
    
    report_md = render_markdown_report([baseline_metrics, multi_metrics])
    
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "benchmark_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    console.print(Panel.fit(report_md, title=f"Benchmark Complete - Check {report_path}"))

if __name__ == "__main__":
    app()
