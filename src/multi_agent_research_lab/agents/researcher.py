"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentResult
from rich.console import Console

console = Console()

class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self):
        self.search_client = SearchClient()
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        state.record_route(self.name)
        
        # 1. Search
        query = state.request.query
        console.print(f"  [Researcher] Searching for: [italic]{query}[/italic]...")
        sources = self.search_client.search(query, max_results=state.request.max_sources)
        state.sources.extend(sources)
        console.print(f"  [Researcher] Found {len(sources)} sources. Summarizing notes...")
        
        # 2. Summarize notes
        system_prompt = "You are a professional Researcher. Given the user's query and the search results, create concise research notes. Be objective and cite sources using [Title](URL)."
        
        sources_text = "\n".join([f"- [{s.title}]({s.url}): {s.snippet}" for s in sources])
        user_prompt = f"Query: {query}\n\nSearch Results:\n{sources_text}"
        
        llm_response = self.llm_client.complete(system_prompt, user_prompt)
        state.research_notes = llm_response.content
        
        # Trace
        state.add_trace_event(self.name, {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd})
        state.agent_results.append(AgentResult(agent=self.name, content=state.research_notes, metadata={"cost": llm_response.cost_usd}))
        
        return state
