"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentResult
from rich.console import Console

console = Console()

class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self):
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        state.record_route(self.name)
        console.print("  [Analyst] Identifying key claims and flagging weak evidence...")
        
        system_prompt = "You are an expert Data Analyst. Given research notes, extract key claims, compare viewpoints, and flag weak evidence or missing information."
        user_prompt = f"Original Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}"
        
        llm_response = self.llm_client.complete(system_prompt, user_prompt)
        state.analysis_notes = llm_response.content
        
        # Trace
        state.add_trace_event(self.name, {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd})
        state.agent_results.append(AgentResult(agent=self.name, content=state.analysis_notes, metadata={"cost": llm_response.cost_usd}))
        
        return state
