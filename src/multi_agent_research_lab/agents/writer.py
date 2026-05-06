"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentResult
from rich.console import Console

console = Console()

class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self):
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        state.record_route(self.name)
        console.print(f"  [Writer] Synthesizing final response for {state.request.audience}...")
        
        system_prompt = f"You are an expert Technical Writer writing for {state.request.audience}. Synthesize a clear response with citations."
        user_prompt = f"Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}\n\nAnalysis Notes:\n{state.analysis_notes}"
        
        llm_response = self.llm_client.complete(system_prompt, user_prompt)
        state.final_answer = llm_response.content
        
        # Trace
        state.add_trace_event(self.name, {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd})
        state.agent_results.append(AgentResult(agent=self.name, content=state.final_answer, metadata={"cost": llm_response.cost_usd}))
        
        return state
