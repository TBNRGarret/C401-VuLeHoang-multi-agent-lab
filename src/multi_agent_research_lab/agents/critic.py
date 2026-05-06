"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentResult
from rich.console import Console

console = Console()

class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self):
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        state.record_route(self.name)
        console.print("  [Critic] Validating output against research notes...")
        
        system_prompt = "You are a Critic. Review the final answer against the research notes. Does it answer the query? Are there hallucinations? Return PASS or FAIL along with your feedback."
        user_prompt = f"Query: {state.request.query}\nResearch Notes: {state.research_notes}\nFinal Answer: {state.final_answer}"
        
        llm_response = self.llm_client.complete(system_prompt, user_prompt)
        feedback = llm_response.content
        
        if "FAIL" in feedback:
            state.errors.append(feedback)
            
        state.add_trace_event(self.name, {"input_tokens": llm_response.input_tokens, "output_tokens": llm_response.output_tokens, "cost_usd": llm_response.cost_usd, "feedback": feedback})
        state.agent_results.append(AgentResult(agent=self.name, content=feedback, metadata={"cost": llm_response.cost_usd}))
        
        return state
