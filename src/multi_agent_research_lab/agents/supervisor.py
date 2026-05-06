"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.config import get_settings
from rich.console import Console

console = Console()

class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        state.record_route(self.name)
        
        settings = get_settings()
        max_iterations = settings.max_iterations
        if state.iteration >= max_iterations:
            console.print(f"[bold red]![/bold red] Max iterations reached ({max_iterations}). Stopping.")
            state.route_history.append("done")
            return state

        # Simple state machine routing
        if not state.research_notes:
            next_agent = "researcher"
        elif not state.analysis_notes:
            next_agent = "analyst"
        elif not state.final_answer:
            next_agent = "writer"
        elif "critic" not in state.route_history[-2:]: # if critic didn't run recently
            next_agent = "critic"
        elif any("FAIL" in error for error in state.errors) and state.iteration < max_iterations - 2:
            next_agent = "writer" # rewrite based on errors
        else:
            next_agent = "done"

        console.print(f"[bold cyan]>>> Supervisor:[/bold cyan] Routing to [bold yellow]{next_agent}[/bold yellow] (Iteration {state.iteration})")
        state.route_history.append(next_agent)
        
        state.add_trace_event(self.name, {"next": next_agent, "iteration": state.iteration})
        return state
