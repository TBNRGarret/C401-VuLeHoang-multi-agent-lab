"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from langgraph.graph import StateGraph, END
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.agents.critic import CriticAgent

class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph."""

    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.app = self.build()

    def build(self):
        """Create a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)
        workflow.add_node("critic", self.critic.run)

        # Edges from workers go back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        # Conditional edges from supervisor
        def route_next(state: ResearchState) -> str:
            return state.route_history[-1] if state.route_history else "done"

        workflow.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END
            }
        )

        workflow.set_entry_point("supervisor")
        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        final_state_dict = self.app.invoke(state.model_dump())
        return ResearchState(**final_state_dict)
