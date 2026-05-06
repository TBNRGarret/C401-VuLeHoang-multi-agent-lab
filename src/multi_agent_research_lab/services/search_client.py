"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


from multi_agent_research_lab.core.config import get_settings
from tavily import TavilyClient
from multi_agent_research_lab.core.schemas import SourceDocument

class SearchClient:
    """Provider-agnostic search client skeleton."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using Tavily."""
        results = []
        settings = get_settings()
        api_key = settings.tavily_api_key
        if not api_key:
            return [SourceDocument(title="Search Mock", snippet="[Mock] Please provide TAVILY_API_KEY for real search results.")]
            
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(query=query, search_depth="advanced", max_results=max_results)
            
            for r in response.get("results", []):
                results.append(SourceDocument(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")
                ))
        except Exception as e:
            results.append(SourceDocument(
                title="Search Error",
                snippet=f"Failed to search: {e}"
            ))
        return results
