"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings

class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        settings = get_settings()
        api_key = settings.openai_api_key
        self.model = settings.openai_model
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        if not self.client:
            # Fallback or error if no key
            return LLMResponse(content="[LLM Mock Response] Please provide OPENAI_API_KEY to get real responses.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # very rough estimate for gpt-4o-mini: $0.150 / 1M input tokens, $0.600 / 1M output tokens
        cost_usd = (input_tokens / 1_000_000 * 0.150) + (output_tokens / 1_000_000 * 0.600)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd
        )
