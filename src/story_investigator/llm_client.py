"""LLM client for OpenAI API."""

import os
from typing import Optional

from openai import OpenAI

from story_investigator.errors import PromptTooLongError, LLMClientError
from story_investigator.prompt_manager import PromptManager


class LLMClient:
    """Wrapper for OpenAI API client with prompt length enforcement."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_manager: Optional[PromptManager] = None,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY is required")

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.prompt_manager = prompt_manager
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as exc:
            raise LLMClientError(f"Failed to initialize OpenAI client: {exc}") from exc

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer from the LLM after validating prompt length."""
        if self.prompt_manager:
            self.prompt_manager.validate_prompt(prompt)

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            return completion.choices[0].message.content
        except PromptTooLongError:
            raise
        except Exception as exc:
            raise LLMClientError(f"LLM generation failed: {exc}") from exc
