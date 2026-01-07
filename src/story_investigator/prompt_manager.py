"""PromptManager - Handles prompt construction and validation.

Enforces the 3000-character hard limit on prompts.
This module is developed using TDD (tests written first in tests/test_prompt_manager.py).
"""

from story_investigator.errors import PromptTooLongError


class PromptManager:
    """Manages prompt construction and enforces character limits.
    
    This class ensures that all prompts sent to the LLM do not exceed
    the maximum allowed length (default: 3000 characters).
    """

    def __init__(self, max_length: int = 3000):
        """Initialize PromptManager with a maximum prompt length.
        
        Args:
            max_length: Maximum allowed characters in a prompt (default: 3000).
        """
        self.max_length = max_length

    def validate_prompt(self, prompt: str) -> bool:
        """Validate that a prompt does not exceed the maximum length.
        
        Args:
            prompt: The prompt string to validate.
            
        Returns:
            True if prompt is valid (within limit).
            
        Raises:
            PromptTooLongError: If prompt length exceeds max_length.
        """
        if prompt is None:
            raise PromptTooLongError(prompt_length=0, max_length=self.max_length)
        prompt_length = len(prompt)
        if prompt_length > self.max_length:
            raise PromptTooLongError(prompt_length=prompt_length, max_length=self.max_length)
        return True

    def build_prompt(self, question: str, context: str = "") -> str:
        """Build a prompt from a question and optional context.
        
        Args:
            question: The user's question.
            context: Optional context (e.g., retrieved story chunks).
            
        Returns:
            Formatted prompt string that includes question and context.
            
        Raises:
            PromptTooLongError: If the resulting prompt exceeds max_length.
        """
        base_prompt = (
            f"Answer the following question based only on the provided context. "
            f"If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        self.validate_prompt(base_prompt)
        return base_prompt

