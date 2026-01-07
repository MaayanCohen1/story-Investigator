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
        raise NotImplementedError("This method must be implemented as part of TDD cycle")

