"""Custom exceptions for the Story Investigator application."""


class StoryInvestigatorError(Exception):
    """Base exception for all Story Investigator errors."""
    
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class PromptTooLongError(StoryInvestigatorError):
    """Raised when a prompt exceeds the maximum allowed length."""
    
    def __init__(self, prompt_length: int = 0, max_length: int = 3000, message: str = ""):
        self.prompt_length = prompt_length
        self.max_length = max_length
        if not message:
            message = f"Prompt length ({prompt_length}) exceeds maximum allowed length ({max_length})"
        super().__init__(message)


class StoryParseError(StoryInvestigatorError):
    """Raised when the story file cannot be parsed."""
    
    def __init__(self, message: str = ""):
        super().__init__(message)


class EmbeddingError(StoryInvestigatorError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str = ""):
        super().__init__(message)


class RetrievalError(StoryInvestigatorError):
    """Raised when retrieval fails."""
    
    def __init__(self, message: str = ""):
        super().__init__(message)


class LLMClientError(StoryInvestigatorError):
    """Raised when the LLM client encounters an error."""
    
    def __init__(self, message: str = ""):
        super().__init__(message)
