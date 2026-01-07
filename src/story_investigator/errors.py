class StoryInvestigatorError(Exception):
    """Base exception for the story-Investigator project."""
    pass

class PromptTooLongError(StoryInvestigatorError):
    """Raised when the prompt exceeds the character limit (3000 chars)."""
    pass

class StoryParseError(StoryInvestigatorError):
    """Raised when XML parsing fails."""
    pass

class EmbeddingError(StoryInvestigatorError):
    """Raised when embedding generation fails."""
    pass

class RetrievalError(StoryInvestigatorError):
    """Raised when vector search fails."""
    pass

# השגיאה שהייתה חסרה לך:
class LLMClientError(StoryInvestigatorError):
    """Raised when the LLM API call fails."""
    pass