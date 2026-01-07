"""Custom exceptions for the Story Investigator application."""


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the maximum allowed length."""
    pass


class StoryParseError(Exception):
    """Raised when the story file cannot be parsed."""
    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(Exception):
    """Raised when retrieval fails."""
    pass

