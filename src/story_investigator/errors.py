"""Custom exceptions for the Story Investigator application."""


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the maximum allowed length."""
    pass


class StoryParseError(Exception):
    """Raised when the story file cannot be parsed."""
    pass

