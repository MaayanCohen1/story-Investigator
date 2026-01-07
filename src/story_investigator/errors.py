"""Custom exceptions for the Story Investigator application."""


class PromptTooLongError(Exception):
    """Raised when a prompt exceeds the maximum allowed length."""
    pass

