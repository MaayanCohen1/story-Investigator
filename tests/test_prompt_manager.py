"""Tests for PromptManager - TDD approach.

Tests are written BEFORE implementing PromptManager.
Following Red-Green-Refactor cycle - RED PHASE: Tests should FAIL.
"""

import pytest

from story_investigator.errors import PromptTooLongError
from story_investigator.prompt_manager import PromptManager


class TestPromptManager:
    """Test suite for PromptManager class."""

    def test_prompt_under_limit_passes(self):
        """Test that a prompt under 3000 characters is accepted."""
        prompt_manager = PromptManager(max_length=3000)
        short_prompt = "A" * 2999
        result = prompt_manager.validate_prompt(short_prompt)
        assert result is True

    def test_prompt_over_limit_raises_error(self):
        """Test that a prompt over 3000 characters raises PromptTooLongError."""
        prompt_manager = PromptManager(max_length=3000)
        long_prompt = "A" * 3001
        with pytest.raises(PromptTooLongError):
            prompt_manager.validate_prompt(long_prompt)

    def test_prompt_exactly_at_limit_passes(self):
        """Test that a prompt exactly at 3000 characters is accepted."""
        prompt_manager = PromptManager(max_length=3000)
        limit_prompt = "A" * 3000
        result = prompt_manager.validate_prompt(limit_prompt)
        assert result is True

    def test_empty_prompt_passes(self):
        """Test that an empty string is accepted."""
        prompt_manager = PromptManager(max_length=3000)
        empty_prompt = ""
        result = prompt_manager.validate_prompt(empty_prompt)
        assert result is True

    def test_custom_limit_at_limit_passes(self):
        """Test that Dependency Injection works with custom max_length at limit."""
        custom_limit = 10
        prompt_manager = PromptManager(max_length=custom_limit)
        valid_prompt = "A" * custom_limit
        result = prompt_manager.validate_prompt(valid_prompt)
        assert result is True

    def test_custom_limit_over_limit_raises_error(self):
        """Test that Dependency Injection works with custom max_length over limit."""
        custom_limit = 10
        prompt_manager = PromptManager(max_length=custom_limit)
        invalid_prompt = "A" * (custom_limit + 1)
        with pytest.raises(PromptTooLongError):
            prompt_manager.validate_prompt(invalid_prompt)

    def test_validate_prompt_returns_true_on_success(self):
        """Test that validate_prompt explicitly returns True (not None)."""
        prompt_manager = PromptManager(max_length=3000)
        valid_prompt = "A" * 100
        result = prompt_manager.validate_prompt(valid_prompt)
        assert result is True

    def test_validate_prompt_returns_not_none_on_success(self):
        """Test that validate_prompt returns not None on success."""
        prompt_manager = PromptManager(max_length=3000)
        valid_prompt = "A" * 100
        result = prompt_manager.validate_prompt(valid_prompt)
        assert result is not None

    def test_validate_prompt_returns_bool_type(self):
        """Test that validate_prompt returns bool type."""
        prompt_manager = PromptManager(max_length=3000)
        valid_prompt = "A" * 100
        result = prompt_manager.validate_prompt(valid_prompt)
        assert type(result) is bool
