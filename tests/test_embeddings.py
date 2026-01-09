"""Tests for OpenAI embedding engine."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from story_investigator.errors import EmbeddingError
from story_investigator.retrieval.embeddings import EmbeddingEngine


class TestEmbeddingEngine:
    """Test suite for OpenAI-based EmbeddingEngine."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        with patch('story_investigator.retrieval.embeddings.OpenAI') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock embeddings response
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3]),
            ]
            mock_client.embeddings.create.return_value = mock_response
            
            yield mock_client

    def test_init_with_api_key(self, mock_openai_client):
        """Test initialization with explicit API key."""
        engine = EmbeddingEngine(api_key="test-key-123")
        
        assert engine.model_name == "text-embedding-3-small"
        assert engine.max_retries == 3

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises EmbeddingError."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(EmbeddingError, match="OpenAI API key not provided"):
                EmbeddingEngine()

    def test_embed_text_calls_openai_api(self, mock_openai_client):
        """Test that embed_text calls OpenAI API with correct parameters."""
        engine = EmbeddingEngine(api_key="test-key")
        
        result = engine.embed_text("test text")
        
        # Verify API was called
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args
        
        assert call_args.kwargs['input'] == ["test text"]
        assert call_args.kwargs['model'] == "text-embedding-3-small"
        assert result == [0.1, 0.2, 0.3]

    def test_embed_text_with_custom_model(self, mock_openai_client):
        """Test that custom model name is used."""
        engine = EmbeddingEngine(api_key="test-key", model_name="text-embedding-3-large")
        
        engine.embed_text("test")
        
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs['model'] == "text-embedding-3-large"

    def test_embed_text_caching(self, mock_openai_client):
        """Test that repeated calls for same text use cache."""
        engine = EmbeddingEngine(api_key="test-key")
        
        # First call
        result1 = engine.embed_text("same text")
        
        # Second call (should use cache)
        result2 = engine.embed_text("same text")
        
        # API should be called only once
        assert mock_openai_client.embeddings.create.call_count == 1
        assert result1 == result2
        assert engine.get_cache_size() == 1

    def test_embed_texts_batch(self, mock_openai_client):
        """Test batch embedding of multiple texts."""
        # Mock response for multiple texts
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
            MagicMock(embedding=[0.5, 0.6]),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        engine = EmbeddingEngine(api_key="test-key")
        
        texts = ["text1", "text2", "text3"]
        results = engine.embed_texts(texts)
        
        # Verify API was called once with all texts
        mock_openai_client.embeddings.create.assert_called_once()
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs['input'] == texts
        
        # Verify results
        assert len(results) == 3
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]
        assert results[2] == [0.5, 0.6]

    def test_embed_texts_partial_cache(self, mock_openai_client):
        """Test that embed_texts uses cache for some texts and fetches others."""
        # First response
        mock_response1 = MagicMock()
        mock_response1.data = [MagicMock(embedding=[0.1, 0.2])]
        
        # Second response (for uncached texts)
        mock_response2 = MagicMock()
        mock_response2.data = [
            MagicMock(embedding=[0.3, 0.4]),
            MagicMock(embedding=[0.5, 0.6]),
        ]
        
        mock_openai_client.embeddings.create.side_effect = [mock_response1, mock_response2]
        
        engine = EmbeddingEngine(api_key="test-key")
        
        # Cache first text
        engine.embed_text("text1")
        
        # Now embed batch with one cached and two new texts
        results = engine.embed_texts(["text1", "text2", "text3"])
        
        # Verify API was called twice (once for text1, once for text2+text3)
        assert mock_openai_client.embeddings.create.call_count == 2
        
        # Verify results
        assert len(results) == 3
        assert results[0] == [0.1, 0.2]  # From cache
        assert results[1] == [0.3, 0.4]  # From new call
        assert results[2] == [0.5, 0.6]  # From new call

    def test_embed_with_retry_on_rate_limit(self, mock_openai_client):
        """Test retry logic on rate limit error."""
        from openai import RateLimitError
        
        # Simulate rate limit on first call, success on second
        mock_openai_client.embeddings.create.side_effect = [
            RateLimitError("Rate limit exceeded", response=Mock(), body=None),
            MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]),
        ]
        
        engine = EmbeddingEngine(api_key="test-key", initial_retry_delay=0.01)
        
        result = engine.embed_text("test")
        
        # Verify retry happened
        assert mock_openai_client.embeddings.create.call_count == 2
        assert result == [0.1, 0.2, 0.3]

    def test_embed_fails_after_max_retries(self, mock_openai_client):
        """Test that embedding fails after max retries."""
        from openai import RateLimitError
        
        # Simulate persistent rate limit
        mock_openai_client.embeddings.create.side_effect = RateLimitError(
            "Rate limit exceeded", response=Mock(), body=None
        )
        
        engine = EmbeddingEngine(api_key="test-key", max_retries=2, initial_retry_delay=0.01)
        
        with pytest.raises(EmbeddingError, match="Failed to generate embeddings after 2 retries"):
            engine.embed_text("test")
        
        # Verify retries happened
        assert mock_openai_client.embeddings.create.call_count == 2

    def test_clear_cache(self, mock_openai_client):
        """Test that cache can be cleared."""
        engine = EmbeddingEngine(api_key="test-key")
        
        engine.embed_text("text1")
        engine.embed_text("text2")
        assert engine.get_cache_size() == 2
        
        engine.clear_cache()
        assert engine.get_cache_size() == 0

    def test_embed_batch_alias(self, mock_openai_client):
        """Test that embed_batch is an alias for embed_texts."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2]),
            MagicMock(embedding=[0.3, 0.4]),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        engine = EmbeddingEngine(api_key="test-key")
        
        results = engine.embed_batch(["text1", "text2"])
        
        assert len(results) == 2
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]

    def test_empty_text_handling(self, mock_openai_client):
        """Test that empty strings are handled properly."""
        engine = EmbeddingEngine(api_key="test-key")
        
        result = engine.embed_text("")
        
        # Verify API was called with empty string
        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs['input'] == [""]
        assert result == [0.1, 0.2, 0.3]

