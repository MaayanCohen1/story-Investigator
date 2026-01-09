"""Embedding generation using OpenAI API."""

import logging
import os
import time
from random import uniform
from typing import List, Dict

from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

from story_investigator.errors import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates embeddings for text using OpenAI's embedding API.
    
    Uses the same embedding space as LightRAG's indexing/retrieval for consistent reranking.
    Includes in-memory caching and automatic retry with exponential backoff.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ):
        """Initialize embedding engine with OpenAI client.
        
        Args:
            model_name: OpenAI embedding model (default: text-embedding-3-small).
                       Also supports: text-embedding-3-large, text-embedding-ada-002
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            max_retries: Maximum number of retry attempts for transient errors.
            initial_retry_delay: Initial delay in seconds for exponential backoff.
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}") from e
        
        # In-memory cache: {text: embedding_vector}
        self._cache: Dict[str, List[float]] = {}
        
        logger.info(f"Initialized OpenAI EmbeddingEngine with model: {model_name}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text with caching and retry.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            Embedding vector as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails after retries.
        """
        if not text:
            text = ""
        
        # Check cache first
        if text in self._cache:
            logger.debug(f"Cache hit for text (length {len(text)})")
            return self._cache[text]
        
        # Generate embedding with retry logic
        embedding = self._embed_with_retry([text])[0]
        
        # Cache the result
        self._cache[text] = embedding
        
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch with caching.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embedding vectors, each as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails after retries.
        """
        if not texts:
            return []
        
        # Separate cached and uncached texts
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            if not text:
                text = ""
            
            if text in self._cache:
                embeddings.append(self._cache[text])
                logger.debug(f"Cache hit for text {i} (length {len(text)})")
            else:
                embeddings.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Fetch embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating {len(uncached_texts)} embeddings (cache miss)")
            new_embeddings = self._embed_with_retry(uncached_texts)
            
            # Fill in the placeholders and update cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                self._cache[texts[idx]] = embedding
        
        return embeddings

    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embedding API with exponential backoff retry.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            EmbeddingError: If all retry attempts fail.
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                
                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]
                
                if attempt > 0:
                    logger.info(f"Embedding succeeded on retry attempt {attempt + 1}")
                
                return embeddings
                
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.initial_retry_delay * (2 ** attempt) + uniform(0, 0.1)
                    logger.warning(
                        f"Embedding API error ({type(e).__name__}), "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Embedding failed after {self.max_retries} attempts")
            
            except Exception as e:
                # Non-retryable error
                raise EmbeddingError(
                    f"Failed to generate embeddings with model {self.model_name}: {e}"
                ) from e
        
        # All retries exhausted
        raise EmbeddingError(
            f"Failed to generate embeddings after {self.max_retries} retries. "
            f"Last error: {last_error}"
        ) from last_error

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._cache)
    
    # Alias for backward compatibility with existing code
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Alias for embed_texts() for backward compatibility."""
        return self.embed_texts(texts)
