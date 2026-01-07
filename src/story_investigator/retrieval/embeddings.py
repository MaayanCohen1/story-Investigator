"""Embedding generation for naive RAG."""

from typing import List

from sentence_transformers import SentenceTransformer

from story_investigator.errors import EmbeddingError


class EmbeddingEngine:
    """Generates embeddings for text using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding engine with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {e}") from e

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single text.
        
        Args:
            text: Text to generate embedding for.
            
        Returns:
            Embedding vector as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text:
            text = ""
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to generate embeddings for.
            
        Returns:
            List of embedding vectors, each as a list of floats.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e


