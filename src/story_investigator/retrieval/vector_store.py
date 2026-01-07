"""Vector store for similarity search."""

from typing import List, Tuple

import faiss
import numpy as np

from story_investigator.errors import RetrievalError
from story_investigator.models import Message


class VectorStore:
    """In-memory vector store using FAISS for similarity search."""

    def __init__(self, dimension: int = 384):
        """Initialize vector store with FAISS index.
        
        Args:
            dimension: Dimension of the embedding vectors (default: 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.messages: List[Message] = []

    def add_messages(self, messages: List[Message], vectors: np.ndarray) -> None:
        """Add messages and their corresponding vectors to the store.
        
        Args:
            messages: List of Message objects to store.
            vectors: NumPy array of shape (n_messages, dimension) containing embeddings.
            
        Raises:
            RetrievalError: If dimensions don't match or array shape is invalid.
        """
        if len(messages) != vectors.shape[0]:
            raise RetrievalError(
                f"Number of messages ({len(messages)}) doesn't match number of vectors ({vectors.shape[0]})"
            )
        
        if vectors.shape[1] != self.dimension:
            raise RetrievalError(
                f"Vector dimension ({vectors.shape[1]}) doesn't match store dimension ({self.dimension})"
            )
        
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        else:
            vectors = vectors.astype(np.float32)
        
        self.index.add(vectors)
        self.messages.extend(messages)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Message, float]]:
        """Search for k most similar messages.
        
        Args:
            query_vector: Query embedding vector of shape (dimension,) or (1, dimension).
            k: Number of similar messages to retrieve.
            
        Returns:
            List of tuples (Message, distance) sorted by similarity (lowest distance first).
            
        Raises:
            RetrievalError: If store is empty, dimension mismatch, or search fails.
        """
        if self.index.ntotal == 0:
            return []
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        else:
            query_vector = query_vector.astype(np.float32)
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        if query_vector.shape[1] != self.dimension:
            raise RetrievalError(
                f"Query vector dimension ({query_vector.shape[1]}) doesn't match store dimension ({self.dimension})"
            )
        
        k = min(k, self.index.ntotal)
        
        try:
            distances, indices = self.index.search(query_vector, k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.messages):
                    results.append((self.messages[idx], float(distances[0][i])))
            return results
        except Exception as e:
            raise RetrievalError(f"Search failed: {e}") from e


