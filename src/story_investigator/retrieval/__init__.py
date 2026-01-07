"""Retrieval components for naive RAG."""

from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.retrieval.vector_store import VectorStore

__all__ = [
    "EmbeddingEngine",
    "VectorStore",
]

