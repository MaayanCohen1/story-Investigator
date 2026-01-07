"""Text chunking strategies for naive RAG."""

from typing import List

from story_investigator.models import Message, MessageChunk


class MessageChunker:
    """Groups messages into chunks using a sliding window strategy."""

    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        """Initialize chunker with sliding window parameters.
        
        Args:
            chunk_size: Number of messages per chunk.
            overlap: Number of overlapping messages between consecutive chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_messages(self, messages: List[Message]) -> List[MessageChunk]:
        """Chunk messages into overlapping windows.
        
        Args:
            messages: List of Message objects to chunk.
            
        Returns:
            List of MessageChunk objects.
        """
        if not messages:
            return []
        
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(messages), step):
            chunk_messages = messages[i:i + self.chunk_size]
            if not chunk_messages:
                break
            
            combined_text = "\n".join(msg.body for msg in chunk_messages)
            
            chunk = MessageChunk(
                messages=chunk_messages,
                combined_text=combined_text
            )
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(messages):
                break
        
        return chunks

