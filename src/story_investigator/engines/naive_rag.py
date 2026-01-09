"""Naive RAG implementation using vector embeddings."""

import logging
from pathlib import Path
from typing import List

import numpy as np

from story_investigator.errors import PromptTooLongError
from story_investigator.investigator_base import BaseInvestigator
from story_investigator.llm_client import LLMClient
from story_investigator.models import Answer, Message, MessageChunk
from story_investigator.prompt_manager import PromptManager
from story_investigator.retrieval.chunking import MessageChunker
from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.retrieval.vector_store import VectorStore
from story_investigator.story_parser import StoryParser

logger = logging.getLogger(__name__)


class NaiveRAGInvestigator(BaseInvestigator):
    """Naive RAG implementation using chunking, embeddings, and vector search."""

    def __init__(
        self,
        story_path: str,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        chunker: MessageChunker,
        prompt_manager: PromptManager,
        llm_client: LLMClient,
        top_k: int = 7,
    ):
        """Initialize Naive RAG investigator.
        
        Args:
            story_path: Path to the story XML file.
            embedding_engine: Engine for generating embeddings.
            vector_store: Vector store for similarity search.
            chunker: Chunker for creating message chunks.
            prompt_manager: Manager for building prompts.
            llm_client: LLM client for generating answers.
            top_k: Number of top chunks to retrieve (will be dynamically reduced to fit prompt limit).
        """
        self.story_path = Path(story_path)
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.chunker = chunker
        self.prompt_manager = prompt_manager
        self.llm_client = llm_client
        self.top_k = top_k
        self.messages: List[Message] = []
        self.chunks: List[MessageChunk] = []
        self._chunk_to_messages: List[List[Message]] = []

    def load_story(self, story_path: str) -> None:
        """Load and index the story for retrieval.
        
        Args:
            story_path: Path to the story XML file.
        """
        story_file = Path(story_path)
        if not story_file.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")
        
        with open(story_file, "r", encoding="utf-8") as f:
            xml_content = f.read()
        
        parser = StoryParser()
        self.messages = parser.parse_string(xml_content)
        
        if not self.messages:
            return
        
        self.chunks = self.chunker.chunk_messages(self.messages)
        
        if not self.chunks:
            return
        
        chunk_texts = [chunk.combined_text for chunk in self.chunks]
        chunk_embeddings = self.embedding_engine.embed_batch(chunk_texts)
        chunk_vectors = np.array(chunk_embeddings, dtype=np.float32)
        
        # Store chunks directly in the vector store (metadata preserved)
        self.vector_store.add_chunks(self.chunks, chunk_vectors)

    def ask(self, question: str) -> Answer:
        """Answer a question about the story with evidence.
        
        Args:
            question: The user's question about the story.
            
        Returns:
            Answer object containing the answer text and evidence XML snippets.
        """
        if not self.chunks:
            return Answer(
                answer_text="No story data available.",
                evidence_ids=[],
                evidence_xml_snippets=[]
            )
        
        query_embedding = self.embedding_engine.embed_text(question)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        search_results = self.vector_store.search(query_vector, k=self.top_k)
        
        if not search_results:
            return Answer(
                answer_text="No relevant information found in the story.",
                evidence_ids=[],
                evidence_xml_snippets=[]
            )
        
        relevant_chunks = [chunk for chunk, _ in search_results]
        
        # Build prompt with dynamic chunk reduction to fit within character limit
        instructions = (
            "You are a professional investigator. Answer the question "
            "based ONLY on the context below.\n\n"
            "IMPORTANT: If the answer is not in the context, or if the evidence is not conclusive, "
            "you MUST return 'UNKNOWN' and explain why (e.g., 'not in story', 'not conclusive', 'ambiguous')."
        )
        
        # Try to fit as many chunks as possible within the prompt limit
        chunks_to_use = relevant_chunks
        for num_chunks in range(len(relevant_chunks), 0, -1):
            chunks_to_use = relevant_chunks[:num_chunks]
            
            # Build context from selected chunks
            context_texts = []
            for chunk in chunks_to_use:
                for message in chunk.messages:
                    context_texts.append(f"[{message.sender}] to [{message.receiver}]: {message.body}")
            
            context = "\n\n".join(context_texts)
            prompt = (
            f"{instructions}\n\n"
            f"<Context>\n{context}\n</Context>\n\n"
            f"<Question>\n{question}\n</Question>"
            )            
            # Check if this fits
            try:
                self.prompt_manager.validate_prompt(prompt)
                # It fits! Use this prompt
                logger.debug(
                    f"Sending {num_chunks} chunks ({len(context_texts)} messages) to LLM. "
                    f"Prompt length: {len(prompt)} chars"
                )
                break
            except PromptTooLongError:
                # Try with fewer chunks
                if num_chunks == 1:
                    # Even 1 chunk is too long - this shouldn't happen with our config
                    # but if it does, raise the error
                    raise
                continue
        
        # Collect evidence from chunks that were sent to the LLM
        evidence_xml_snippets = []
        seen_xml = set()
        
        for chunk in chunks_to_use:
            for message in chunk.messages:
                if message.original_xml not in seen_xml:
                    evidence_xml_snippets.append(message.original_xml)
                    seen_xml.add(message.original_xml)
        
        # Generate answer with the prompt that fits
        logger.info(f"Sending prompt to LLM: {len(prompt)} characters (limit: 3000)")
        try:
            answer_text = self.llm_client.generate_answer(prompt)
        except Exception as e:
            return Answer(
                answer_text=f"Error generating answer: {str(e)}",
                evidence_ids=[],
                evidence_xml_snippets=evidence_xml_snippets
            )
        
        return Answer(
            answer_text=answer_text,
            evidence_ids=[],
            evidence_xml_snippets=evidence_xml_snippets
        )

