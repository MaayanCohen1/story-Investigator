"""Naive RAG implementation using vector embeddings."""

from pathlib import Path
from typing import List

import numpy as np

from story_investigator.investigator_base import BaseInvestigator
from story_investigator.models import Answer, Message, MessageChunk
from story_investigator.prompt_manager import PromptManager
from story_investigator.llm_client import LLMClient
from story_investigator.retrieval.chunking import MessageChunker
from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.retrieval.vector_store import VectorStore
from story_investigator.story_parser import StoryParser


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
        top_k: int = 3,
    ):
        """Initialize Naive RAG investigator.
        
        Args:
            story_path: Path to the story XML file.
            embedding_engine: Engine for generating embeddings.
            vector_store: Vector store for similarity search.
            chunker: Chunker for creating message chunks.
            prompt_manager: Manager for building prompts.
            top_k: Number of top chunks to retrieve for answering.
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
        
        representative_messages = []
        for chunk in self.chunks:
            if chunk.messages:
                representative_messages.append(chunk.messages[0])
            else:
                raise ValueError("Chunk has no messages")
        
        self._chunk_to_messages = [chunk.messages for chunk in self.chunks]
        
        self.vector_store.add_messages(representative_messages, chunk_vectors)

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
                evidence_xml_snippets=[]
            )
        
        query_embedding = self.embedding_engine.embed_text(question)
        query_vector = np.array(query_embedding, dtype=np.float32)
        
        search_results = self.vector_store.search(query_vector, k=self.top_k)
        
        if not search_results:
            return Answer(
                answer_text="No relevant information found in the story.",
                evidence_xml_snippets=[]
            )
        
        retrieved_representative_messages = [msg for msg, _ in search_results]
        
        relevant_chunks = []
        for chunk_idx, chunk in enumerate(self.chunks):
            if chunk.messages and chunk.messages[0] in retrieved_representative_messages:
                relevant_chunks.append(chunk)
                if len(relevant_chunks) >= self.top_k:
                    break
        
        evidence_xml_snippets = []
        context_texts = []
        seen_xml = set()
        
        for chunk in relevant_chunks:
            for message in chunk.messages:
                if message.original_xml not in seen_xml:
                    evidence_xml_snippets.append(message.original_xml)
                    seen_xml.add(message.original_xml)
                context_texts.append(message.body)
        
        context = "\n\n".join(context_texts)
        
        try:
            instructions = (
            "You are a professional investigator. Your task is to answer the user's question "
            "based ONLY on the provided context below. If the information needed to answer "
            "is not present in the context, state clearly that you do not know."
            )
            prompt = (
            f"{instructions}\n\n"
            f"<Context>\n{context}\n</Context>\n\n"
            f"Question: {question}"
            )
            # Enforce prompt length limit before LLM call
            self.prompt_manager.validate_prompt(prompt)
            answer_text = self.llm_client.generate_answer(prompt)
        except Exception as e:
            return Answer(
                answer_text=f"Error preparing or generating answer: {str(e)}",
                evidence_xml_snippets=evidence_xml_snippets
            )
        
        return Answer(
            answer_text=answer_text,
            evidence_xml_snippets=evidence_xml_snippets
        )

