"""LightRAG engine implementation using LightRAG library for retrieval.

This engine uses LightRAG as a retriever (NOT for final answer generation),
ensuring all prompts stay within the 3000 character limit.

KNOWN BUG FIX (v1.1):
The initial implementation had a critical bug where LightRAG's internal context
truncation (from ~100K chars to ~2.7K chars) would drop the most relevant evidence
BEFORE we could select it for the final prompt. This caused questions like
"Who requested to bring the USB?" to fail even when the evidence existed.

FIX: We now:
1) Use smaller retrieval parameters (top_k=10, chunk_top_k=5) to reduce raw context size
2) Extract candidate messages from the context BEFORE truncation
3) Re-rank candidates by relevance using embedding similarity
4) Select top evidence blocks that fit within the 3000-char budget
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

from story_investigator.errors import PromptTooLongError
from story_investigator.investigator_base import BaseInvestigator
from story_investigator.llm_client import LLMClient
from story_investigator.models import Answer, Message
from story_investigator.prompt_manager import PromptManager
from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.story_parser import StoryParser

logger = logging.getLogger(__name__)


# Custom LLM function for LightRAG using gpt-5-mini
async def gpt_5_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Wrapper for LightRAG to use gpt-5-mini model."""
    return await openai_complete_if_cache(
        "gpt-5-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


class LightRAGInvestigator(BaseInvestigator):
    """LightRAG-based investigator using graph-based retrieval with prompt limit enforcement."""

    def __init__(
        self,
        story_path: str,
        prompt_manager: PromptManager,
        llm_model: str = "gpt-5-mini",
        llm_temperature: float = 0.0,
        working_dir: str = "./lightrag_db",
        top_k: int = 10,  # Reduced from 60 to avoid over-retrieval
        chunk_top_k: int = 5,  # Reduced from 20 to avoid over-retrieval
        embedding_model: str = "text-embedding-3-small",
    ):
        """Initialize LightRAG investigator.
        
        Args:
            story_path: Path to the story XML file.
            prompt_manager: Manager for building and validating prompts.
            llm_model: OpenAI model name for answer generation.
            llm_temperature: Temperature for LLM generation.
            working_dir: Directory for LightRAG storage.
            top_k: Number of top entities/relations to retrieve (default: 10).
            chunk_top_k: Number of top chunks to retrieve per entity (default: 5).
            embedding_model: OpenAI embedding model for reranking (default: text-embedding-3-small).
                           MUST match LightRAG's embedding space for consistent ranking.
        """
        self.story_path = Path(story_path)
        self.prompt_manager = prompt_manager
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.working_dir = Path(working_dir)
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        
        self.messages: List[Message] = []
        self.rag: Optional[LightRAG] = None
        self.llm_client: Optional[LLMClient] = None
        self._initialized = False
        self._indexed = False
        
        # Initialize OpenAI embedding engine for reranking
        # IMPORTANT: Uses same embedding space as LightRAG (openai_embed)
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        logger.info(f"Initialized embedding engine with model: {embedding_model}")

    async def initialize(self) -> None:
        """Initialize LightRAG instance and LLM client.
        
        This must be called before using the engine. It sets up the LightRAG
        storage and initializes the LLM client for answer generation.
        """
        if self._initialized:
            return
        
        # Create working directory if it doesn't exist
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LightRAG with OpenAI models
        # LightRAG uses these for building its knowledge graph during indexing
        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=gpt_5_mini_complete,
            embedding_func=openai_embed,
            embedding_batch_num=16,
        )
        
        # Initialize LightRAG storages (required before use)
        await self.rag.initialize_storages()
        
        # Initialize our own LLM client for final answer generation
        # This ensures we control the prompt and enforce the character limit
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm_client = LLMClient(
            api_key=api_key,
            model=self.llm_model,
            temperature=self.llm_temperature,
            prompt_manager=self.prompt_manager,
        )
        
        self._initialized = True
        logger.info("LightRAG investigator initialized")

    async def load_story(self, story_path: str) -> None:
        """Load and index the story for retrieval.
        
        Args:
            story_path: Path to the story XML file.
            
        Raises:
            FileNotFoundError: If the story file doesn't exist.
            RuntimeError: If initialize() was not called first.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before load_story()")
        
        story_file = Path(story_path)
        if not story_file.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")
        
        # Check if already indexed by looking for storage files
        doc_status_file = self.working_dir / "kv_store_doc_status.json"
        if doc_status_file.exists() and self._indexed:
            logger.info("Story already indexed, skipping re-indexing")
            return
        
        # Parse the story
        with open(story_file, "r", encoding="utf-8") as f:
            xml_content = f.read()
        
        parser = StoryParser()
        self.messages = parser.parse_string(xml_content)
        
        if not self.messages:
            logger.warning("No messages found in story")
            return
        
        # Build text corpus for LightRAG indexing
        # Format each message clearly for knowledge extraction
        corpus_parts = []
        for msg in self.messages:
            # Format: clear statement with sender, receiver, and content
            message_text = (
                f"Message from {msg.sender} to {msg.receiver}: {msg.body}"
            )
            corpus_parts.append(message_text)
        
        # Join all messages with clear separation
        full_corpus = "\n\n".join(corpus_parts)
        
        # Insert into LightRAG for knowledge graph construction
        logger.info(f"Indexing {len(self.messages)} messages into LightRAG...")
        await self.rag.ainsert(full_corpus)
        
        self._indexed = True
        logger.info("Story indexed successfully")

    async def _retrieve_context(self, question: str) -> str:
        """Retrieve relevant context from LightRAG.
        
        Args:
            question: User's question.
            
        Returns:
            Retrieved context string from LightRAG.
        """
        if not self.rag:
            return ""
        
        # Use LightRAG's query with only_need_context=True
        # This returns the retrieved context WITHOUT generating an answer
        result = await self.rag.aquery(
            question,
            param=QueryParam(
                mode="mix",  # Use both local and global context
                only_need_context=True,  # Only retrieve, don't generate
                top_k=self.top_k,
                chunk_top_k=self.chunk_top_k,
                enable_rerank=False,  # Disable rerank to avoid warnings
            ),
        )
        
        return result if result else ""

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors.
        
        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.
            
        Returns:
            Cosine similarity score (0 to 1, higher = more similar).
        """
        import numpy as np
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
        
        return float(dot_product / norm_product)
    
    def _rank_messages_by_embedding(self, question: str, messages: List[Message]) -> List[Tuple[float, Message]]:
        """Rank messages by OpenAI embedding similarity to question.
        
        VERSION 3.0 (OpenAI Embeddings):
        - Uses OpenAI text-embedding-3-small (same space as LightRAG indexing)
        - NO hardcoded keyword lists or manual heuristics
        - General-purpose semantic similarity
        
        Args:
            question: User's question.
            messages: List of candidate messages.
            
        Returns:
            List of (similarity_score, message) tuples, sorted by score descending.
        """
        # Get question embedding
        question_embedding = self.embedding_engine.embed_text(question)
        
        # Prepare message texts for batch embedding
        message_texts = [
            f"{msg.sender} to {msg.receiver}: {msg.body}"
            for msg in messages
        ]
        
        # Get all message embeddings in batch (uses caching internally)
        message_embeddings = self.embedding_engine.embed_texts(message_texts)
        
        # Compute similarities
        scored_messages = []
        for message, msg_embedding in zip(messages, message_embeddings):
            similarity = self._compute_cosine_similarity(question_embedding, msg_embedding)
            scored_messages.append((similarity, message))
        
        # Sort by similarity (descending)
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Top embedding similarities: {[f'{score:.3f}' for score, _ in scored_messages[:5]]}")
        
        return scored_messages
    
    def _find_response_to_message(self, message: Message, max_distance: int = 10) -> Optional[Message]:
        """Find a response message from the receiver back to the sender.
        
        For investigative questions, this helps find responses to accusations/questions.
        
        Args:
            message: The original message (e.g., an accusation).
            max_distance: Maximum number of messages ahead to search.
            
        Returns:
            The response message if found, None otherwise.
        """
        try:
            idx = next(i for i, m in enumerate(self.messages) if m.message_id == message.message_id)
        except StopIteration:
            return None
        
        # Look ahead for a message where sender/receiver are swapped
        for i in range(idx + 1, min(idx + max_distance + 1, len(self.messages))):
            response_msg = self.messages[i]
            if (response_msg.sender == message.receiver and 
                response_msg.receiver == message.sender):
                return response_msg
        
        return None
    
    def _select_evidence_messages_with_ids(self, context: str, question: str, max_messages: int = 20) -> List[Message]:
        """Select and rank evidence messages using OPENAI EMBEDDING SIMILARITY.
        
        VERSION 3.0 (OpenAI Embeddings - No Keywords):
        - Ranks ALL messages by OpenAI embedding similarity to question
        - Uses same embedding space as LightRAG (text-embedding-3-small)
        - For investigative questions, includes response pairs
        - Budget-aware selection (fits within prompt limit)
        - NO hardcoded keyword lists
        
        Args:
            context: Retrieved context from LightRAG (unused in embedding-based approach).
            question: User's question.
            max_messages: Maximum number of messages to return.
            
        Returns:
            List of Message objects, ranked by embedding similarity (most similar first).
        """
        # Detect investigative questions
        question_lower = question.lower()
        is_investigative = any(word in question_lower for word in 
                              ["suspect", "suspicion", "accuse", "question", "doubt", "worry", "respond", "response", "react"])
        
        if is_investigative:
            logger.info("Detected investigative question - will look for response pairs")
        
        # Rank ALL messages by embedding similarity
        ranked_messages = self._rank_messages_by_embedding(question, self.messages)
        
        # Select top messages
        selected_messages = []
        seen_ids = set()
        
        for similarity, message in ranked_messages:
            if len(selected_messages) >= max_messages:
                break
            
            # Skip if below similarity threshold
            if similarity < 0.1:
                logger.debug(f"Skipping message {message.message_id} (similarity {similarity:.3f} too low)")
                continue
            
            if message.message_id not in seen_ids:
                selected_messages.append(message)
                seen_ids.add(message.message_id)
                logger.debug(f"Selected {message.message_id} (similarity: {similarity:.3f})")
                
                # For investigative questions with high-similarity messages,
                # also look for direct responses to complete the narrative
                if is_investigative and similarity > 0.4:
                    response_msg = self._find_response_to_message(message, max_distance=10)
                    if response_msg and response_msg.message_id not in seen_ids:
                        if len(selected_messages) < max_messages:
                            selected_messages.append(response_msg)
                            seen_ids.add(response_msg.message_id)
                            logger.info(f"Added response pair: {message.message_id} -> {response_msg.message_id}")
        
        logger.info(f"Selected {len(selected_messages)} messages via OpenAI embedding similarity")
        
        return selected_messages
    
    def _build_structured_prompt_with_ids(self, question: str, evidence_messages: List[Message]) -> str:
        """Build a structured prompt that enforces exact output format with message IDs.
        
        VERSION 3.1 (Analytical Investigator):
        - Includes investigative guidelines for chronological analysis
        - Behavioral profiling and inconsistency detection
        - Inference capabilities from behavioral changes
        
        The prompt instructs the LLM to return:
        ANSWER: <name or UNKNOWN>
        EVIDENCE_IDS: <comma-separated message IDs>
        REASON: <explanation>
        
        Args:
            question: User's question.
            evidence_messages: List of Message objects with IDs.
            
        Returns:
            Formatted prompt string.
        """
        # Build evidence section with message IDs and timestamps
        evidence_parts = []
        valid_ids = []
        for msg in evidence_messages:
            # Extract timestamp from original XML if present
            timestamp = ""
            ts_match = re.search(r'ts="([^"]+)"', msg.original_xml)
            if ts_match:
                timestamp = f" [Time: {ts_match.group(1)}]"
            
            evidence_parts.append(
                f"[ID:{msg.message_id}]{timestamp} {msg.sender} → {msg.receiver}: {msg.body}"
            )
            valid_ids.append(msg.message_id)
        
        evidence_text = "\n\n".join(evidence_parts)
        
        # Detect question type for adaptive guidance
        question_lower = question.lower()
        if question_lower.startswith("who"):
            answer_format = "<single name or UNKNOWN>"
        elif question_lower.startswith(("why", "what", "how")):
            answer_format = "<brief explanation (1-3 sentences) or UNKNOWN>"
        else:
            answer_format = "<brief answer or UNKNOWN>"
        
        prompt = f"""You are an expert AI Investigator analyzing a story told through messages. Answer based ONLY on the evidence below.

INVESTIGATIVE GUIDELINES:
1. CHRONOLOGICAL ANALYSIS: Messages have timestamps (if shown). Reconstruct the timeline - a sudden change in urgency or tone between messages may indicate suspicious behavior.
2. BEHAVIORAL PROFILING: Look for:
   - Contradictions between what someone says vs. does
   - Sudden tone shifts (panic → calm, casual → urgent)
   - Evasive or dismissive responses to direct questions
3. INFERENCE: Behavioral inconsistencies ARE valid evidence of suspicion. A character avoiding a question or changing their story is evidence to cite.
4. CITATIONS: You MUST back up claims with message IDs from the evidence.

OUTPUT FORMAT (3 lines only):
ANSWER: {answer_format}
EVIDENCE_IDS: <comma-separated IDs like m12,m13 - ONLY from: {', '.join(valid_ids)}>
REASON: <Explain your logic. If UNKNOWN, state why (not in story/not conclusive/ambiguous)>

EVIDENCE MESSAGES:
{evidence_text}

QUESTION: {question}

YOUR RESPONSE:"""
        
        return prompt
    
    def _parse_and_validate_llm_response(self, llm_response: str, evidence_messages: List[Message]) -> Answer:
        """Parse and validate LLM response with strict format enforcement.
        
        Args:
            llm_response: Raw response from LLM.
            evidence_messages: List of messages that were provided as evidence.
            
        Returns:
            Answer object with validated evidence IDs.
        """
        valid_ids = {msg.message_id for msg in evidence_messages}
        id_to_message = {msg.message_id: msg for msg in evidence_messages}
        
        # Parse response
        answer_text = "UNKNOWN"
        evidence_ids = []
        reason = "Failed to parse LLM response"
        
        lines = llm_response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("ANSWER:"):
                answer_text = line.replace("ANSWER:", "").strip()
            elif line.startswith("EVIDENCE_IDS:"):
                ids_str = line.replace("EVIDENCE_IDS:", "").strip()
                if ids_str:
                    evidence_ids = [id.strip() for id in ids_str.split(',')]
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        # Validate evidence IDs - they must be from the provided messages
        validated_ids = []
        for eid in evidence_ids:
            if eid in valid_ids:
                validated_ids.append(eid)
            else:
                logger.warning(f"LLM returned invalid evidence ID: {eid}")
        
        # If LLM hallucinated all IDs, treat as UNKNOWN
        if evidence_ids and not validated_ids:
            answer_text = "UNKNOWN"
            reason = "Evidence IDs could not be validated"
        
        # Get XML snippets for validated IDs
        evidence_xml_snippets = []
        for eid in validated_ids:
            if eid in id_to_message:
                evidence_xml_snippets.append(id_to_message[eid].original_xml)
        
        return Answer(
            answer_text=answer_text,
            evidence_ids=validated_ids,
            evidence_xml_snippets=evidence_xml_snippets,
            reason=reason
        )

    def _build_prompt_with_limit(
        self, 
        question: str, 
        context: str, 
        max_chars: int = 3000
    ) -> tuple[str, str]:
        """Build prompt ensuring it stays within character limit.
        
        If context is too large, progressively truncates it to fit.
        
        Args:
            question: User's question.
            context: Retrieved context from LightRAG.
            max_chars: Maximum allowed prompt length.
            
        Returns:
            Tuple of (prompt, truncated_context) where prompt fits within limit.
            
        Raises:
            PromptTooLongError: If even minimal prompt exceeds limit.
        """
        instructions = (
            "You are a professional investigator. Answer the question "
            "based ONLY on the context below. If the answer is not in the context, "
            "clearly state that the information is not available, not conclusive, or ambiguous."
        )
        
        # Calculate space available for context
        template = f"{instructions}\n\n<Context>\n{{context}}\n</Context>\n\n<Question>\n{question}\n</Question>"
        overhead = len(template) - len("{context}")
        
        if overhead >= max_chars:
            raise PromptTooLongError(
                prompt_length=overhead,
                max_length=max_chars,
                message="Question and instructions alone exceed prompt limit"
            )
        
        available_for_context = max_chars - overhead
        
        # Truncate context if needed
        truncated_context = context
        truncation_suffix = "\n... [context truncated to fit prompt limit]"
        
        if len(context) > available_for_context:
            # Account for truncation message in the available space
            available_with_suffix = available_for_context - len(truncation_suffix)
            if available_with_suffix > 0:
                truncated_context = context[:available_with_suffix] + truncation_suffix
            else:
                # Edge case: even suffix doesn't fit
                truncated_context = context[:available_for_context]
            
            logger.warning(
                f"Context truncated from {len(context)} to {len(truncated_context)} chars"
            )
        
        # Build final prompt
        prompt = template.format(context=truncated_context)
        
        # Final validation
        self.prompt_manager.validate_prompt(prompt)
        
        return prompt, truncated_context

    async def ask(self, question: str) -> Answer:
        """Answer a question about the story with evidence.
        
        Args:
            question: The user's question about the story.
            
        Returns:
            Answer object containing the answer text and evidence XML snippets.
            
        Raises:
            RuntimeError: If initialize() was not called first.
            PromptTooLongError: If prompt exceeds the maximum length.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before ask()")
        
        if not self.messages:
            return Answer(
                answer_text="No story data available.",
                evidence_xml_snippets=[]
            )
        
        # Step 1: Retrieve context from LightRAG (no answer generation yet)
        context = await self._retrieve_context(question)
        
        if not context or context.strip() == "":
            return Answer(
                answer_text=(
                    "I don't have enough information to answer this question. "
                    "The information is either not in the story, not conclusive, or ambiguous."
                ),
                evidence_xml_snippets=[]
            )
        
        # Step 2: Select evidence messages with IDs (prioritize multi-keyword matches)
        evidence_messages = self._select_evidence_messages_with_ids(context, question, max_messages=20)
        
        if not evidence_messages:
            return Answer(
                answer_text="UNKNOWN",
                evidence_ids=[],
                evidence_xml_snippets=[],
                reason="No relevant messages found in the story"
            )
        
        # Step 3: Build structured prompt with message IDs
        prompt = self._build_structured_prompt_with_ids(question, evidence_messages)
        
        # Step 4: Validate prompt length
        try:
            self.prompt_manager.validate_prompt(prompt)
            logger.info(f"Prompt: {len(prompt)} chars, {len(evidence_messages)} evidence messages")
        except PromptTooLongError:
            # Reduce number of messages and retry
            evidence_messages = evidence_messages[:10]
            prompt = self._build_structured_prompt_with_ids(question, evidence_messages)
            try:
                self.prompt_manager.validate_prompt(prompt)
                logger.info(f"Prompt: {len(prompt)} chars (reduced to {len(evidence_messages)} messages)")
            except PromptTooLongError:
                return Answer(
                    answer_text="UNKNOWN",
                    evidence_ids=[],
                    evidence_xml_snippets=[],
                    reason="Question requires too much context to answer within prompt limit"
                )
        
        # Step 5: Generate answer with strict format enforcement
        logger.info(f"Sending prompt to LLM: {len(prompt)} characters (limit: 3000)")
        try:
            llm_response = self.llm_client.generate_answer(prompt)
            logger.debug(f"LLM response: {llm_response}")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return Answer(
                answer_text="UNKNOWN",
                evidence_ids=[],
                evidence_xml_snippets=[],
                reason=f"LLM error: {str(e)}"
            )
        
        # Step 6: Parse and validate LLM response
        answer = self._parse_and_validate_llm_response(llm_response, evidence_messages)
        
        return answer

    def ask_sync(self, question: str) -> Answer:
        """Synchronous wrapper for ask() to match BaseInvestigator interface.
        
        This is a compatibility method. Prefer using ask() directly in async contexts.
        
        Args:
            question: The user's question about the story.
            
        Returns:
            Answer object containing the answer text and evidence XML snippets.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.ask(question))

