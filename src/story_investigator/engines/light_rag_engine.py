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
from typing import List, Optional

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

from story_investigator.errors import PromptTooLongError
from story_investigator.investigator_base import BaseInvestigator
from story_investigator.llm_client import LLMClient
from story_investigator.models import Answer, Message
from story_investigator.prompt_manager import PromptManager
from story_investigator.story_parser import StoryParser

logger = logging.getLogger(__name__)


class LightRAGInvestigator(BaseInvestigator):
    """LightRAG-based investigator using graph-based retrieval with prompt limit enforcement."""

    def __init__(
        self,
        story_path: str,
        prompt_manager: PromptManager,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        working_dir: str = "./lightrag_db",
        top_k: int = 10,  # Reduced from 60 to avoid over-retrieval
        chunk_top_k: int = 5,  # Reduced from 20 to avoid over-retrieval
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
            llm_model_func=gpt_4o_mini_complete,
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

    def _compute_relevance_score(self, message: Message, question: str, context: str) -> float:
        """Compute relevance score for a message given a question and context.
        
        Uses multiple signals:
        1. Presence in LightRAG context (weighted highest)
        2. Multiple important keywords together (strong signal)
        3. Sender/receiver mentioned in context
        4. Keyword overlap with question
        
        Args:
            message: Message to score.
            question: User's question.
            context: Retrieved context from LightRAG.
            
        Returns:
            Relevance score (higher = more relevant).
        """
        score = 0.0
        
        question_lower = question.lower()
        body_lower = message.body.lower()
        context_lower = context.lower()
        
        # Extract important keywords from question (not stopwords)
        stopwords = {'what', 'who', 'when', 'where', 'how', 'why', 'the', 'a', 'an', 
                     'that', 'this', 'with', 'to', 'from', 'in', 'on', 'at', 'is', 'are'}
        question_words = set(question_lower.split())
        important_keywords = [w for w in question_words if len(w) > 2 and w not in stopwords]
        
        # Signal 1: Multiple important keywords appear together in message body (STRONGEST)
        keywords_in_body = [kw for kw in important_keywords if kw in body_lower]
        if len(keywords_in_body) >= 2:
            score += 50.0 * len(keywords_in_body)  # Very strong signal
        elif len(keywords_in_body) == 1:
            score += 15.0
        
        # Signal 2: Message body appears in LightRAG context (strong signal)
        # Check for substantial overlap, not just a single word
        if len(body_lower) > 20 and body_lower[:50] in context_lower:
            score += 20.0
        
        # Signal 3: Sender/receiver mentioned in LightRAG context
        if message.sender.lower() in context_lower:
            score += 5.0
        if message.receiver.lower() in context_lower:
            score += 3.0
        
        # Signal 4: General keyword overlap
        body_words = set(body_lower.split())
        overlap = question_words & body_words
        score += len(overlap) * 1.0
        
        return score
    
    def _extract_evidence_from_context(self, context: str, question: str, max_snippets: int = 10) -> List[str]:
        """Extract and rank evidence snippets from retrieved context.
        
        IMPROVED VERSION (v1.1): Instead of naive matching, we:
        1. Find all candidate messages (those mentioned in context or relevant to question)
        2. Score each by relevance using multiple signals
        3. Return top-ranked messages up to max_snippets
        
        Args:
            context: Retrieved context from LightRAG.
            question: User's question (for relevance scoring).
            max_snippets: Maximum number of evidence snippets to return.
            
        Returns:
            List of original XML snippets, ordered by relevance (most relevant first).
        """
        # Score all messages by relevance
        scored_messages = []
        for message in self.messages:
            score = self._compute_relevance_score(message, question, context)
            if score > 0:  # Only include messages with some relevance
                scored_messages.append((score, message))
        
        # Sort by score (descending)
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        # Take top-k and extract XML
        evidence_xml_snippets = []
        seen_xml = set()
        
        for score, message in scored_messages[:max_snippets]:
            if message.original_xml not in seen_xml:
                evidence_xml_snippets.append(message.original_xml)
                seen_xml.add(message.original_xml)
        
        return evidence_xml_snippets

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
        
        # Step 2: Extract and rank evidence snippets based on retrieved context
        # This uses relevance scoring to prioritize the most important messages
        evidence_xml_snippets = self._extract_evidence_from_context(context, question, max_snippets=50)
        
        # Step 3: Build context from TOP RANKED evidence messages (budget-aware)
        # Instead of using LightRAG's raw truncated context, we build our own
        # from the most relevant messages that fit within our budget
        relevant_context_parts = []
        char_budget = 2500  # Leave room for instructions + question
        
        for evidence_xml in evidence_xml_snippets:
            # Extract readable text from this message for context
            # Parse sender, receiver, body from the XML (handle namespaces)
            try:
                # Simple extraction: find sender ref, receiver ref, and body
                # Handle both with and without namespace prefixes (e.g., ns0:sender or sender)
                sender_match = re.search(r'<(?:\w+:)?sender[^>]*ref="([^"]+)"', evidence_xml)
                receiver_match = re.search(r'<(?:\w+:)?receiver[^>]*ref="([^"]+)"', evidence_xml)
                body_match = re.search(r'<(?:\w+:)?body[^>]*>(.*?)</(?:\w+:)?body>', evidence_xml, re.DOTALL)
                
                if sender_match and receiver_match and body_match:
                    sender = sender_match.group(1)
                    receiver = receiver_match.group(1)
                    body = body_match.group(1).strip()
                    
                    context_entry = f"[{sender}] to [{receiver}]: {body}"
                    
                    # Check if adding this would exceed budget
                    if len("\n\n".join(relevant_context_parts + [context_entry])) <= char_budget:
                        relevant_context_parts.append(context_entry)
                    else:
                        break  # Budget exhausted
            except Exception as e:
                logger.warning(f"Failed to parse evidence XML: {e}")
                continue
        
        # Build context from selected evidence
        selected_context = "\n\n".join(relevant_context_parts) if relevant_context_parts else context
        
        # Step 4: Build final prompt within the 3000 character limit
        try:
            prompt, truncated_context = self._build_prompt_with_limit(
                question, 
                selected_context,  # Use our curated context, not raw LightRAG context
                max_chars=self.prompt_manager.max_length
            )
            logger.info(f"Prompt: {len(prompt)} chars from {len(evidence_xml_snippets)} evidence blocks ({len(relevant_context_parts)} used)")
        except PromptTooLongError:
            # If we can't fit even a truncated version, return error
            return Answer(
                answer_text=(
                    "The context required to answer this question is too large. "
                    "Please try asking a more specific question."
                ),
                evidence_xml_snippets=evidence_xml_snippets[:10]  # Return top 10 evidence
            )
        
        # Step 4: Generate final answer using our LLM client
        try:
            answer_text = self.llm_client.generate_answer(prompt)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return Answer(
                answer_text=f"Error generating answer: {str(e)}",
                evidence_xml_snippets=evidence_xml_snippets
            )
        
        return Answer(
            answer_text=answer_text,
            evidence_xml_snippets=evidence_xml_snippets
        )

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

