"""Nano-GraphRAG engine implementation using nano-graphrag library for graph-based retrieval.

This engine uses nano-graphrag to build a knowledge graph from the story,
then retrieves relevant context for answering questions.
Uses OpenAI embeddings (text-embedding-3-small) and gpt-5-mini for consistency.
"""

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from openai import AsyncOpenAI

from story_investigator.errors import PromptTooLongError
from story_investigator.investigator_base import BaseInvestigator
from story_investigator.models import Answer, Message
from story_investigator.prompt_manager import PromptManager
from story_investigator.story_parser import StoryParser

# Import the existing cached LLM function from lightrag_engine
from lightrag.llm.openai import openai_complete_if_cache

logger = logging.getLogger(__name__)


# Custom OpenAI embedding function for nano-graphrag
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding_func(texts: list[str]) -> np.ndarray:
    """Custom embedding function using OpenAI text-embedding-3-small.
    
    Args:
        texts: List of texts to embed.
        
    Returns:
        Numpy array of embeddings (shape: [n_texts, 1536]).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = AsyncOpenAI(api_key=api_key)
    
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)
    logger.debug(f"Generated embeddings for {len(texts)} texts")
    return embeddings


# Custom LLM function for nano-graphrag with caching support
async def openai_llm_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = None,
    **kwargs
) -> str:
    """Custom LLM function using OpenAI with caching support.
    
    Args:
        prompt: User prompt.
        system_prompt: Optional system prompt.
        history_messages: Optional conversation history.
        **kwargs: Additional arguments.
        
    Returns:
        LLM response text.
    """
    # Handle parameter naming for OpenAI API compatibility
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    
    # Use the cached OpenAI completion function from lightrag
    response = await openai_complete_if_cache(
        model="gpt-5-mini",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        **kwargs
    )
    
    return response


class NanoGraphInvestigator(BaseInvestigator):
    """RAG engine using nano-graphrag for knowledge graph-based retrieval.
    
    This engine uses nano-graphrag to build a knowledge graph from the story,
    then retrieves relevant context for answering questions.
    Uses OpenAI embeddings (text-embedding-3-small) and gpt-4o-mini for consistency.
    """

    def __init__(
        self,
        story_path: str,
        prompt_manager: PromptManager,
        llm_model: str = "gpt-5-mini",
        llm_temperature: float = 0.0,
        working_dir: str = "./nano_graph_db",
    ):
        """Initialize Nano-GraphRAG investigator.
        
        Args:
            story_path: Path to the story XML file.
            prompt_manager: Manager for building and validating prompts.
            llm_model: OpenAI model name for answer generation (default: gpt-5-mini).
            llm_temperature: Temperature for LLM generation.
            working_dir: Directory for nano-graphrag storage.
        """
        self.story_path = Path(story_path)
        self.prompt_manager = prompt_manager
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.working_dir = Path(working_dir)
        
        self.messages: List[Message] = []
        self.rag: GraphRAG = None
        self._initialized = False
        self._indexed = False

    async def initialize(self) -> None:
        """Initialize nano-graphrag instance.
        
        This must be called before using the engine. It sets up the GraphRAG
        instance and injects our custom investigator prompt.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not set.
        """
        if self._initialized:
            logger.info("NanoGraphInvestigator already initialized")
            return
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Inject custom investigator system prompt before initializing GraphRAG
        from nano_graphrag.prompt import PROMPTS
        PROMPTS["local_rag_response"] = self._get_investigator_prompt()
        
        # Initialize GraphRAG with custom embedding and LLM functions
        self.rag = GraphRAG(
            working_dir=str(self.working_dir),
            embedding_func=openai_embedding_func,
            best_model_func=openai_llm_func,
            cheap_model_func=openai_llm_func,
            best_model_max_token_size=3000,
        )
        
        self._initialized = True
        logger.info(f"NanoGraphInvestigator initialized with working_dir: {self.working_dir}")

    def _get_investigator_prompt(self) -> str:
        """Get the custom investigator system prompt for nano-graphrag.
        
        Returns:
            Investigator system prompt with analytical guidelines.
            Uses nano-graphrag's expected format string variables:
            - {context_data}: The retrieved context (entities, relations, text chunks)
            - {response_type}: The desired response format/length
            
        Note: The user's query is passed separately to the LLM, not in the system prompt.
        """
        return """---Role---

You are an expert AI Investigator analyzing a story told through messages.

---INVESTIGATIVE GUIDELINES---

1. CHRONOLOGICAL ANALYSIS: Pay attention to timestamps and sequence of events. Sudden changes in urgency or tone may indicate suspicious behavior.

2. BEHAVIORAL PROFILING: Look for:
   - Contradictions between what someone says vs. does
   - Sudden tone shifts (panic → calm, casual → urgent)
   - Evasive or dismissive responses to direct questions

3. INFERENCE: Behavioral inconsistencies ARE valid evidence of suspicion. A character avoiding a question or changing their story is evidence to cite.

4. CITATIONS: You MUST back up claims with specific references to the messages.

If the context lacks sufficient information, clearly state the answer is "not in the story", "not conclusive", or "ambiguous".

---Goal---

Generate a response that answers the user's question based ONLY on the provided context information below.

---Target Response Format---

{response_type}

---Context Data---

{context_data}

---Instructions---

Use ONLY the context data above to answer the question. Cite specific messages and entities. If the answer is not supported by the context, explain why (not in story, not conclusive, ambiguous)."""

    async def load_story(self, story_path: str) -> None:
        """Load and index the story for retrieval.
        
        Args:
            story_path: Path to the story XML file.
            
        Raises:
            RuntimeError: If initialize() was not called first.
            FileNotFoundError: If story file doesn't exist.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before load_story()")
        
        story_file = Path(story_path)
        if not story_file.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")
        
        # Parse the story
        with open(story_file, "r", encoding="utf-8") as f:
            xml_content = f.read()
        
        parser = StoryParser()
        self.messages = parser.parse_string(xml_content)
        
        if not self.messages:
            logger.warning("No messages found in story")
            return
        
        logger.info(f"Parsed {len(self.messages)} messages from story")
        
        # Check if already indexed
        if self._indexed:
            logger.info("Story already indexed in nano-graphrag")
            return
        
        # Prepare text for nano-graphrag indexing
        # Combine all messages into a single document with structure
        story_text_parts = []
        for msg in self.messages:
            # Include message ID, sender, receiver, and body for context
            msg_text = f"Message {msg.message_id}: {msg.sender} sent to {msg.receiver}: {msg.body}"
            story_text_parts.append(msg_text)
        
        story_text = "\n\n".join(story_text_parts)
        
        # Insert into nano-graphrag
        logger.info(f"Indexing {len(self.messages)} messages into nano-graphrag...")
        await self.rag.ainsert(story_text)
        
        self._indexed = True
        logger.info("Story successfully indexed in nano-graphrag")

    async def ask(self, question: str) -> Answer:
        """Answer a question about the story with evidence.
        
        Args:
            question: The user's question about the story.
            
        Returns:
            Answer object containing the answer text and evidence XML snippets.
            
        Raises:
            RuntimeError: If initialize() was not called first.
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() before ask()")
        
        if not self.messages:
            return Answer(
                answer_text="UNKNOWN",
                evidence_ids=[],
                evidence_xml_snippets=[],
                reason="No story data has been loaded"
            )
        
        # Query nano-graphrag with local mode for focused retrieval
        logger.info(f"Querying nano-graphrag: {question}")
        
        try:
            # Use query with mode="local" for focused retrieval
            result = await self.rag.aquery(
                question,
                param=QueryParam(mode="local", top_k=10)
            )
            
            logger.info(f"Nano-graphrag returned {len(result)} characters")
            
            # Validate prompt length constraint
            if len(result) > 3000:
                logger.warning(f"Nano-graphrag result exceeds 3000 chars: {len(result)}")
                # Truncate if necessary while maintaining readability
                result = result[:2900] + "\n\n... (response truncated to meet 3000 character limit)"
            
            logger.info(f"Final answer length: {len(result)} characters (limit: 3000)")
            
            # Extract evidence from the result
            # Since nano-graphrag returns the final answer, we need to identify
            # relevant messages as evidence
            evidence_xml_snippets = self._extract_evidence_from_result(result, question)
            
            # Parse the result to extract answer components
            answer_text = result.strip()
            
            # Check if it's an "unknown" answer
            answer_lower = answer_text.lower()
            is_unknown = any(phrase in answer_lower for phrase in [
                "don't know", "do not know",
                "not in the story", "not in story",
                "not conclusive", "not enough information",
                "ambiguous", "insufficient", "cannot determine"
            ])
            
            if is_unknown:
                return Answer(
                    answer_text="UNKNOWN",
                    evidence_ids=[],
                    evidence_xml_snippets=evidence_xml_snippets,
                    reason=answer_text
                )
            
            return Answer(
                answer_text=answer_text,
                evidence_ids=[],
                evidence_xml_snippets=evidence_xml_snippets,
                reason=""
            )
            
        except Exception as e:
            logger.error(f"Error querying nano-graphrag: {e}", exc_info=True)
            return Answer(
                answer_text="UNKNOWN",
                evidence_ids=[],
                evidence_xml_snippets=[],
                reason=f"Error querying graph: {str(e)}"
            )

    def _extract_evidence_from_result(self, result: str, question: str) -> List[str]:
        """Extract relevant message XML snippets as evidence.
        
        Since nano-graphrag returns a text result, we need to identify which
        messages from the original story are most relevant as evidence.
        
        Uses keyword matching and relevance scoring to find the best evidence.
        
        Args:
            result: The result from nano-graphrag.
            question: The original question.
            
        Returns:
            List of XML snippets from the original messages.
        """
        # Extract keywords from question and result
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
                    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        # Get keywords from question
        question_words = [w.lower() for w in question.split() if len(w) > 3 and w.lower() not in stopwords]
        
        # Get keywords from result
        result_words = [w.lower() for w in result.split() if len(w) > 3 and w.lower() not in stopwords]
        
        # Combine all keywords
        all_keywords = set(question_words) | set(result_words)
        
        # Score messages by keyword relevance
        scored_messages = []
        for msg in self.messages:
            body_lower = msg.body.lower()
            sender_lower = msg.sender.lower()
            receiver_lower = msg.receiver.lower()
            
            score = 0
            # Check for keyword matches in body (highest weight)
            for keyword in all_keywords:
                if keyword in body_lower:
                    score += 3
                if keyword in sender_lower or keyword in receiver_lower:
                    score += 1
            
            # Check for message ID mentions in result
            if msg.message_id in result:
                score += 10  # Strong signal if message ID is mentioned
            
            if score > 0:
                scored_messages.append((score, msg))
        
        # Sort by score (highest first) and take top 5
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        
        evidence_snippets = []
        for score, msg in scored_messages[:5]:
            evidence_snippets.append(msg.original_xml)
            logger.debug(f"Selected evidence: {msg.message_id} (score: {score})")
        
        return evidence_snippets

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
