"""Tests for LightRAG engine implementation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from story_investigator.engines.light_rag_engine import LightRAGInvestigator
from story_investigator.errors import PromptTooLongError
from story_investigator.models import Answer
from story_investigator.prompt_manager import PromptManager

# Configure pytest to use anyio for async tests
pytestmark = pytest.mark.anyio


class TestLightRAGInvestigator:
    """Test suite for LightRAGInvestigator class."""

    @pytest.fixture
    def test_story_xml(self, tmp_path):
        """Create a test story XML file."""
        story_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1">
        <sender ref="alice"/>
        <receiver ref="bob"/>
        <body>I have the secret document.</body>
    </message>
    <message id="m2">
        <sender ref="bob"/>
        <receiver ref="alice"/>
        <body>Meet me at the park at 3pm.</body>
    </message>
</story>
"""
        story_file = tmp_path / "test_story.xml"
        story_file.write_text(story_content)
        return str(story_file)

    @pytest.fixture
    def prompt_manager(self):
        """Create a PromptManager instance."""
        return PromptManager(max_length=3000)
    
    @pytest.fixture
    def mock_embedding_engine(self):
        """Create a mock EmbeddingEngine."""
        with patch('story_investigator.engines.light_rag_engine.EmbeddingEngine') as mock_class:
            mock_engine = MagicMock()
            # Mock embed_text to return a dummy embedding
            mock_engine.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            # Mock embed_texts to return dummy embeddings for batch
            mock_engine.embed_texts.return_value = [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
            ]
            mock_class.return_value = mock_engine
            yield mock_engine

    async def test_initialize_creates_working_directory(self, test_story_xml, prompt_manager, tmp_path):
        """Test that initialize creates the working directory."""
        working_dir = tmp_path / "lightrag_test"
        
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
            working_dir=str(working_dir),
        )
        
        with patch('story_investigator.engines.light_rag_engine.LightRAG') as mock_lightrag, \
             patch('story_investigator.engines.light_rag_engine.LLMClient') as mock_llm_client:
            
            # Mock the LightRAG instance with async methods
            mock_rag_instance = AsyncMock()
            mock_rag_instance.initialize_storages = AsyncMock()
            mock_lightrag.return_value = mock_rag_instance
            
            await investigator.initialize()
            
            assert working_dir.exists()
            assert investigator._initialized
            mock_rag_instance.initialize_storages.assert_called_once()

    async def test_ask_without_initialize_raises_error(self, test_story_xml, prompt_manager):
        """Test that ask() raises error if initialize() wasn't called."""
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
        )
        
        with pytest.raises(RuntimeError, match="Must call initialize"):
            await investigator.ask("test question")

    async def test_load_story_without_initialize_raises_error(self, test_story_xml, prompt_manager):
        """Test that load_story() raises error if initialize() wasn't called."""
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
        )
        
        with pytest.raises(RuntimeError, match="Must call initialize"):
            await investigator.load_story(test_story_xml)

    async def test_ask_returns_answer_with_evidence(self, test_story_xml, prompt_manager, tmp_path, mock_embedding_engine):
        """Test that ask() returns an Answer with evidence snippets."""
        working_dir = tmp_path / "lightrag_test"
        
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
            working_dir=str(working_dir),
        )
        
        # Mock LightRAG and LLM client
        with patch('story_investigator.engines.light_rag_engine.LightRAG') as mock_lightrag_class, \
             patch('story_investigator.engines.light_rag_engine.LLMClient') as mock_llm_client_class:
            
            # Setup mock LightRAG instance
            mock_rag = AsyncMock()
            mock_rag.ainsert = AsyncMock()
            mock_rag.aquery = AsyncMock(return_value="alice has the secret document. bob wants to meet.")
            mock_lightrag_class.return_value = mock_rag
            
            # Setup mock LLM client with structured response
            mock_llm_client = MagicMock()
            mock_llm_client.generate_answer = MagicMock(return_value="""ANSWER: Alice
EVIDENCE_IDS: m1
REASON: Message m1 states that Alice has the secret document.""")
            mock_llm_client_class.return_value = mock_llm_client
            
            await investigator.initialize()
            await investigator.load_story(test_story_xml)
            
            answer = await investigator.ask("Who has the secret document?")
            
            assert isinstance(answer, Answer)
            assert "alice" in answer.answer_text.lower()
            assert len(answer.evidence_xml_snippets) > 0
            assert len(answer.evidence_ids) > 0

    async def test_ask_with_no_context_returns_not_available(self, test_story_xml, prompt_manager, tmp_path):
        """Test that ask() handles empty context gracefully."""
        working_dir = tmp_path / "lightrag_test"
        
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
            working_dir=str(working_dir),
        )
        
        # Mock LightRAG to return empty context
        with patch('story_investigator.engines.light_rag_engine.LightRAG') as mock_lightrag_class, \
             patch('story_investigator.engines.light_rag_engine.LLMClient') as mock_llm_client_class:
            
            mock_rag = AsyncMock()
            mock_rag.ainsert = AsyncMock()
            mock_rag.aquery = AsyncMock(return_value="")  # Empty context
            mock_lightrag_class.return_value = mock_rag
            
            mock_llm_client = MagicMock()
            mock_llm_client_class.return_value = mock_llm_client
            
            await investigator.initialize()
            await investigator.load_story(test_story_xml)
            
            answer = await investigator.ask("What is the meaning of life?")
            
            assert "not in the story" in answer.answer_text.lower() or \
                   "not conclusive" in answer.answer_text.lower() or \
                   "don't have enough information" in answer.answer_text.lower()

    async def test_prompt_length_validation(self, test_story_xml, prompt_manager, tmp_path):
        """Test that prompts are validated against the 3000 character limit."""
        working_dir = tmp_path / "lightrag_test"
        
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
            working_dir=str(working_dir),
        )
        
        # Mock LightRAG to return very large context
        with patch('story_investigator.engines.light_rag_engine.LightRAG') as mock_lightrag_class, \
             patch('story_investigator.engines.light_rag_engine.LLMClient') as mock_llm_client_class:
            
            mock_rag = AsyncMock()
            mock_rag.ainsert = AsyncMock()
            # Return context that mentions messages
            large_context = "alice has secret document. " * 100 + " bob meet park"
            mock_rag.aquery = AsyncMock(return_value=large_context)
            mock_lightrag_class.return_value = mock_rag
            
            mock_llm_client = MagicMock()
            mock_llm_client.generate_answer = MagicMock(return_value="""ANSWER: Alice
EVIDENCE_IDS: m1
REASON: Message m1 shows Alice has the document.""")
            mock_llm_client_class.return_value = mock_llm_client
            
            await investigator.initialize()
            await investigator.load_story(test_story_xml)
            
            answer = await investigator.ask("Test question?")
            
            # Should reduce messages and still generate an answer
            # The mock LLM client should have been called with a prompt <= 3000 chars
            assert mock_llm_client.generate_answer.called
            called_prompt = mock_llm_client.generate_answer.call_args[0][0]
            assert len(called_prompt) <= 3000

    def test_build_prompt_with_limit_truncates_long_context(self, test_story_xml, prompt_manager):
        """Test that _build_prompt_with_limit truncates long context."""
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
        )
        
        question = "Short question?"
        long_context = "X" * 5000
        
        prompt, truncated_context = investigator._build_prompt_with_limit(
            question, 
            long_context, 
            max_chars=3000
        )
        
        assert len(prompt) <= 3000
        assert len(truncated_context) < len(long_context)
        assert "truncated" in truncated_context.lower()

    def test_build_prompt_with_limit_raises_error_if_question_too_long(self, test_story_xml, prompt_manager):
        """Test that _build_prompt_with_limit raises error if question alone is too long."""
        investigator = LightRAGInvestigator(
            story_path=test_story_xml,
            prompt_manager=prompt_manager,
        )
        
        # Create a question that by itself (with instructions) exceeds the limit
        very_long_question = "Q" * 3000
        
        with pytest.raises(PromptTooLongError):
            investigator._build_prompt_with_limit(
                very_long_question, 
                "context", 
                max_chars=3000
            )
    
    async def test_usb_question_regression(self, tmp_path, prompt_manager):
        """Regression test for USB question bug.
        
        Bug: When asking 'Who requested to bring the USB?', LightRAG returned
        too much context (~100K chars), which got truncated to ~2.7K chars in a
        naive way, dropping the most relevant evidence (Marcus's USB request).
        
        Fix: We now use relevance scoring to prioritize the most important messages
        BEFORE building the prompt, ensuring critical evidence is never lost.
        """
        # Create a story with USB message buried among other messages
        story_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1">
        <sender ref="alice"/>
        <receiver ref="bob"/>
        <body>Let's meet at the park.</body>
    </message>
    <message id="m2">
        <sender ref="bob"/>
        <receiver ref="alice"/>
        <body>Sure, what time?</body>
    </message>
    <message id="m3">
        <sender ref="marcus"/>
        <receiver ref="alex"/>
        <body>DM: Bring that USB you borrowed. I need it back tonight. No excuses.</body>
    </message>
    <message id="m4">
        <sender ref="alice"/>
        <receiver ref="bob"/>
        <body>How about 3pm?</body>
    </message>
</story>
"""
        story_file = tmp_path / "usb_story.xml"
        story_file.write_text(story_content)
        working_dir = tmp_path / "lightrag_usb_test"
        
        investigator = LightRAGInvestigator(
            story_path=str(story_file),
            prompt_manager=prompt_manager,
            working_dir=str(working_dir),
        )
        
        # Mock LightRAG to simulate the bug scenario
        with patch('story_investigator.engines.light_rag_engine.LightRAG') as mock_lightrag_class, \
             patch('story_investigator.engines.light_rag_engine.LLMClient') as mock_llm_client_class:
            
            # Setup mock LightRAG that returns context mentioning "USB" and "Marcus"
            mock_rag = AsyncMock()
            mock_rag.ainsert = AsyncMock()
            # Simulate large context that mentions USB and Marcus
            mock_rag.aquery = AsyncMock(return_value=(
                "Alice and Bob discussed meeting at the park. "
                "Marcus requested that Alex bring the USB back. "
                "The USB was borrowed and Marcus needs it tonight. "
                "There were discussions about timing."
            ))
            mock_lightrag_class.return_value = mock_rag
            mock_rag.initialize_storages = AsyncMock()
            
            # Setup mock LLM to return structured answer
            mock_llm_client = MagicMock()
            mock_llm_client.generate_answer = MagicMock(return_value="""ANSWER: Marcus
EVIDENCE_IDS: m3
REASON: Message m3 explicitly states 'Bring that USB you borrowed'""")
            mock_llm_client_class.return_value = mock_llm_client
            
            await investigator.initialize()
            await investigator.load_story(str(story_file))
            
            # Ask the USB question
            answer = await investigator.ask("Who requested to bring the USB?")
            
            # Verify the answer contains Marcus
            assert "marcus" in answer.answer_text.lower(), f"Expected Marcus in answer, got: {answer.answer_text}"
            
            # Verify that evidence IDs include m3 (the USB request)
            assert "m3" in answer.evidence_ids, f"Expected m3 in evidence IDs, got: {answer.evidence_ids}"
            
            # Verify that the USB message is in the evidence snippets
            usb_message_found = False
            for snippet in answer.evidence_xml_snippets:
                if "USB" in snippet and "marcus" in snippet.lower():
                    usb_message_found = True
                    break
            
            assert usb_message_found, (
                "USB request message should be in evidence snippets. "
                f"Got {len(answer.evidence_xml_snippets)} snippets"
            )

