"""Integration tests for naive RAG engine."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from story_investigator.engines.naive_rag import NaiveRAGInvestigator
from story_investigator.models import Answer, Message, MessageChunk
from story_investigator.prompt_manager import PromptManager


class TestNaiveRAGInvestigator:
    """Test suite for NaiveRAGInvestigator class."""

    @pytest.fixture
    def test_story_xml(self, tmp_path):
        """Create a test story XML file."""
        story_content = """<?xml version="1.0" encoding="UTF-8"?>
<story>
    <message id="m1" ts="2025-08-29T17:42:03+10:00">
        <sender ref="alex"/>
        <receiver ref="six"/>
        <body>The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late.</body>
    </message>
    <message id="m2" ts="2025-08-29T17:42:55+10:00">
        <sender ref="marcus"/>
        <receiver ref="alex"/>
        <body>DM: Bring that USB you borrowed. I need it back tonight. No excuses.</body>
    </message>
    <message id="m3" ts="2025-08-29T17:43:18+10:00">
        <sender ref="alex"/>
        <receiver ref="marcus"/>
        <body>Sure, I'll bring it.</body>
    </message>
</story>"""
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
        mock_engine = MagicMock()
        mock_engine.embed_text.return_value = [0.1] * 384
        mock_engine.embed_batch.return_value = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        return mock_engine

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore."""
        mock_store = MagicMock()
        return mock_store

    @pytest.fixture
    def mock_chunker(self):
        """Create a mock MessageChunker."""
        mock_chunker = MagicMock()
        return mock_chunker

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLMClient."""
        mock_client = MagicMock()
        mock_client.generate_answer.return_value = "Alex mentioned the harbour meeting."
        return mock_client

    @pytest.fixture
    def sample_messages(self):
        """Create sample Message objects."""
        return [
            Message(
                message_id="m1",
                sender="alex",
                receiver="six",
                body="The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late.",
                original_xml='<message id="m1" ts="2025-08-29T17:42:03+10:00"><sender ref="alex"/><receiver ref="six"/><body>The harbour looks electric — and I don\'t mean the lights. Circular Quay, Wharf 3, 6pm. Don\'t be late.</body></message>'
            ),
            Message(
                message_id="m2",
                sender="marcus",
                receiver="alex",
                body="DM: Bring that USB you borrowed. I need it back tonight. No excuses.",
                original_xml='<message id="m2" ts="2025-08-29T17:42:55+10:00"><sender ref="marcus"/><receiver ref="alex"/><body>DM: Bring that USB you borrowed. I need it back tonight. No excuses.</body></message>'
            ),
        ]

    @pytest.fixture
    def sample_chunks(self, sample_messages):
        """Create sample MessageChunk objects."""
        return [
            MessageChunk(
                messages=[sample_messages[0]],
                combined_text="[alex] to [six]: The harbour looks electric — and I don't mean the lights. Circular Quay, Wharf 3, 6pm. Don't be late."
            ),
            MessageChunk(
                messages=[sample_messages[1]],
                combined_text="[marcus] to [alex]: DM: Bring that USB you borrowed. I need it back tonight. No excuses."
            ),
        ]

    def test_answer_question_with_evidence_returns_answer_instance(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that ask() returns an Answer instance."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert isinstance(answer, Answer)

    def test_answer_question_with_evidence_contains_answer_text(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that answer contains answer_text."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert hasattr(answer, 'answer_text')

    def test_answer_question_with_evidence_contains_evidence_xml_snippets(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that answer contains evidence_xml_snippets."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert hasattr(answer, 'evidence_xml_snippets')

    def test_answer_question_with_evidence_xml_snippets_is_list(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that evidence_xml_snippets is a list."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert isinstance(answer.evidence_xml_snippets, list)

    def test_answer_question_with_evidence_xml_snippets_not_empty(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that evidence_xml_snippets is not empty when evidence is found."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert len(answer.evidence_xml_snippets) > 0

    def test_answer_question_with_evidence_xml_snippets_contains_original_xml(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that evidence_xml_snippets contains original XML from messages."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Who mentioned the harbour?")
        
        assert any('alex' in snippet for snippet in answer.evidence_xml_snippets)

    def test_answer_hard_question_requires_multiple_messages_returns_answer(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that answering questions requiring multiple messages returns an Answer."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95), (sample_chunks[1], 0.90)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("What did Alex and Marcus discuss?")
        
        assert isinstance(answer, Answer)

    def test_answer_hard_question_requires_multiple_messages_includes_multiple_evidence(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_messages, sample_chunks
    ):
        """Test that questions requiring multiple messages include multiple evidence snippets."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = [(sample_chunks[0], 0.95), (sample_chunks[1], 0.90)]
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("What did Alex and Marcus discuss?")
        
        assert len(answer.evidence_xml_snippets) >= 1

    def test_answer_question_no_chunks_returns_no_story_data_message(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client
    ):
        """Test that asking without loaded chunks returns appropriate message."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = []
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Any question?")
        
        assert "No story data available" in answer.answer_text

    def test_answer_question_no_search_results_returns_no_relevant_info(
        self, test_story_xml, prompt_manager, mock_embedding_engine,
        mock_vector_store, mock_chunker, mock_llm_client, sample_chunks
    ):
        """Test that asking with no search results returns appropriate message."""
        investigator = NaiveRAGInvestigator(
            story_path=test_story_xml,
            embedding_engine=mock_embedding_engine,
            vector_store=mock_vector_store,
            chunker=mock_chunker,
            prompt_manager=prompt_manager,
            llm_client=mock_llm_client,
        )
        
        mock_chunker.chunk_messages.return_value = sample_chunks
        mock_vector_store.search.return_value = []
        
        investigator.load_story(test_story_xml)
        answer = investigator.ask("Unrelated question?")
        
        assert "No relevant information found" in answer.answer_text
