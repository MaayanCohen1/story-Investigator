"""LightRAG engine adapter."""

from pathlib import Path
from typing import List

from lightrag import LightRAG

from story_investigator.investigator_base import BaseInvestigator
from story_investigator.models import Answer, Message
from story_investigator.prompt_manager import PromptManager
from story_investigator.story_parser import StoryParser


class LightRAGInvestigator(BaseInvestigator):
    """LightRAG implementation adapter."""

    def __init__(
        self,
        story_path: str,
        prompt_manager: PromptManager,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
    ):
        self.story_path = Path(story_path)
        self.prompt_manager = prompt_manager
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.messages: List[Message] = []
        self.rag = LightRAG()

    def load_story(self, story_path: str) -> None:
        story_file = Path(story_path)
        if not story_file.exists():
            raise FileNotFoundError(f"Story file not found: {story_path}")

        with open(story_file, "r", encoding="utf-8") as f:
            story_text = f.read()

        parser = StoryParser()
        self.messages = parser.parse_string(story_text)

        # LightRAG indexing on full text
        self.rag.add_documents([story_text])

    def ask(self, question: str) -> Answer:
        # LightRAG handles its own retrieval and answering
        try:
            response = self.rag.query(question)
            answer_text = response if isinstance(response, str) else str(response)
        except Exception as exc:
            return Answer(
                answer_text=f"LightRAG query failed: {exc}",
                evidence_xml_snippets=[],
            )

        # Lightweight evidence: pick messages containing key terms from question
        evidence_xml_snippets = []
        question_tokens = {t.strip('?,.!').lower() for t in question.split() if len(t) >= 3}
        for msg in self.messages:
            body_lower = msg.body.lower()
            if any(tok in body_lower for tok in question_tokens):
                evidence_xml_snippets.append(msg.original_xml)
                if len(evidence_xml_snippets) >= 5:
                    break

        return Answer(
            answer_text=answer_text,
            evidence_xml_snippets=evidence_xml_snippets,
        )

