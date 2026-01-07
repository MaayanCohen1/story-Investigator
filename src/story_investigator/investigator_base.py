"""Abstract base class for all RAG investigators."""

from abc import ABC, abstractmethod

from story_investigator.models import Answer


class BaseInvestigator(ABC):
    """Abstract interface for all RAG engine implementations."""

    @abstractmethod
    def ask(self, question: str) -> Answer:
        """Answer a question about the story with evidence.
        
        Args:
            question: The user's question about the story.
            
        Returns:
            Answer object containing the answer text and evidence XML snippets.
        """
        pass

    @abstractmethod
    def load_story(self, story_path: str) -> None:
        """Load and index the story for retrieval.
        
        Args:
            story_path: Path to the story XML file.
        """
        pass

