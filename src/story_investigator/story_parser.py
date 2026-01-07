"""Story parser for XML story files."""

from typing import List

from story_investigator.models import Message


class StoryParser:
    """Parses XML story files into Message objects."""

    def parse_string(self, xml_content: str) -> List[Message]:
        """Parse XML string content into a list of Message objects.
        
        Args:
            xml_content: XML string containing story messages.
            
        Returns:
            List of Message objects parsed from the XML.
            
        Raises:
            NotImplementedError: This method must be implemented as part of TDD cycle.
        """
        raise NotImplementedError("This method must be implemented as part of TDD cycle")
