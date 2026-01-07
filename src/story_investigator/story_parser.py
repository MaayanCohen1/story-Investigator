"""Story parser for XML story files."""

import xml.etree.ElementTree as ET
from typing import List

from story_investigator.errors import StoryParseError
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
            StoryParseError: If the XML cannot be parsed or is invalid.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            raise StoryParseError(f"Failed to parse XML: {e}") from e
        
        messages = []
        
        for message_elem in root.iter():
            if message_elem.tag.endswith("message") or message_elem.tag == "message":
                try:
                    sender_elem = None
                    receiver_elem = None
                    body_elem = None
                    
                    for child in message_elem:
                        tag_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                        if tag_name == "sender":
                            sender_elem = child
                        elif tag_name == "receiver":
                            receiver_elem = child
                        elif tag_name == "body":
                            body_elem = child
                    
                    if sender_elem is None or receiver_elem is None or body_elem is None:
                        continue
                    
                    sender = sender_elem.get("ref", "")
                    receiver = receiver_elem.get("ref", "")
                    body = body_elem.text or ""
                    
                    if not sender or not receiver:
                        continue
                    
                    original_xml = ET.tostring(message_elem, encoding="unicode")
                    
                    message = Message(
                        sender=sender,
                        receiver=receiver,
                        body=body,
                        original_xml=original_xml
                    )
                    messages.append(message)
                    
                except (AttributeError, ValueError) as e:
                    raise StoryParseError(f"Failed to parse message element: {e}") from e
        
        return messages
