"""Data models for Story Investigator."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Represents a message in the story.
    
    Attributes:
        message_id: The message ID from XML (e.g., "m1", "m2").
        sender: The sender's identifier (from ref attribute).
        receiver: The receiver's identifier (from ref attribute).
        body: The message body text.
        original_xml: The original XML snippet for evidence tracking.
    """
    
    model_config = ConfigDict(frozen=True)
    
    message_id: str = Field(..., description="Message ID from XML id attribute")
    sender: str = Field(..., description="Sender identifier from XML ref attribute")
    receiver: str = Field(..., description="Receiver identifier from XML ref attribute")
    body: str = Field(..., description="Message body text")
    original_xml: str = Field(..., description="Original XML snippet for evidence")


class MessageChunk(BaseModel):
    """Represents a chunk of messages for retrieval.
    
    Attributes:
        messages: List of Message objects in this chunk.
        combined_text: All message bodies joined together for embedding.
    """
    
    model_config = ConfigDict(frozen=True)
    
    messages: List[Message] = Field(..., description="List of Message objects in this chunk")
    combined_text: str = Field(..., description="Combined text from all message bodies")


class Answer(BaseModel):
    """Represents an answer with evidence.
    
    Attributes:
        answer_text: The answer text.
        evidence_ids: List of message IDs that were used as evidence (e.g., ["m9", "m31a"]).
        evidence_xml_snippets: List of original XML snippets used as evidence.
        reason: Optional explanation (used when answer is UNKNOWN).
    """
    
    model_config = ConfigDict(frozen=True)
    
    answer_text: str = Field(..., description="The answer text")
    evidence_ids: List[str] = Field(default_factory=list, description="Message IDs used as evidence")
    evidence_xml_snippets: List[str] = Field(..., description="Original XML snippets used as evidence")
    reason: str = Field(default="", description="Explanation for UNKNOWN answers")
