"""Data models for Story Investigator."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """Represents a message in the story.
    
    Attributes:
        sender: The sender's identifier (from ref attribute).
        receiver: The receiver's identifier (from ref attribute).
        body: The message body text.
        original_xml: The original XML snippet for evidence tracking.
    """
    
    model_config = ConfigDict(frozen=True)
    
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
        evidence_xml_snippets: List of original XML snippets used as evidence.
    """
    
    model_config = ConfigDict(frozen=True)
    
    answer_text: str = Field(..., description="The answer text")
    evidence_xml_snippets: List[str] = Field(..., description="Original XML snippets used as evidence")
