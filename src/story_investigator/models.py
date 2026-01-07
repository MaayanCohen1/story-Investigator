"""Data models for Story Investigator."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Represents a message in the story.
    
    Attributes:
        sender: The sender's identifier (from ref attribute).
        receiver: The receiver's identifier (from ref attribute).
        body: The message body text.
        original_xml: The original XML snippet for evidence tracking.
    """
    
    sender: str = Field(..., description="Sender identifier from XML ref attribute")
    receiver: str = Field(..., description="Receiver identifier from XML ref attribute")
    body: str = Field(..., description="Message body text")
    original_xml: str = Field(..., description="Original XML snippet for evidence")

    class Config:
        frozen = True  # Make Message immutable
