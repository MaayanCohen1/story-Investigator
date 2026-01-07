"""Configuration management for Story Investigator."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class AppConfig:
    openai_api_key: str
    story_path: Path
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    max_prompt_length: int = 3000
    chunk_size: int = 5
    chunk_overlap: int = 1
    top_k: int = 3


def load_config(env_path: str | None = None) -> AppConfig:
    """Load configuration from .env and environment variables."""
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    story_path = Path(os.getenv("STORY_PATH", "story/story.xml"))

    return AppConfig(
        openai_api_key=openai_api_key or "",
        story_path=story_path,
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "3000")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "5")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "1")),
        top_k=int(os.getenv("TOP_K", "3")),
    )
