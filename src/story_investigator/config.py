"""Configuration management for Story Investigator."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class AppConfig:
    openai_api_key: str
    story_path: Path
    rag_engine: str = "naive"  # naive, lightrag, or nano
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    llm_model: str = "gpt-5-mini"  # OpenAI model for LightRAG entity extraction
    llm_temperature: float = 0.0
    max_prompt_length: int = 3000
    chunk_size: int = 5
    chunk_overlap: int = 1
    top_k: int = 7  # Retrieve more chunks, then dynamically fit to prompt limit


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
        rag_engine=os.getenv("RAG_ENGINE", "naive"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.getenv("LLM_MODEL", "gpt-5-mini"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        max_prompt_length=int(os.getenv("MAX_PROMPT_LENGTH", "3000")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "5")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "1")),
        top_k=int(os.getenv("TOP_K", "7")),
    )
