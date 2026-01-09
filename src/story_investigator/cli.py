"""Command-line interface for Story Investigator."""

import asyncio
import logging
import os
import sys

from story_investigator.config import load_config
from story_investigator.engines.light_rag_engine import LightRAGInvestigator
from story_investigator.engines.naive_rag import NaiveRAGInvestigator
from story_investigator.errors import PromptTooLongError
from story_investigator.llm_client import LLMClient
from story_investigator.prompt_manager import PromptManager
from story_investigator.retrieval.chunking import MessageChunker
from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,  # Show INFO logs including prompt character counts
    format='%(levelname)s: %(message)s'
)


async def main():
    """Main entry point for the CLI."""
    config = load_config()
    
    # Ensure OPENAI_API_KEY is set in environment for LightRAG
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key

    prompt_manager = PromptManager(max_length=config.max_prompt_length)

    engine_choice = config.rag_engine.lower()
    if engine_choice not in {"naive", "lightrag"}:
        engine_choice = input("Choose engine [naive/lightrag] (default naive): ").strip().lower() or "naive"

    if engine_choice == "lightrag":
        investigator = LightRAGInvestigator(
            story_path=str(config.story_path),
            prompt_manager=prompt_manager,
            llm_model=config.llm_model,
            llm_temperature=config.llm_temperature,
        )
        # Initialize LightRAG asynchronously
        await investigator.initialize()
    else:
        embedding_engine = EmbeddingEngine(model_name=config.embedding_model)
        # OpenAI text-embedding-3-small has 1536 dimensions (not 384 like sentence-transformers)
        vector_store = VectorStore(dimension=1536)
        chunker = MessageChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        llm_client = LLMClient(
            api_key=config.openai_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            prompt_manager=prompt_manager,
        )

        investigator = NaiveRAGInvestigator(
            story_path=str(config.story_path),
            embedding_engine=embedding_engine,
            vector_store=vector_store,
            chunker=chunker,
            prompt_manager=prompt_manager,
            llm_client=llm_client,
            top_k=5,  # Hardcoded to 5 to ensure prompt stays under 3000 chars (assignment constraint)
        )

    try:
        if engine_choice == "lightrag":
            await investigator.load_story(str(config.story_path))
        else:
            investigator.load_story(str(config.story_path))
    except FileNotFoundError:
        print(f"Story file not found at {config.story_path}")
        sys.exit(1)
    except Exception as exc:
        print(f"Failed to load story: {exc}")
        sys.exit(1)

    print("AI Investigator 1.0. Ask me any question about the story.")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        try:
            if engine_choice == "lightrag":
                answer = await investigator.ask(question)
            else:
                answer = investigator.ask(question)
            
            # Format output based on answer type
            if answer.answer_text.upper() == "UNKNOWN":
                # Unknown answer - show reason and closest evidence
                print(f"I don't know. Reason: {answer.reason if answer.reason else 'not in story'}")
                if answer.evidence_xml_snippets:
                    print("\nHere are the closest lines found:")
                    for snippet in answer.evidence_xml_snippets[:5]:
                        print(snippet)
            else:
                # Definitive answer - show answer and evidence
                print(f"{answer.answer_text}. Here are some of the lines that show it:")
                
                if answer.evidence_xml_snippets:
                    total_snippets = len(answer.evidence_xml_snippets)
                    max_display = 5
                    
                    for snippet in answer.evidence_xml_snippets[:max_display]:
                        print(snippet)
                    
                    if total_snippets > max_display:
                        remaining = total_snippets - max_display
                        print(f"\n(... and {remaining} more snippets found)")
                else:
                    print("(No evidence snippets available)")
                
        except PromptTooLongError:
            print("\nThe story context for this question is too long. Try being more specific.")
        except Exception as exc:
            print(f"\nAn error occurred: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

