"""Command-line interface for Story Investigator."""

import sys

from story_investigator.config import load_config
from story_investigator.engines.naive_rag import NaiveRAGInvestigator
from story_investigator.llm_client import LLMClient
from story_investigator.prompt_manager import PromptManager
from story_investigator.retrieval.chunking import MessageChunker
from story_investigator.retrieval.embeddings import EmbeddingEngine
from story_investigator.retrieval.vector_store import VectorStore


def main():
    """Main entry point for the CLI."""
    config = load_config()

    prompt_manager = PromptManager(max_length=config.max_prompt_length)
    embedding_engine = EmbeddingEngine(model_name=config.embedding_model)
    vector_store = VectorStore(dimension=384)
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
        top_k=config.top_k,
    )

    try:
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

        answer = investigator.ask(question)
        print(answer.answer_text)
        if answer.evidence_xml_snippets:
            print("Here are some of the lines that show it:")
            for snippet in answer.evidence_xml_snippets:
                print(snippet)
        else:
            print("No evidence snippets available.")


if __name__ == "__main__":
    main()

