# Story Investigator

An AI-powered investigation tool that answers questions about story data using multiple RAG (Retrieval-Augmented Generation) systems.

## Features

- **Console-based Q&A**: Interactive CLI for investigating story messages
- **Multiple RAG Systems**: Compare three different retrieval approaches
  - `naive_rag.py`: Custom vector-based chunking and retrieval
  - `light_rag_engine.py`: LightRAG integration with graph-based knowledge extraction
  - `nano_graphrag_engine.py`: nano-graphrag integration
- **Evidence-Based Answers**: Every answer includes XML snippets showing the source
- **Prompt Safety**: Hard 3000-character limit on LLM prompts with exception handling
- **Clean Architecture**: Dependency injection, abstract interfaces, TDD

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
STORY_PATH=story/story.xml
RAG_ENGINE=naive  # Options: naive, lightrag, nano_graphrag
```

## Usage

```bash
python -m story_investigator.cli
```

Example interaction:
```
AI Investigator 1.0. Ask me any question about the story.
> Who requested to bring the USB?
Marcus. Here are some of the lines that show it:
<sender ref="marcus"/>
<receiver ref="alex"/>
<body>DM: Bring that USB you borrowed. I need it back tonight. No excuses.</body>
```

## Development

### Run Tests

```bash
pytest
```

### Run with specific RAG engine

Set `RAG_ENGINE` in your `.env` file:
```
RAG_ENGINE=naive     # Use custom vector-based RAG
RAG_ENGINE=lightrag  # Use LightRAG with knowledge graph
```

Or the CLI will prompt you to choose.

## Architecture

- **TDD**: `PromptManager` developed test-first
- **Dependency Injection**: All components receive dependencies via constructor
- **Abstract Base Classes**: Common interface for all RAG engines
- **Evidence Tracking**: Original XML lines preserved and returned with answers

