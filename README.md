# Country Info AI Agent

An AI agent that answers natural language questions about countries using [LangGraph](https://langchain-ai.github.io/langgraph/), [OpenRouter](https://openrouter.ai/), and the public [REST Countries API](https://restcountries.com/).

## Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  extract_intent     │  LLM parses country name + requested fields
└────────┬────────────┘
         │ (conditional: skip fetch on error)
         ▼
┌─────────────────────┐
│   fetch_data        │  Calls REST Countries API (no auth, no DB)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  synthesize_answer  │  LLM composes grounded answer from raw data
└─────────────────────┘
```

## Setup

```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

uv sync
uv run chainlit run app.py
```

## Example Questions

- What is the population of Germany?
- What currency does Japan use?
- What is the capital and population of Brazil?
- What languages are spoken in Switzerland?

## Tech Stack

- **LangGraph** — agent orchestration with explicit intent → fetch → synthesis nodes
- **LangChain OpenRouter** — LLM access via `google/gemini-2.0-flash-001`
- **REST Countries API** — public, no-auth data source
- **Chainlit** — chat UI
- **uv** — package management
