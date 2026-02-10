# Architecture

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│  ┌─────────────┬──────────────┬───────────────┐  │
│  │  Framework   │   Prompt     │    Agent      │  │
│  │  Settings    │   Editor     │    Feedback   │  │
│  └─────────────┴──────────────┴───────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            LangGraph Orchestrator                 │
│                                                   │
│   ┌──────────┐     ┌──────────┐                  │
│   │Architect │────▶│  Linter  │──── loop ───┐    │
│   │  Agent   │     │  Agent   │              │    │
│   └──────────┘     └──────────┘◀─────────────┘    │
│                                                   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│              Ollama (Local LLM)                   │
│              Model: llama3:8b                     │
│              Port: 11434                          │
└─────────────────────────────────────────────────┘
```

## Data Flow

1. User enters a rough prompt idea in the Streamlit editor
2. Input is validated against the `PromptDocument` Pydantic schema
3. LangGraph sends the input to the **Architect Agent**, which structures it using the CO-STAR framework
4. The structured prompt passes to the **Linter Agent**, which critiques it for ambiguity, missing constraints, etc.
5. If issues are found, the loop cycles back to Architect for refinement
6. Final prompt is returned to the UI with suggestions and a quality score

## Key Design Decisions

**LangGraph** — Chosen for its native support for cyclic agent graphs (ReAct pattern). Agents can loop, reflect, and refine rather than just running a linear chain.

**Ollama** — Zero-cost local inference. Provides an OpenAI-compatible API, making it trivial to swap models or switch to a cloud provider later.

**Pydantic** — Typed schemas enforce structure on prompts. Every prompt is a validated `PromptDocument`, not a loose string.

**Docker + host.docker.internal** — On macOS, Ollama runs natively (using Metal GPU acceleration) while the Python app can run in Docker. The `docker-compose.yml` also includes an Ollama container for fully containerized setups.

## Configuration

All configuration flows through `src/config.py` (Pydantic Settings) loaded from a single `.env` file. No duplication between Docker environment and Python config.
