# Architecture

## System Overview

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                     │
│  ┌─────────────┬──────────────┬───────────────┐  │
│  │  Framework   │   Prompt     │    Agent      │  │
│  │  Selector    │   Editor     │    Feedback   │  │
│  └─────────────┴──────────────┴───────────────┘  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│            LangGraph Orchestrator                 │
│                                                   │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│   │ Intent   │─▶│Architect │─▶│Simulation│      │
│   │Extractor │  │  Agent   │  │  Node    │      │
│   └──────────┘  └──────────┘  └──────────┘      │
│                       ▲            │              │
│                       │            ▼              │
│                       │      ┌──────────┐        │
│                       └──────│  Linter  │        │
│                    critique  │  Agent   │        │
│                    (if <7)   └──────────┘        │
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

1. User enters a rough prompt idea in the Streamlit editor and selects a framework (CO-STAR, RACE, APE, CRISPE)
2. **Intent Extractor** parses free-form input into structured framework sections via Ollama
3. **Architect Agent** builds a full `PromptSchema` using the selected framework's section definitions
4. **Simulation Node** runs the compiled prompt against local Ollama, capturing response + metrics
5. **Linter Agent** evaluates quality (clarity, specificity, structure, constraints, token efficiency) and produces a `QualityScore`
6. If score < 7, the loop cycles back to Architect with critique for refinement (max 3 iterations)
7. Final prompt is returned to the UI with the compiled output, test result, score, and suggestions

## Multi-Framework Support

The system supports multiple prompt engineering frameworks via a **registry pattern**:

| Framework | Sections |
|-----------|----------|
| **CO-STAR** (default) | Context, Objective, Style, Tone, Audience, Response |
| **RACE** | Role, Action, Context, Expectation |
| **APE** | Action, Purpose, Expectation |
| **CRISPE** | Context, Role, Instruction, Schema, Persona, Examples |

Frameworks are defined as pure data in `src/schemas/frameworks.py`. Each `FrameworkDef` specifies:
- Ordered sections with required/optional flags and defaults
- A compile template that renders the final prompt string
- Min-length constraints per section

Custom frameworks can be registered at runtime via `register_framework()`.

A single `PromptSchema` model handles all frameworks — the `framework` field selects which definition to validate against, and `sections: Dict[str, str]` holds the content.

## Schema Architecture

```
src/schemas/
├── frameworks.py    # FrameworkName, SectionDef, FrameworkDef, registry
├── prompt.py        # PromptSchema, QualityScore, PromptTestResult, PromptIteration
└── __init__.py      # Re-exports all public symbols

src/agents/
├── intent_extractor.py   # IntentExtractor, ExtractedIntent, build_extraction_prompt
└── __init__.py           # Re-exports
```

## Key Design Decisions

**Framework Registry** — Frameworks are defined as data (not subclasses), making it trivial to add new ones without writing Python classes. All agents work with a single `PromptSchema` type regardless of framework.

**LangGraph** — Chosen for its native support for cyclic agent graphs. The reflection loop (architect → simulate → linter → conditional re-draft) is a natural fit.

**Ollama** — Zero-cost local inference. Provides an OpenAI-compatible API, making it trivial to swap models or switch to a cloud provider later.

**Pydantic** — Typed schemas enforce structure on prompts. `model_validator` handles backward compatibility, framework-specific validation, and section defaults.

**Docker + host.docker.internal** — On macOS, Ollama runs natively (using Metal GPU acceleration) while the Python app can run in Docker. The `docker-compose.yml` also includes an Ollama container for fully containerized setups.

## Configuration

All configuration flows through `src/config.py` (Pydantic Settings) loaded from a single `.env` file. No duplication between Docker environment and Python config.

Domain-specific knowledge bases live in `knowledge-base/*.json` and are loaded by the Architect Agent for best-practice injection.
