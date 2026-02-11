# Agentic Prompt IDE

A zero-cost, local-first IDE for crafting, critiquing, and refining AI prompts — powered by LangGraph agents and Ollama.

A multi-agent system (Intent Extractor, Architect, Simulator, Linter) works in a reflection loop to transform rough ideas into structured, high-quality prompts. Supports multiple prompt frameworks (CO-STAR, RACE, APE, CRISPE) with CO-STAR as default. Everything runs locally with no API costs.

## Supported Frameworks

| Framework | Sections | Best For |
|-----------|----------|----------|
| **CO-STAR** (default) | Context, Objective, Style, Tone, Audience, Response | General-purpose prompting |
| **RACE** | Role, Action, Context, Expectation | Task-oriented prompts |
| **APE** | Action, Purpose, Expectation | Concise, goal-driven prompts |
| **CRISPE** | Context, Role, Instruction, Schema, Persona, Examples | Complex, structured outputs |

Custom frameworks can be registered at runtime.

## Tech Stack

- **Python + LangGraph** — Agent orchestration (reflection loop)
- **Ollama** — Local LLM inference (default: llama3:8b)
- **Streamlit** — IDE-style UI
- **Pydantic** — Typed prompt schemas with framework-aware validation
- **Docker** — Portable, reproducible environment

## Quickstart

### Local Development

```bash
# Clone and enter the project
cd IED_Prompter

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

### Docker

```bash
# Start everything (app + Ollama)
docker compose up

# Or just Ollama (if running app locally)
docker compose up ollama
```

### Ollama Setup (Local)

```bash
# Run the helper script
chmod +x scripts/setup_ollama.sh
./scripts/setup_ollama.sh
```

## Project Structure

```
src/
├── schemas/
│   ├── frameworks.py      # Framework registry (CO-STAR, RACE, APE, CRISPE)
│   └── prompt.py          # PromptSchema, QualityScore, PromptTestResult
├── agents/
│   └── intent_extractor.py # Parses user input into framework sections
├── graph/                  # LangGraph orchestration (Phase 2F)
├── ui/                     # Streamlit frontend (Phase 3)
└── config.py               # Pydantic Settings from .env

knowledge-base/             # Domain-specific best practices (JSON)
tests/                      # pytest suite (64 tests)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.
See [ROADMAP.md](ROADMAP.md) for project phases.
See [PROGRESS.md](PROGRESS.md) for current status.
