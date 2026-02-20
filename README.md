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
- **FastAPI** — REST API serving frontend and pipeline endpoints
- **Electron + HTML/CSS/JS** — Desktop IDE-style UI (KAIJU STATION)
- **Pydantic** — Typed prompt schemas with framework-aware validation
- **Docker** — Portable, reproducible environment

## Quickstart

### Local Development

```bash
# Clone and enter the project
cd IED_Prompter

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests (304 passing)
./scripts/test.sh -q
```

### Run the IDE

```bash
# Start Ollama (required for LLM features)
ollama serve

# Start the FastAPI server
.venv/bin/python3 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Open http://127.0.0.1:8000 in your browser
```

Or use the Electron desktop wrapper:

```bash
cd electron && npm install && npm start
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
│   ├── intent_extractor.py # Parses user input into framework sections
│   └── architect.py       # Builds/revises prompts from extracted intent
├── graph/
│   └── builder.py         # LangGraph reflection loop orchestration
├── api/
│   ├── main.py            # FastAPI app (6 endpoints + static files)
│   └── schemas.py         # Pydantic request/response models
└── config.py               # Pydantic Settings from .env

frontend/
├── index.html             # KAIJU STATION prompt builder UI
└── app.js                 # API wiring, state management, output renderers

electron/
├── main.js                # Electron wrapper (spawns uvicorn, opens window)
└── package.json           # Electron dependency

knowledge-base/             # Domain-specific best practices (JSON)
tests/                      # pytest suite (304 tests)
```

See [FRONTEND_PLAN.md](FRONTEND_PLAN.md) for the frontend integration plan.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.
See [ROADMAP.md](ROADMAP.md) for project phases.
See [PROGRESS.md](PROGRESS.md) for current status.
