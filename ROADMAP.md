# Project Roadmap

## Phase 1: Environment Setup (COMPLETE)
Establish the project foundation. Python project with virtual environment, core dependencies (LangGraph, Ollama, Streamlit, Pydantic), Docker infrastructure for portability, and a basic smoke test to verify everything works.

## Phase 2: The Agent Core ("The Brain") (COMPLETE)
Build the reflection loop using LangGraph with multi-framework support:
- **Pydantic Schemas** — Framework-agnostic `PromptSchema` with registry for CO-STAR, RACE, APE, CRISPE (and custom frameworks). Quality scoring, test results, and iteration tracking models.
- **Intent Extractor** — Parses free-form user input into structured framework sections via Ollama
- **Architect Agent** — Takes extracted intent and builds a full prompt using the selected framework's structure, incorporating domain knowledge
- **Simulation Node** — Runs compiled prompts against local Ollama, captures response + metrics
- **Linter Agent** — Multi-dimensional quality evaluation (clarity, specificity, structure, constraints, token efficiency) with risk detection
- **Orchestration** — LangGraph flow: extract_intent -> architect -> simulate -> linter -> conditional refinement loop (score >= 7 passes, else re-drafts with critique, max 3 iterations)

## Phase 3: The IDE UI (COMPLETE — Feb 20, 2026)
Desktop interface built with Electron + HTML + FastAPI:
- ✅ FastAPI backend serving REST API (6 endpoints: health, frameworks, domains, compile, run, static files)
- ✅ Reused KAIJU STATION HTML design with full API wiring (`frontend/app.js`)
- ✅ Framework selector, model/format config, constraints, variables, few-shot examples
- ✅ A/B comparison view (Prompt A vs Prompt B with score deltas)
- ✅ System status indicator (Grey/Green/Red with 30s health polling)
- ✅ AbortController for in-flight request cancellation
- ✅ Electron wrapper spawning Python backend (`electron/main.js`)
- See [FRONTEND_PLAN.md](FRONTEND_PLAN.md) for full specification

## Phase 4: Packaging & Distribution
Make the project fully portable:
- Optimized lightweight production Dockerfile
- Auto-setup script (install Ollama, pull default model)
- Persistent prompt history via Docker volume mapping
- One-command deployment: `docker compose up`
