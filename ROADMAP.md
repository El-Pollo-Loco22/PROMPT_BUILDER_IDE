# Project Roadmap

## Phase 1: Environment Setup (COMPLETE)
Establish the project foundation. Python project with virtual environment, core dependencies (LangGraph, Ollama, Streamlit, Pydantic), Docker infrastructure for portability, and a basic smoke test to verify everything works.

## Phase 2: The Agent Core ("The Brain") (IN PROGRESS)
Build the reflection loop using LangGraph with multi-framework support:
- **Pydantic Schemas** — Framework-agnostic `PromptSchema` with registry for CO-STAR, RACE, APE, CRISPE (and custom frameworks). Quality scoring, test results, and iteration tracking models.
- **Intent Extractor** — Parses free-form user input into structured framework sections via Ollama
- **Architect Agent** — Takes extracted intent and builds a full prompt using the selected framework's structure, incorporating domain knowledge
- **Simulation Node** — Runs compiled prompts against local Ollama, captures response + metrics
- **Linter Agent** — Multi-dimensional quality evaluation (clarity, specificity, structure, constraints, token efficiency) with risk detection
- **Orchestration** — LangGraph flow: extract_intent -> architect -> simulate -> linter -> conditional refinement loop (score >= 7 passes, else re-drafts with critique, max 3 iterations)

## Phase 3: The IDE UI
Build the Streamlit interface with:
- Framework selector dropdown (CO-STAR, RACE, APE, CRISPE)
- Three-column layout: Framework sections | Prompt editor | Agent feedback
- Real-time agent suggestions and Linter feedback display
- Side-by-side comparison view (Prompt A vs Prompt B)
- Connect Streamlit editor to LangGraph backend
- Prompt history / versioning

## Phase 4: Packaging & Distribution
Make the project fully portable:
- Optimized lightweight production Dockerfile
- Auto-setup script (install Ollama, pull default model)
- Persistent prompt history via Docker volume mapping
- One-command deployment: `docker compose up`
