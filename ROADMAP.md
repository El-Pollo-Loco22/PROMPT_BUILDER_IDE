# Project Roadmap

## Phase 1: Environment Setup
Establish the project foundation. Python project with virtual environment, core dependencies (LangGraph, Ollama, Streamlit, Pydantic), Docker infrastructure for portability, and a basic smoke test to verify everything works.

## Phase 2: The Agent Core ("The Brain")
Build the ReAct (Reason + Act) loop using LangGraph:
- **Architect Agent** — Takes a user's rough idea and maps it into a structured prompt framework (CO-STAR)
- **Linter Agent** — Critiques generated prompts for ambiguity, missing constraints, and structural issues
- **Orchestration** — LangGraph flow that chains Architect → Linter → refinement loop

## Phase 3: The IDE UI
Build the Streamlit interface with:
- Three-column layout: Framework settings | Prompt editor | Agent feedback
- Real-time agent suggestions and Linter feedback display
- Side-by-side comparison view (Prompt A vs Prompt B)
- Connect Streamlit editor to LangGraph backend

## Phase 4: Packaging & Distribution
Make the project fully portable:
- Optimized lightweight production Dockerfile
- Auto-setup script (install Ollama, pull default model)
- Persistent prompt history via Docker volume mapping
- One-command deployment: `docker compose up`
