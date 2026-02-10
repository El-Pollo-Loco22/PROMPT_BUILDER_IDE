# Agentic Prompt IDE

A zero-cost, local-first IDE for crafting, critiquing, and refining AI prompts — powered by LangGraph agents and Ollama.

Two AI agents (Architect and Linter) work in a ReAct loop to transform rough ideas into structured, high-quality prompts. Everything runs locally with no API costs.

## Tech Stack

- **Python + LangGraph** — Agent orchestration (ReAct loop)
- **Ollama** — Local LLM inference (default: llama3:8b)
- **Streamlit** — IDE-style UI
- **Pydantic** — Typed prompt schemas
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

# Run smoke tests
pytest tests/test_smoke.py -v
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

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design.
See [ROADMAP.md](ROADMAP.md) for project phases.
See [PROGRESS.md](PROGRESS.md) for current status.
