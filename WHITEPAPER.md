# IED Prompter — Progress White Paper

## Executive Summary

IED Prompter is a framework-agnostic prompt engineering platform that automates intent extraction, prompt construction, simulation, and multi-dimensional quality evaluation. This white paper summarizes the work completed so far, outlines the architecture and implementation, and describes the validation strategy used to ensure robustness.

## Progress Overview

- Phase 1 (Environment & scaffolding): Completed — repository structure, Docker files, requirements, and initial smoke tests. See [PROGRESS.md](PROGRESS.md) for the full checklist.
- Phase 2 (Agent Core): Completed — core components implemented and well-tested: Pydantic schemas, intent extraction, architect agent, simulation node, linter agent, and LangGraph-based orchestration (reflection loop).
- Phase 3 (Streamlit UI): In progress — UI components planned (framework selector, dynamic editor, three-column layout, backend wiring).
- Phase 4 (Packaging & Distribution): Planned — production Dockerfile optimizations and deployment testing.

Key metrics: ~302 tests passing across unit and end-to-end suites (295 unit + 7 e2e), live-model integration tests executed against Ollama (llama3:8b).

## Architecture

The system is modular and node-oriented. Core pieces:

- Schema & Registry: Central `PromptSchema` with framework registry enables multiple frameworks (CO-STAR, RACE, APE, CRISPE) and runtime extension. See [src/schemas/prompt.py](src/schemas/prompt.py).
- Intent Extraction: Framework-aware `IntentExtractor` uses LLM calls to convert free-form user input into structured section values. See [src/agents/intent_extractor.py](src/agents/intent_extractor.py).
- Architect Agent: Builds and revises prompts from schema-driven templates and knowledge-base context. See [src/agents/architect.py](src/agents/architect.py).
- Simulation Node: Runs compiled prompts against Ollama, captures responses, execution time, token estimates, and checks format compliance.
- Linter Agent: Heuristic evaluation producing `QualityScore` (clarity, specificity, token efficiency, risk flags) and human-readable feedback.
- Orchestration: LangGraph wires the above nodes into a reflection loop: extract -> architect -> simulate -> linter -> conditional routing. See [src/graph/builder.py](src/graph/builder.py).
- Knowledge Base: Domain JSON files used to enrich prompt context. See [knowledge-base/general.json](knowledge-base/general.json).

## Implementation Summary

- Language: Python (project uses Pydantic for schemas and typed models).
- Prompt representation: `PromptSchema` supports frameworks and backward-compatible CO-STAR mappings; `compile_prompt()` renders templates per framework.
- Extensibility: `register_framework()` allows adding frameworks at runtime; nodes operate framework-agnostically.
- LLM integration: Local Ollama endpoint is used for model simulation and live integration tests (llama3:8b). The helper script to set up Ollama is in `scripts/setup_ollama.sh`.
- Testing: `pytest` + `pytest-asyncio` for async tests; tests are in the `tests/` directory (examples: [tests/test_architect.py](tests/test_architect.py), [tests/test_simulator.py](tests/test_simulator.py), [tests/test_e2e_live.py](tests/test_e2e_live.py)).

## Validation Strategy

The project uses a layered validation approach:

- Unit Tests: Fine-grained tests for each module (schema validation, compile templates, node logic). These tests run fast and mock external LLM calls where appropriate.
- Integration Tests (Mocked LLM): End-to-end flows are exercised with mocked LLM responses to validate wiring, state transitions, and format checking without needing a live model.
- Live LLM E2E Tests: A small set of end-to-end tests run against a local Ollama model (llama3:8b) to validate real-world model behavior, format parsing, and resilience to imperfect model outputs.
- Format & Content Validation: Simulation captures model output and runs format compliance checks (JSON, markdown, code, list, plain text) and content detectors that assert required/must-not content per `PromptSchema` rules.
- Heuristic Linting: `LinterAgent` flags prompt-injection risk, redundancy, token bloat, and ambiguity; its feedback is used by the reflection loop to drive revisions.
- Test Coverage & CI: The project includes `pytest.ini` and a comprehensive test suite; continuous integration should run unit, integration (mocked), and optional live-model tests in controlled environments. See [pytest.ini](pytest.ini) and [requirements.txt](requirements.txt).

Implementation validation highlights:

- Parser hardening: case-insensitive keys, nested dict flattening, unknown key filtering, and truncated JSON recovery logic are tested through parser resilience suites.
- Iteration control: reflection loop has a configurable max-iteration cap and routing threshold to avoid infinite refinement cycles.

## Tech Stack (Very Brief)

- Core: Python, Pydantic, pytest/pytest-asyncio.
- LLM host: Ollama (local model runner, tested with `llama3:8b`).
- Orchestration: LangGraph-style node wiring (internal graph builder).
- UI (planned): Streamlit for rapid interactive UI.
- Deployment: Docker + docker-compose for containerized runs.

Key files and manifests: [README.md](README.md), [PROGRESS.md](PROGRESS.md), [ARCHITECTURE.md](ARCHITECTURE.md), [requirements.txt](requirements.txt), [Dockerfile](Dockerfile), [docker-compose.yml](docker-compose.yml).

## Next Steps

1. Complete Streamlit UI and wire to the LangGraph backend.
2. Harden production Dockerfile and run E2E deployment tests.
3. Add CI workflows to run unit and integration tests automatically; gate live-model tests to optional runs.

---

This white paper captures progress to date and provides a concise technical summary. For full granular details and the living progress checklist, see [PROGRESS.md](PROGRESS.md).
