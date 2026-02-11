# Progress Tracker

## Phase 1: Environment Setup
- [x] Project structure created
- [x] .gitignore configured
- [x] Tracking docs (README, PROGRESS, ROADMAP, ARCHITECTURE)
- [x] requirements.txt with core dependencies
- [x] Python source skeleton (config, schemas, package init files)
- [x] Smoke tests written
- [x] Ollama setup helper script
- [x] Dockerfile + docker-compose.yml
- [x] Dev Container configuration
- [x] Venv created and smoke tests passing (3/3 pass)
- [x] Ollama model pulled and verified (llama3:8b)
- [x] Initial git commit

## Phase 2: Agent Core

### 2A — Pydantic Schemas (Foundation)
- [x] Framework registry (`FrameworkName`, `SectionDef`, `FrameworkDef`)
      with 4 built-in frameworks: CO-STAR, RACE, APE, CRISPE
- [x] `register_framework()` for custom frameworks at runtime
- [x] Framework-agnostic `PromptSchema` with `framework` + `sections` fields
      (backward-compat properties for CO-STAR: context, objective, style, tone, audience)
- [x] Framework-aware `compile_prompt()` using compile templates
- [x] `model_validator` for section validation, min-lengths, defaults, legacy migration
- [x] `ModelTarget` and `ResponseFormat` enums
- [x] `PromptTestResult` model (prompt_used, model_response, token_count,
      execution_time_ms, follows_format, includes/missing_required, unwanted)
- [x] `QualityScore` model (component scores, strengths/issues/suggestions,
      risk flags, reasoning, passes_threshold())
- [x] `PromptIteration` model (version tracking across reflection loop)
- [x] `knowledge-base/` with starter domain files (software_development, general)
- [x] `pytest-asyncio` added to requirements.txt
- [x] Re-exports in `__init__.py` for schemas and agents packages
- [x] Unit tests: schema validation, compile_prompt(), token_estimate(),
      multi-framework creation, registry, custom frameworks (64 tests passing)

### 2B — Intent Extraction Node
- [x] `IntentExtractor` — framework-aware, parses free-form user input via Ollama
- [x] `build_extraction_prompt()` — dynamically builds LLM system prompt from any framework's section definitions
- [x] `ExtractedIntent` — framework-agnostic model with `sections` dict
      (backward-compat properties for CO-STAR fields)
- [x] Accepts `framework` param (default CO-STAR), works with any registered framework
- [x] Unit tests with mocked LLM responses + multi-framework extraction tests

### 2C — Architect Agent (Draft Node)
- [x] `ArchitectAgent` — builds/revises prompts using selected framework's structure
      (calls `get_framework()` to discover required sections)
- [x] Domain-specific knowledge base integration (load `knowledge-base/*.json`)
- [x] Critique-driven revision mode (accept linter feedback + `QualityScore.issues`,
      revise `PromptSchema.sections` accordingly)
- [x] Unit tests with mocked LLM responses (36 tests: CO-STAR, RACE, APE, CRISPE,
      knowledge base, critique revision, compile round-trip) — 100 tests passing

### 2D — Simulation Node
- [x] `SimulationNode` — run `prompt.compile_prompt()` against local Ollama
      (framework-agnostic: compile_prompt() handles framework rendering)
- [x] Capture: response text, token count, execution time
- [x] Format compliance check (JSON, code, markdown, list, plain text)
- [x] Required/unwanted content detection against `PromptSchema.must_include`/`must_not_include`
- [x] Unit tests with mocked LLM (51 tests: format checkers, content detection,
      async/sync simulate, multi-framework) — 151 tests passing
- [x] Integration test against live Ollama (llama3:8b)

### 2E — Linter Agent (Analysis Node)
- [x] `LinterAgent` — multi-dimensional quality evaluation (heuristic-based, no LLM needed)
- [x] Structure scoring: evaluate against `get_framework(schema.framework).required_keys`
      (framework-agnostic — works with CO-STAR, RACE, APE, CRISPE)
- [x] Heuristic scoring: clarity, specificity, constraints, token efficiency
- [x] Risk detection: prompt injection, token bloat, ambiguity/contradictions
- [x] Human-readable feedback generation (strengths, issues, suggestions)
- [x] Simulation result integration (format compliance, missing/unwanted content feedback)
- [x] Unit tests: 37 tests covering each scoring dimension, risk detector,
      multi-framework evaluation, and test result integration — 188 tests passing

### 2F — LangGraph Orchestration (Reflection Loop)
- [x] `PromptBuilderState` TypedDict (includes `framework`, `domain`, `expected_format`,
      user_input, intent, current_prompt, test_result, quality_score, iteration tracking)
- [x] Wire nodes: extract_intent -> architect -> simulate -> linter
- [x] Conditional routing: score >= 7 -> finalize, else -> architect with critique
- [x] Max iteration cap (default 3) to prevent infinite loops
- [x] `finalize` node — prepare final output with score + iterations history
- [x] Unit tests with mocked agents: graph build, routing logic, individual nodes,
      iteration tracking (21 tests) — 209 tests passing
- [x] End-to-end integration test (full graph with live Ollama)
      7 e2e tests: individual agents, chained pipeline, full graph (CO-STAR + RACE)
- [x] Live-LLM parser hardening: case-insensitive keys, nested dict flattening,
      unknown key filtering, truncated JSON recovery
- [x] Pre-Phase-3 scenario test suite (65 tests across 11 test classes):
      parser resilience (12), architect parser resilience (9),
      schema edge cases (7), error recovery (7), APE full-pipeline (4),
      CRISPE full-pipeline (3), revision loop (7), content requirements (4),
      format compliance edge cases (8), graph state management (3),
      custom framework (1) — 302 total tests passing (295 unit + 7 e2e)

## Phase 3: Streamlit UI
- [ ] Framework selector dropdown (CO-STAR, RACE, APE, CRISPE)
- [ ] Dynamic section editor (fields adapt to selected framework)
- [ ] Three-column layout (Framework | Editor | Feedback)
- [ ] Connect UI to LangGraph backend
- [ ] Side-by-side prompt comparison view (A vs B)
- [ ] Real-time agent suggestions display
- [ ] Prompt history / versioning

## Phase 4: Packaging & Distribution
- [ ] Optimized production Dockerfile
- [ ] setup.sh auto-installs Ollama + pulls model
- [ ] Volume mapping for persistent prompt history
- [ ] End-to-end Docker deployment test
