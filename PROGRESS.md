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
- [ ] `ArchitectAgent` — builds/revises prompts using selected framework's structure
      (must call `get_framework()` to discover required sections)
- [ ] Domain-specific knowledge base integration (load `knowledge-base/*.json`)
- [ ] Critique-driven revision mode (accept linter feedback + `QualityScore.issues`,
      revise `PromptSchema.sections` accordingly)
- [ ] Unit tests with mocked LLM responses (test with CO-STAR + at least one alt framework)

### 2D — Simulation Node
- [ ] `SimulationNode` — run `prompt.compile_prompt()` against local Ollama
      (framework-agnostic: compile_prompt() handles framework rendering)
- [ ] Capture: response text, token count, execution time
- [ ] Format compliance check (JSON, code, markdown, list, plain text)
- [ ] Required/unwanted content detection against `PromptSchema.must_include`/`must_not_include`
- [ ] Integration test against live Ollama (llama3:8b)

### 2E — Linter Agent (Analysis Node)
- [ ] `LinterAgent` — multi-dimensional quality evaluation
- [ ] Structure scoring: evaluate against `get_framework(schema.framework).required_keys`
      (not hardcoded CO-STAR — must work for any framework)
- [ ] Heuristic scoring: clarity, specificity, constraints, token efficiency
- [ ] Risk detection: prompt injection, token bloat, ambiguity/contradictions
- [ ] Human-readable feedback generation (strengths, issues, suggestions)
- [ ] Unit tests for each scoring dimension and risk detector (test across frameworks)

### 2F — LangGraph Orchestration (Reflection Loop)
- [ ] `PromptBuilderState` TypedDict (must include `framework: str` alongside
      shared state: user_input, intent, current_prompt, test_result, quality_score, etc.)
- [ ] Wire nodes: extract_intent -> architect -> simulate -> linter
- [ ] Conditional routing: score >= 7 -> finalize, else -> architect with critique
- [ ] Max iteration cap (default 3) to prevent infinite loops
- [ ] `finalize` node — prepare final output with score + iterations history
- [ ] End-to-end integration test (full graph with mocked Ollama)
- [ ] End-to-end integration test (full graph with live Ollama)

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
