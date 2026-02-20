# Frontend Integration Plan

This document outlines the plan to turn the static HTML prompt-builder mock into a functional desktop application. The approach: **Electron + HTML/CSS/JS + FastAPI backend** — reuse the existing KAIJU STATION design, wire it to the LangGraph pipeline via REST API, and wrap everything in Electron for a native desktop experience.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design. See [ROADMAP.md](ROADMAP.md) for project phases.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Electron App (Chromium)                                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  HTML/CSS/JS (prompt-builder-merged.html)       │   │
│  │  Loaded from http://127.0.0.1:8000/             │   │
│  └─────────────────────────┬───────────────────────┘   │
└────────────────────────────┼───────────────────────────┘
                             │ fetch API
                             ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend (:8000)                                  │
│  GET /  GET /api/frameworks  GET /api/domains            │
│  GET /api/health  POST /api/compile  POST /api/run       │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  LangGraph Pipeline (extract → architect → simulate →   │
│  linter → conditional refine loop)                       │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│  Ollama (Local LLM)                                       │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure (Additions)

```
IED_Prompter/
├── src/
│   └── api/                    # NEW — FastAPI app
│       ├── __init__.py
│       ├── main.py
│       └── schemas.py
├── frontend/                   # NEW — HTML + API wiring
│   └── index.html
├── electron/                   # NEW — Desktop wrapper
│   ├── package.json
│   ├── main.js
│   └── package-lock.json
└── ...
```

---

## 1. HTML ID Scheme (Required)

Every input that maps to an API parameter MUST have a deterministic ID. Add these when copying the HTML.

| DOM ID | Field | API param | Notes |
|--------|-------|-----------|-------|
| `#input-rough-concept` | Rough concept | `user_input` | Required |
| `#select-framework` | Framework | `framework` | Map CO-STAR → co_star |
| `#select-domain` | Domain | `domain` | |
| `#select-target-model` | Target Model | `target_model` | |
| `#select-response-format` | Response Format | `expected_format` | |
| `#input-max-tokens` | Max Tokens | `max_tokens` | 100–8192 |
| `#input-temperature` | Temperature | `temperature` | 0–2 (slider/100) |
| `#tag-area-must-include` | Must include tags | `must_include` | Array |
| `#tag-area-must-not` | Must not include | `must_not_include` | Array |
| `#tag-area-constraints` | Technical constraints | `constraints` | Array |
| `#var-rows` | Variables | `variables` | Object {key: value} |
| `#shot-pairs` | Few-shot examples | `examples` | Array of {input, output} |
| `#input-max-iterations` | Max Iterations | `max_iterations` | 1–10 |
| `#input-quality-threshold` | Quality Threshold | `quality_threshold` | 1–10 |

---

## 2. Payload Builder

- **Function:** `buildPayload()` — queries each ID, normalizes values, returns JSON for API.
- **Validation:** If `#input-rough-concept` is empty or whitespace-only, return `null` and do NOT call fetch. Block the request client-side.
- **Contract:** This mapping is the single source of truth for form → API.

---

## 3. State Management

- **Structure:** `appState = { form: {}, lastRun: null, previousRun: null }`
- **On framework change:** Reset output panels; set `lastRun`/`previousRun` to `null`; keep form inputs.
- **On RUN:** Shift `lastRun` → `previousRun`; set `lastRun` = API response.
- **COMPARE A/B:** Diff `lastRun` vs `previousRun`.

---

## 4. AbortController (Ghost State Prevention)

If the user clicks RUN, then COMPILE or changes framework before RUN returns, the old response can overwrite the UI. Fix: before every new API call, call `controller.abort()` on the previous AbortController. Pass `signal` to `fetch()`. Ignore aborted responses. Create a new AbortController per request.

---

## 5. System Status Indicator

- **Footer:** Grey = Initializing; Green = API Connected; Red = API/Ollama disconnected.
- **Poll:** `GET /api/health` on load and periodically.
- **Disable:** RUN and COMPILE buttons until status is Green.
- **Reason:** Prevents clicks before Python/LangChain finish loading on cold start.

---

## 6. Button Wiring

| Button | Action | Updates |
|--------|--------|---------|
| COMPILE | `POST /api/compile` | Compiled Prompt tab |
| RUN | `POST /api/run` | All output panels, sidebar meta, appState |
| COMPARE A/B | — | Diff lastRun vs previousRun |

---

## 7. Electron Wrapper

- **projectRoot:** `path.join(__dirname, '..')` when main.js is in `electron/`
- **Python path:** `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Mac/Linux)
- **spawn:** `python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000`
- **cwd:** `projectRoot` (required for knowledge-base paths)
- **env:** `PYTHONPATH: projectRoot`
- **On quit:** Kill Python child process.

---

## 8. Run Modes

| Mode | Command | Use case |
|------|---------|----------|
| API only | `uvicorn src.api.main:app --reload` | Dev in browser |
| Electron | `cd electron && npm start` | Desktop app |
| Manual | Start uvicorn + run Electron | Debugging |

---

## 9. Implementation Checklist

- [x] Copy `prompt-builder-merged.html` → `frontend/index.html`
- [x] Add required DOM ids per table
- [x] Implement `buildPayload()` with empty-input validation
- [x] Implement `appState` and framework-change reset
- [x] Implement AbortController for in-flight cancellation
- [x] Add system status indicator (footer) and 30s health polling
- [x] Wire COMPILE, RUN, COMPARE A/B to API
- [x] Add loading/error states

*All items completed Feb 20, 2026.*
