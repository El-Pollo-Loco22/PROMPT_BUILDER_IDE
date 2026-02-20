"""
FastAPI Backend (Phase 3)

Serves the frontend and exposes REST endpoints for the LangGraph pipeline.
Entrypoint: uvicorn src.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.agents.architect import ArchitectAgent, load_knowledge_base
from src.agents.intent_extractor import IntentExtractor
from src.api.schemas import (
    CompileResponse,
    FrameworkItem,
    HealthResponse,
    IterationResponse,
    RunRequest,
    RunResponse,
)
from src.config import settings
from src.graph.builder import PromptBuilderState, compile_prompt_graph
from src.schemas.frameworks import get_framework, list_frameworks
from src.schemas.prompt import PromptSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
_KB_DIR = _PROJECT_ROOT / "knowledge-base"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prompt Builder IDE API",
    version="0.1.0",
    description="REST API for the LangGraph prompt engineering pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# GET /  — serve frontend
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(str(index_path), media_type="text/html")


# ---------------------------------------------------------------------------
# GET /api/health
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check Ollama reachability."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                return HealthResponse(status="ok", ollama_reachable=True)
            return HealthResponse(
                status="error",
                ollama_reachable=False,
                detail=f"Ollama returned {resp.status_code}",
            )
    except Exception as exc:
        return HealthResponse(
            status="error",
            ollama_reachable=False,
            detail=str(exc),
        )


# ---------------------------------------------------------------------------
# GET /api/frameworks
# ---------------------------------------------------------------------------


@app.get("/api/frameworks", response_model=List[FrameworkItem])
async def get_frameworks():
    """List all registered frameworks for dropdowns."""
    items = []
    for name in list_frameworks():
        fw = get_framework(name)
        items.append(FrameworkItem(name=name, display_name=fw.display_name))
    return items


# ---------------------------------------------------------------------------
# GET /api/domains
# ---------------------------------------------------------------------------


@app.get("/api/domains")
async def get_domains():
    """List available knowledge-base domains."""
    kb = load_knowledge_base(_KB_DIR)
    domains = sorted(kb.keys()) if kb else ["General"]
    if "General" not in domains:
        domains.insert(0, "General")
    return {"domains": domains}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_overrides(req: RunRequest) -> Dict[str, Any]:
    """Extract user override fields from the request."""
    overrides: Dict[str, Any] = {}
    if req.must_include:
        overrides["must_include"] = req.must_include
    if req.must_not_include:
        overrides["must_not_include"] = req.must_not_include
    if req.constraints:
        overrides["constraints"] = req.constraints
    if req.variables:
        overrides["variables"] = req.variables
    if req.examples:
        overrides["examples"] = req.examples
    if req.target_model:
        overrides["target_model"] = req.target_model
    if req.temperature is not None:
        overrides["temperature"] = req.temperature
    if req.max_tokens is not None:
        overrides["max_tokens"] = req.max_tokens
    return overrides


def _serialize_iteration(it: Any) -> IterationResponse:
    """Convert a PromptIteration into a safe response dict."""
    # it is a PromptIteration Pydantic model
    test = it.test_result
    return IterationResponse(
        iteration_number=it.iteration_number,
        prompt_text=it.prompt_version.compile_prompt(),
        overall_score=it.quality_score.overall_score,
        clarity_score=it.quality_score.clarity_score,
        specificity_score=it.quality_score.specificity_score,
        structure_score=it.quality_score.structure_score,
        constraint_score=it.quality_score.constraint_score,
        token_efficiency_score=it.quality_score.token_efficiency_score,
        strengths=it.quality_score.strengths,
        issues=it.quality_score.issues,
        suggestions=it.quality_score.suggestions,
        critique=it.critique,
        action_taken=it.action_taken,
        model_response=test.model_response if test else None,
        follows_format=test.follows_format if test else None,
        includes_required=test.includes_required if test else None,
        missing_required=test.missing_required if test else None,
        unwanted_content=test.unwanted_content if test else None,
    )


def _schema_to_dict(schema: PromptSchema) -> Dict[str, Any]:
    """Serialize a PromptSchema to a JSON-safe dict (handles enums)."""
    data = schema.model_dump()
    # Ensure enum values are strings
    if hasattr(data.get("target_model"), "value"):
        data["target_model"] = data["target_model"].value
    elif isinstance(data.get("target_model"), str):
        pass  # already a string after model_dump
    return data


# ---------------------------------------------------------------------------
# POST /api/compile
# ---------------------------------------------------------------------------


@app.post("/api/compile", response_model=CompileResponse)
async def compile_prompt(req: RunRequest):
    """
    Extract intent + architect + merge overrides + compile.
    Returns compiled prompt text without running simulation.
    """
    try:
        # Extract intent
        extractor = IntentExtractor(framework=req.framework)
        intent = await extractor.extract(req.user_input)

        # Draft prompt
        architect = ArchitectAgent()
        prompt = await architect.draft(
            intent=intent.model_dump(),
            framework=req.framework,
            domain=req.domain or "General",
        )

        # Merge user overrides
        overrides = _build_user_overrides(req)
        if overrides:
            data = prompt.model_dump()
            override_fields = {}
            for key in ("must_include", "must_not_include", "constraints", "variables", "examples"):
                if key in overrides and overrides[key]:
                    override_fields[key] = overrides[key]
            for key in ("target_model", "temperature", "max_tokens"):
                if key in overrides and overrides[key] is not None:
                    override_fields[key] = overrides[key]
            if override_fields:
                data.update(override_fields)
                prompt = PromptSchema(**data)

        compiled = prompt.compile_prompt()
        return CompileResponse(
            compiled=compiled,
            prompt_schema=_schema_to_dict(prompt),
        )

    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid output. Try again or adjust your prompt. ({exc})",
        )
    except Exception as exc:
        logger.exception("Compile failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /api/run
# ---------------------------------------------------------------------------


@app.post("/api/run", response_model=RunResponse)
async def run_pipeline(req: RunRequest):
    """
    Full graph ainvoke: extract → architect → simulate → linter → refine loop.
    Returns serialized final state.
    """
    try:
        # Build initial state
        overrides = _build_user_overrides(req)
        initial_state: PromptBuilderState = {
            "user_input": req.user_input,
            "framework": req.framework,
            "domain": req.domain or "General",
            "expected_format": req.expected_format or "plain_text",
            "max_iterations": req.max_iterations or 3,
            "quality_threshold": req.quality_threshold or 7,
        }
        if overrides:
            initial_state["user_overrides"] = overrides

        # Compile and invoke graph
        graph = compile_prompt_graph()
        result = await graph.ainvoke(initial_state)

        # Serialize response
        final_prompt_schema = result.get("final_prompt") or result.get("current_prompt")
        final_score = result.get("final_score") or result.get("quality_score")
        iterations = result.get("iterations_history", [])
        test_result = result.get("test_result")

        return RunResponse(
            final_prompt=final_prompt_schema.compile_prompt() if final_prompt_schema else "",
            final_prompt_schema=_schema_to_dict(final_prompt_schema) if final_prompt_schema else {},
            final_score=final_score.model_dump() if final_score else {},
            iterations_history=[_serialize_iteration(it) for it in iterations],
            test_result=test_result.model_dump() if test_result else None,
            status=result.get("status", "complete"),
        )

    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model returned invalid output. Try again or adjust your prompt. ({exc})",
        )
    except Exception as exc:
        logger.exception("Run pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Static files — mount AFTER all routes
# ---------------------------------------------------------------------------

app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend-static")
