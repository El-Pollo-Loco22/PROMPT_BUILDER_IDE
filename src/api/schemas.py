"""
API Request/Response Schemas (Phase 3)

Pydantic models that map between the frontend JSON shapes and
the LangGraph PromptBuilderState.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """POST /api/run and POST /api/compile request body."""

    user_input: str = Field(..., min_length=1, description="The rough prompt concept.")
    framework: str = Field(
        default="co_star",
        description="Framework key: co_star, race, ape, crispe.",
    )
    domain: Optional[str] = Field(default="General")
    expected_format: Optional[str] = Field(default="plain_text")
    max_iterations: Optional[int] = Field(default=3, ge=1, le=10)
    max_tokens: Optional[int] = Field(default=None, ge=100, le=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    quality_threshold: Optional[int] = Field(default=7, ge=1, le=10)
    target_model: Optional[str] = Field(default=None)

    # User overrides (Constraints tab)
    must_include: Optional[List[str]] = Field(default=None)
    must_not_include: Optional[List[str]] = Field(default=None)
    constraints: Optional[List[str]] = Field(default=None)
    variables: Optional[Dict[str, str]] = Field(default=None)
    examples: Optional[List[Dict[str, str]]] = Field(default=None)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class IterationResponse(BaseModel):
    """A single iteration in the history."""

    iteration_number: int
    prompt_text: str = Field(description="Compiled prompt string for this iteration.")
    overall_score: int
    clarity_score: int
    specificity_score: int
    structure_score: int
    constraint_score: int
    token_efficiency_score: int
    strengths: List[str]
    issues: List[str]
    suggestions: List[str]
    critique: str
    action_taken: str

    # Simulation results
    model_response: Optional[str] = None
    follows_format: Optional[bool] = None
    includes_required: Optional[List[str]] = None
    missing_required: Optional[List[str]] = None
    unwanted_content: Optional[List[str]] = None


class CompileResponse(BaseModel):
    """POST /api/compile response."""

    compiled: str = Field(description="The compiled prompt string.")
    prompt_schema: Dict[str, Any] = Field(description="Full PromptSchema as dict.")


class RunResponse(BaseModel):
    """POST /api/run response."""

    final_prompt: str = Field(description="Compiled final prompt string.")
    final_prompt_schema: Dict[str, Any] = Field(description="Full PromptSchema as dict.")
    final_score: Dict[str, Any] = Field(description="QualityScore as dict.")
    iterations_history: List[IterationResponse]
    test_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Last simulation PromptTestResult as dict.",
    )
    status: str


class FrameworkItem(BaseModel):
    """A single framework for the dropdown."""

    name: str
    display_name: str


class HealthResponse(BaseModel):
    """GET /api/health response."""

    status: str  # "ok" or "error"
    ollama_reachable: bool
    detail: Optional[str] = None
