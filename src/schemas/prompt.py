from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.schemas.frameworks import FrameworkDef, FrameworkName, get_framework


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTarget(str, Enum):
    """Supported local Ollama models."""
    LLAMA3_8B = "llama3:8b"
    LLAMA3_70B = "llama3:70b"
    CODELLAMA = "codellama:7b"
    MISTRAL = "mistral:7b"


class ResponseFormat(str, Enum):
    """Expected response structure."""
    TEXT = "plain_text"
    JSON = "json"
    CODE = "code_block"
    MARKDOWN = "markdown"
    LIST = "bulleted_list"


# ---------------------------------------------------------------------------
# Legacy models (kept for backward-compat with Phase 1 smoke tests)
# ---------------------------------------------------------------------------

class PromptBlock(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    metadata: Optional[dict] = None


class PromptDocument(BaseModel):
    blocks: list[PromptBlock]
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Framework-Agnostic Prompt Schema
# ---------------------------------------------------------------------------

# CO-STAR field names that callers may pass as top-level kwargs
_LEGACY_COSTAR_FIELDS = {"context", "objective", "style", "tone", "audience"}


class PromptSchema(BaseModel):
    """
    Framework-agnostic prompt schema.

    Supports CO-STAR (default), RACE, APE, CRISPE, and custom frameworks.
    Framework-specific content lives in ``sections``; execution metadata
    and constraints are framework-agnostic.
    """

    # -- Framework selection -------------------------------------------------
    framework: str = Field(
        default=FrameworkName.COSTAR.value,
        description="Which prompt engineering framework to use.",
    )

    # -- Framework-specific sections (key → value) ---------------------------
    sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Ordered section content keyed by framework section name.",
    )

    # -- Execution metadata (framework-agnostic) -----------------------------
    target_model: ModelTarget = Field(
        default=ModelTarget.LLAMA3_8B,
        description="Which local Ollama model to target.",
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum response length.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Creativity vs consistency (0 = deterministic, 2 = creative).",
    )

    # -- Constraints & requirements ------------------------------------------
    must_include: List[str] = Field(
        default_factory=list,
        description="Required elements in the response.",
    )
    must_not_include: List[str] = Field(
        default_factory=list,
        description="Things to avoid in the response.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Technical or business limitations.",
    )

    # -- Variable placeholders -----------------------------------------------
    variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Placeholder variables to inject (e.g. {{user_name}}).",
    )

    # -- Few-shot examples ---------------------------------------------------
    examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Input-output examples to guide the model.",
    )

    # -- Validators ----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_and_apply_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. Backward compat: migrate top-level CO-STAR kwargs into sections.
        2. Apply framework defaults for missing optional sections.
        """
        if not isinstance(values, dict):
            return values

        framework_name = values.get("framework", FrameworkName.COSTAR)
        # Normalize to FrameworkName enum if it's a known built-in
        if isinstance(framework_name, str):
            try:
                framework_name = FrameworkName(framework_name)
            except ValueError:
                pass  # custom framework — keep as string
        framework_def = get_framework(framework_name)

        sections = dict(values.get("sections") or {})

        # Migrate legacy CO-STAR top-level fields into sections
        for key in _LEGACY_COSTAR_FIELDS:
            if key in values and key not in sections:
                sections[key] = values.pop(key)

        # Special case: response_format → response section
        if "response_format" in values and "response" not in sections:
            rf = values.pop("response_format")
            if isinstance(rf, ResponseFormat):
                sections["response"] = rf.value
            elif isinstance(rf, str):
                sections["response"] = rf

        # Apply defaults from framework definition
        for key, default in framework_def.defaults.items():
            if key not in sections:
                sections[key] = default

        values["sections"] = sections
        return values

    @model_validator(mode="after")
    def _validate_sections(self) -> PromptSchema:
        """Validate sections against the active framework definition."""
        framework_def = get_framework(self.framework)

        # Reject unknown section keys
        unknown = set(self.sections.keys()) - framework_def.all_keys
        if unknown:
            raise ValueError(
                f"Unknown sections for {framework_def.display_name}: {unknown}"
            )

        # Validate required sections are present and non-blank
        for key in framework_def.required_keys:
            val = self.sections.get(key, "").strip()
            if not val:
                raise ValueError(
                    f"Section '{key}' is required for {framework_def.display_name}"
                )
            self.sections[key] = val  # store stripped

        # Enforce min-length constraints from framework definition
        for key, min_len in framework_def.min_lengths.items():
            val = self.sections.get(key, "")
            if val and len(val) < min_len:
                raise ValueError(
                    f"'{key}' must be at least {min_len} characters"
                )

        return self

    # -- Backward-compatible property accessors (CO-STAR) --------------------

    @property
    def context(self) -> str:
        return self.sections.get("context", "")

    @property
    def objective(self) -> str:
        return self.sections.get("objective", "")

    @property
    def style(self) -> str:
        return self.sections.get("style", "professional and technical")

    @property
    def tone(self) -> str:
        return self.sections.get("tone", "helpful and informative")

    @property
    def audience(self) -> str:
        return self.sections.get("audience", "")

    @property
    def response_format(self) -> str:
        return self.sections.get("response", ResponseFormat.TEXT.value)

    # -- Methods -------------------------------------------------------------

    def compile_prompt(self) -> str:
        """Compile sections into a prompt string using the framework template."""
        framework_def = get_framework(self.framework)

        # Build shared blocks (constraints, required elements, examples)
        constraints_block = ""
        if self.constraints or self.must_not_include:
            lines = [f"- {c}" for c in self.constraints]
            lines += [f"- Do NOT include: {item}" for item in self.must_not_include]
            constraints_block = "**Constraints:**\n" + "\n".join(lines) + "\n\n"

        required_elements_block = ""
        if self.must_include:
            lines = [f"- {item}" for item in self.must_include]
            required_elements_block = "**Required Elements:**\n" + "\n".join(lines) + "\n\n"

        examples_block = ""
        if self.examples:
            ex_lines = [
                f"Input: {ex['input']}\nOutput: {ex['output']}\n"
                for ex in self.examples
            ]
            examples_block = "**Examples:**\n" + "\n".join(ex_lines) + "\n"

        # Format template with sections + shared blocks
        format_vars = {
            **self.sections,
            "constraints_block": constraints_block,
            "required_elements_block": required_elements_block,
            "examples_block": examples_block,
        }
        prompt = framework_def.compile_template.format(**format_vars)

        # Inject variables
        for key, value in self.variables.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)

        return prompt

    def token_estimate(self) -> int:
        """Rough token count estimate (~ 4 chars per token)."""
        return len(self.compile_prompt()) // 4


# ---------------------------------------------------------------------------
# Test / Evaluation Models
# ---------------------------------------------------------------------------

class PromptTestResult(BaseModel):
    """Result from testing a prompt against a local model."""

    prompt_used: str
    model_response: str
    token_count: int = Field(ge=0)
    execution_time_ms: int = Field(ge=0)
    model_used: str

    follows_format: bool = Field(
        description="Did the output match the expected format?",
    )
    includes_required: List[str] = Field(
        default_factory=list,
        description="Which required elements were found.",
    )
    missing_required: List[str] = Field(
        default_factory=list,
        description="Which required elements were missing.",
    )
    unwanted_content: List[str] = Field(
        default_factory=list,
        description="Detected must-not-include items present in output.",
    )


class QualityScore(BaseModel):
    """
    Linter evaluation of prompt quality.
    Score 1-10; 7+ passes threshold by default.
    """

    overall_score: int = Field(ge=1, le=10)

    # Component scores
    clarity_score: int = Field(ge=1, le=10)
    specificity_score: int = Field(ge=1, le=10)
    structure_score: int = Field(ge=1, le=10)
    constraint_score: int = Field(ge=1, le=10)
    token_efficiency_score: int = Field(ge=1, le=10)

    # Feedback
    strengths: List[str]
    issues: List[str]
    suggestions: List[str]

    # Risk flags
    has_injection_risk: bool = False
    has_token_bloat: bool = False
    has_ambiguity: bool = False

    reasoning: str = Field(description="Explanation of the score.")

    def passes_threshold(self, threshold: int = 7) -> bool:
        return self.overall_score >= threshold


class PromptIteration(BaseModel):
    """Tracks a single iteration inside the reflection loop."""

    iteration_number: int = Field(ge=1)
    prompt_version: PromptSchema
    test_result: Optional[PromptTestResult] = None
    quality_score: QualityScore
    critique: str
    action_taken: str
