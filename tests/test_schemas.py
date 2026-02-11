"""Unit tests for Pydantic schemas — CO-STAR compat + multi-framework."""

import sys
import os

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schemas.frameworks import (
    FrameworkName,
    get_framework,
    list_frameworks,
    register_framework,
    FrameworkDef,
    SectionDef,
)
from src.schemas.prompt import (
    ModelTarget,
    PromptIteration,
    PromptSchema,
    PromptTestResult,
    QualityScore,
    ResponseFormat,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _minimal_prompt(**overrides) -> PromptSchema:
    """Return a valid PromptSchema with minimal required fields."""
    defaults = dict(
        context="You are a Python expert helping a junior developer learn testing.",
        objective="Write a unit test for an add() function",
        audience="junior Python developer",
    )
    defaults.update(overrides)
    return PromptSchema(**defaults)


def _minimal_quality_score(**overrides) -> QualityScore:
    defaults = dict(
        overall_score=8,
        clarity_score=8,
        specificity_score=7,
        structure_score=9,
        constraint_score=6,
        token_efficiency_score=8,
        strengths=["Clear objective"],
        issues=[],
        suggestions=[],
        reasoning="Good prompt.",
    )
    defaults.update(overrides)
    return QualityScore(**defaults)


# ── Enum tests ─────────────────────────────────────────────────────────────

class TestEnums:
    def test_model_target_values(self):
        assert ModelTarget.LLAMA3_8B.value == "llama3:8b"
        assert ModelTarget.MISTRAL.value == "mistral:7b"

    def test_response_format_values(self):
        assert ResponseFormat.JSON.value == "json"
        assert ResponseFormat.CODE.value == "code_block"
        assert ResponseFormat.LIST.value == "bulleted_list"


# ── PromptSchema validation ───────────────────────────────────────────────

class TestPromptSchemaValidation:
    def test_minimal_valid(self):
        p = _minimal_prompt()
        assert p.style == "professional and technical"
        assert p.tone == "helpful and informative"
        assert p.target_model == ModelTarget.LLAMA3_8B
        assert p.max_tokens == 2048
        assert p.temperature == 0.7

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            PromptSchema(context="short", objective="ok enough objective", audience="dev")

    def test_context_too_short(self):
        with pytest.raises(ValidationError):
            _minimal_prompt(context="too short")

    def test_objective_too_short(self):
        with pytest.raises(ValidationError):
            _minimal_prompt(objective="short")

    def test_blank_audience_rejected(self):
        with pytest.raises(ValidationError):
            _minimal_prompt(audience="   ")

    def test_whitespace_stripped(self):
        p = _minimal_prompt(
            context="  You are a Python expert helping a junior developer learn testing.  ",
            audience="  junior dev  ",
        )
        assert not p.context.startswith(" ")
        assert not p.audience.startswith(" ")

    def test_max_tokens_bounds(self):
        with pytest.raises(ValidationError):
            _minimal_prompt(max_tokens=50)
        with pytest.raises(ValidationError):
            _minimal_prompt(max_tokens=10000)

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            _minimal_prompt(temperature=-0.1)
        with pytest.raises(ValidationError):
            _minimal_prompt(temperature=2.5)

    def test_roundtrip_serialization(self):
        p = _minimal_prompt(
            must_include=["error handling"],
            constraints=["Python 3.8+"],
            variables={"user_name": "Jay"},
            examples=[{"input": "add(1,2)", "output": "3"}],
        )
        data = p.model_dump()
        restored = PromptSchema.model_validate(data)
        assert restored.must_include == ["error handling"]
        assert restored.variables == {"user_name": "Jay"}
        assert restored.examples[0]["input"] == "add(1,2)"


# ── compile_prompt() ──────────────────────────────────────────────────────

class TestCompilePrompt:
    def test_contains_core_sections(self):
        text = _minimal_prompt().compile_prompt()
        assert "**Context:**" in text
        assert "**Objective:**" in text
        assert "**Response Format:**" in text
        assert "**Style & Tone:**" in text

    def test_constraints_section_present_when_set(self):
        p = _minimal_prompt(constraints=["Python 3.8+"], must_not_include=["eval()"])
        text = p.compile_prompt()
        assert "**Constraints:**" in text
        assert "Python 3.8+" in text
        assert "Do NOT include: eval()" in text

    def test_constraints_section_absent_when_empty(self):
        text = _minimal_prompt().compile_prompt()
        assert "**Constraints:**" not in text

    def test_required_elements_section(self):
        p = _minimal_prompt(must_include=["docstring", "type hints"])
        text = p.compile_prompt()
        assert "**Required Elements:**" in text
        assert "docstring" in text
        assert "type hints" in text

    def test_examples_section(self):
        p = _minimal_prompt(examples=[{"input": "2+2", "output": "4"}])
        text = p.compile_prompt()
        assert "**Examples:**" in text
        assert "Input: 2+2" in text
        assert "Output: 4" in text

    def test_variable_injection(self):
        p = _minimal_prompt(
            context="Help {{user_name}} build a {{project_type}} project in Python.",
            variables={"user_name": "Jay", "project_type": "web scraper"},
        )
        text = p.compile_prompt()
        assert "Jay" in text
        assert "web scraper" in text
        assert "{{user_name}}" not in text

    def test_audience_and_style_in_output(self):
        p = _minimal_prompt(style="casual", tone="encouraging", audience="beginner")
        text = p.compile_prompt()
        assert "beginner" in text
        assert "casual" in text
        assert "encouraging" in text


# ── token_estimate() ──────────────────────────────────────────────────────

class TestTokenEstimate:
    def test_returns_positive_int(self):
        est = _minimal_prompt().token_estimate()
        assert isinstance(est, int)
        assert est > 0

    def test_longer_prompt_has_more_tokens(self):
        short = _minimal_prompt()
        long = _minimal_prompt(
            must_include=["a", "b", "c"],
            constraints=["x", "y", "z"],
            examples=[{"input": "foo", "output": "bar"}],
        )
        assert long.token_estimate() > short.token_estimate()


# ── PromptTestResult ──────────────────────────────────────────────────────

class TestPromptTestResult:
    def test_valid_creation(self):
        r = PromptTestResult(
            prompt_used="test prompt",
            model_response="hello world",
            token_count=5,
            execution_time_ms=120,
            model_used="llama3:8b",
            follows_format=True,
            includes_required=["greeting"],
            missing_required=[],
            unwanted_content=[],
        )
        assert r.follows_format is True
        assert r.token_count == 5

    def test_negative_token_count_rejected(self):
        with pytest.raises(ValidationError):
            PromptTestResult(
                prompt_used="x",
                model_response="y",
                token_count=-1,
                execution_time_ms=0,
                model_used="llama3:8b",
                follows_format=True,
            )


# ── QualityScore ──────────────────────────────────────────────────────────

class TestQualityScore:
    def test_passes_threshold_default(self):
        assert _minimal_quality_score(overall_score=7).passes_threshold()
        assert not _minimal_quality_score(overall_score=6).passes_threshold()

    def test_passes_threshold_custom(self):
        assert _minimal_quality_score(overall_score=5).passes_threshold(threshold=5)
        assert not _minimal_quality_score(overall_score=5).passes_threshold(threshold=6)

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            _minimal_quality_score(overall_score=0)
        with pytest.raises(ValidationError):
            _minimal_quality_score(overall_score=11)
        with pytest.raises(ValidationError):
            _minimal_quality_score(clarity_score=0)

    def test_risk_flags_default_false(self):
        q = _minimal_quality_score()
        assert q.has_injection_risk is False
        assert q.has_token_bloat is False
        assert q.has_ambiguity is False


# ── PromptIteration ──────────────────────────────────────────────────────

class TestPromptIteration:
    def test_valid_creation(self):
        prompt = _minimal_prompt()
        score = _minimal_quality_score()
        it = PromptIteration(
            iteration_number=1,
            prompt_version=prompt,
            quality_score=score,
            critique="Missing examples",
            action_taken="Initial draft",
        )
        assert it.iteration_number == 1
        assert it.test_result is None

    def test_iteration_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            PromptIteration(
                iteration_number=0,
                prompt_version=_minimal_prompt(),
                quality_score=_minimal_quality_score(),
                critique="n/a",
                action_taken="n/a",
            )

    def test_with_test_result(self):
        result = PromptTestResult(
            prompt_used="p",
            model_response="r",
            token_count=10,
            execution_time_ms=50,
            model_used="llama3:8b",
            follows_format=True,
        )
        it = PromptIteration(
            iteration_number=2,
            prompt_version=_minimal_prompt(),
            test_result=result,
            quality_score=_minimal_quality_score(),
            critique="Needs examples",
            action_taken="Added few-shot examples",
        )
        assert it.test_result is not None
        assert it.test_result.token_count == 10


# ── Framework Registry ───────────────────────────────────────────────────

class TestFrameworkRegistry:
    def test_list_frameworks_returns_all_four(self):
        names = list_frameworks()
        assert FrameworkName.COSTAR in names
        assert FrameworkName.RACE in names
        assert FrameworkName.APE in names
        assert FrameworkName.CRISPE in names

    def test_get_framework_returns_correct_def(self):
        defn = get_framework(FrameworkName.RACE)
        assert defn.display_name == "RACE"
        assert "role" in defn.required_keys
        assert "action" in defn.required_keys

    def test_register_custom_framework(self):
        custom = FrameworkDef(
            name="custom_test",
            display_name="Custom Test",
            sections=(
                SectionDef("goal", "Goal", "What to achieve", required=True),
                SectionDef("details", "Details", "Extra info", default="none"),
            ),
            compile_template="**Goal:**\n{goal}\n\n**Details:**\n{details}\n",
        )
        register_framework(custom)
        retrieved = get_framework("custom_test")
        assert retrieved.display_name == "Custom Test"

    def test_custom_framework_prompt_creation(self):
        custom = FrameworkDef(
            name="simple_qa",
            display_name="Simple Q&A",
            sections=(
                SectionDef("question", "Question", "The question to answer", required=True),
                SectionDef("background", "Background", "Extra context", default="none"),
            ),
            compile_template="{constraints_block}{required_elements_block}{examples_block}**Question:**\n{question}\n\n**Background:**\n{background}\n",
        )
        register_framework(custom)
        p = PromptSchema(
            framework="simple_qa",
            sections={"question": "What is Python?"},
        )
        text = p.compile_prompt()
        assert "**Question:**" in text
        assert "What is Python?" in text


# ── Multi-Framework PromptSchema ─────────────────────────────────────────

class TestMultiFramework:
    def test_race_valid_creation(self):
        p = PromptSchema(
            framework=FrameworkName.RACE,
            sections={
                "role": "Senior Python developer",
                "action": "Review the following code for security vulnerabilities",
                "context": "A web application handling user authentication with Flask",
                "expectation": "A bulleted list of vulnerabilities with severity ratings",
            },
        )
        assert p.framework == FrameworkName.RACE
        assert p.sections["role"] == "Senior Python developer"

    def test_race_compile_prompt(self):
        p = PromptSchema(
            framework=FrameworkName.RACE,
            sections={
                "role": "Senior Python developer",
                "action": "Review the following code for security vulnerabilities",
                "context": "A web application handling user authentication with Flask",
                "expectation": "A bulleted list of vulnerabilities with severity ratings",
            },
        )
        text = p.compile_prompt()
        assert "**Role:**" in text
        assert "**Action:**" in text
        assert "**Context:**" in text
        assert "**Expectation:**" in text
        assert "Senior Python developer" in text

    def test_ape_valid_creation(self):
        p = PromptSchema(
            framework=FrameworkName.APE,
            sections={
                "action": "Summarize the key findings from this research paper",
                "purpose": "To create an executive briefing for stakeholders",
                "expectation": "3-5 bullet points, no jargon, under 200 words",
            },
        )
        text = p.compile_prompt()
        assert "**Action:**" in text
        assert "**Purpose:**" in text
        assert "**Expectation:**" in text

    def test_crispe_valid_creation(self):
        p = PromptSchema(
            framework=FrameworkName.CRISPE,
            sections={
                "context": "Building a customer support chatbot for an e-commerce platform",
                "role": "UX writing specialist",
                "instruction": "Generate 5 empathetic response templates for order delay complaints",
                "schema": "JSON array with fields: id, trigger_phrase, response_text",
            },
        )
        text = p.compile_prompt()
        assert "**Context:**" in text
        assert "**Role:**" in text
        assert "**Instruction:**" in text
        assert "**Schema:**" in text
        assert "**Persona:**" in text  # default applied

    def test_missing_required_section_raises(self):
        with pytest.raises(ValidationError, match="required"):
            PromptSchema(
                framework=FrameworkName.RACE,
                sections={
                    "role": "Developer",
                    # missing action, context, expectation
                },
            )

    def test_unknown_section_key_raises(self):
        with pytest.raises(ValidationError, match="Unknown sections"):
            PromptSchema(
                framework=FrameworkName.APE,
                sections={
                    "action": "Do something important and meaningful",
                    "purpose": "For a good reason",
                    "expectation": "Good output",
                    "bogus_field": "should not be here",
                },
            )

    def test_framework_defaults_applied(self):
        p = PromptSchema(
            framework=FrameworkName.CRISPE,
            sections={
                "context": "Building a customer support chatbot for an e-commerce platform",
                "role": "UX writer",
                "instruction": "Generate response templates for complaints",
                "schema": "JSON array",
            },
        )
        # persona should get the default
        assert p.sections["persona"] == "professional"

    def test_costar_backward_compat_with_sections(self):
        """CO-STAR can also be created with explicit sections dict."""
        p = PromptSchema(
            framework=FrameworkName.COSTAR,
            sections={
                "context": "You are a Python expert helping a junior developer learn testing.",
                "objective": "Write a unit test for an add() function",
                "audience": "junior Python developer",
            },
        )
        assert p.context == "You are a Python expert helping a junior developer learn testing."
        assert p.objective == "Write a unit test for an add() function"
        assert p.style == "professional and technical"  # default

    def test_race_with_constraints_and_examples(self):
        p = PromptSchema(
            framework=FrameworkName.RACE,
            sections={
                "role": "Data analyst",
                "action": "Write a SQL query to find duplicate customer records",
                "context": "PostgreSQL database with a customers table containing 1M rows",
                "expectation": "Optimized SQL query with explanation",
            },
            constraints=["Must use window functions"],
            must_include=["PARTITION BY"],
            examples=[{"input": "Find duplicates by email", "output": "SELECT ... PARTITION BY email"}],
        )
        text = p.compile_prompt()
        assert "**Constraints:**" in text
        assert "Must use window functions" in text
        assert "**Required Elements:**" in text
        assert "**Examples:**" in text

    def test_min_length_enforced_per_framework(self):
        """RACE requires context min_length=20 and action min_length=10."""
        with pytest.raises(ValidationError, match="at least"):
            PromptSchema(
                framework=FrameworkName.RACE,
                sections={
                    "role": "Dev",
                    "action": "short",  # < 10 chars
                    "context": "A web application handling user authentication with Flask",
                    "expectation": "Good output",
                },
            )
