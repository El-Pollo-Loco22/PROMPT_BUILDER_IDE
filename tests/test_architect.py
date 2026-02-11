"""Unit tests for Phase 2C — ArchitectAgent with mocked LLM."""

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.architect import (
    ArchitectAgent,
    build_architect_prompt,
    load_knowledge_base,
    _format_best_practices,
)
from src.schemas.frameworks import FrameworkName, get_framework
from src.schemas.prompt import PromptSchema


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm_response(content: str):
    """Create a mock LLM response object with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_architect(knowledge_base=None):
    """Create an ArchitectAgent with mocked LLM (no real Ollama needed)."""
    agent = ArchitectAgent.__new__(ArchitectAgent)
    agent.llm = AsyncMock()
    agent.knowledge_base = knowledge_base or {}
    return agent


# ── Mock LLM responses ──────────────────────────────────────────────────

COSTAR_RESPONSE = json.dumps({
    "context": "A Python web application using Flask for a small business",
    "objective": "Create a user registration endpoint with email validation",
    "style": "Clean, production-ready Python code with type hints",
    "tone": "Professional and instructive",
    "audience": "Intermediate Python developers",
    "response": "Provide the complete endpoint code with inline comments",
})

COSTAR_FENCED = f"""```json
{COSTAR_RESPONSE}
```"""

COSTAR_WITH_PREAMBLE = f"""Here's the structured prompt:
{COSTAR_RESPONSE}
I've included all required sections."""

RACE_RESPONSE = json.dumps({
    "role": "Senior Python developer specializing in REST APIs",
    "action": "Design and implement a rate-limiting middleware",
    "context": "FastAPI application serving public endpoints with high traffic",
    "expectation": "Complete middleware code with configuration options and tests",
})

APE_RESPONSE = json.dumps({
    "action": "Generate a SQL query to find duplicate records",
    "purpose": "Clean up a customer database before migration",
    "expectation": "Standard SQL compatible with PostgreSQL 14+",
})

CRISPE_RESPONSE = json.dumps({
    "context": "Enterprise CRM system migration project",
    "role": "Senior data engineer with ETL expertise",
    "instruction": "Design a data migration pipeline from legacy Oracle DB to PostgreSQL",
    "schema": "Provide a step-by-step plan with rollback procedures",
    "persona": "Methodical, risk-aware, detail-oriented",
    "crispe_examples": "Include sample transformation rules for common data types",
})

REVISED_COSTAR_RESPONSE = json.dumps({
    "context": "A Python web application using Flask for a small business, handling user data with GDPR compliance",
    "objective": "Create a secure user registration endpoint with email validation and data encryption",
    "style": "Clean, production-ready Python code with type hints and security best practices",
    "tone": "Professional and security-focused",
    "audience": "Intermediate Python developers familiar with web security",
    "response": "Provide the complete endpoint code with security annotations and OWASP compliance notes",
})


# ── load_knowledge_base tests ────────────────────────────────────────────

class TestLoadKnowledgeBase:
    def test_load_from_project_dir(self):
        kb = load_knowledge_base("knowledge-base")
        assert "Software Development" in kb
        assert "General" in kb

    def test_load_nonexistent_dir(self):
        kb = load_knowledge_base("nonexistent-dir-xyz")
        assert kb == {}

    def test_loaded_data_has_best_practices(self):
        kb = load_knowledge_base("knowledge-base")
        sw = kb.get("Software Development", {})
        assert "bestPractices" in sw
        assert len(sw["bestPractices"]) > 0


# ── _format_best_practices tests ─────────────────────────────────────────

class TestFormatBestPractices:
    def test_formats_known_domain(self):
        kb = load_knowledge_base("knowledge-base")
        result = _format_best_practices(kb, "Software Development")
        assert "Specify the programming language" in result
        assert result.startswith("- ")

    def test_falls_back_to_general(self):
        kb = load_knowledge_base("knowledge-base")
        result = _format_best_practices(kb, "UnknownDomain")
        # Should fall back to General
        assert "- " in result or result == ""

    def test_empty_kb_returns_empty(self):
        result = _format_best_practices({}, "anything")
        assert result == ""

    def test_limit_parameter(self):
        kb = load_knowledge_base("knowledge-base")
        result = _format_best_practices(kb, "Software Development", limit=2)
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) <= 2


# ── build_architect_prompt tests ─────────────────────────────────────────

class TestBuildArchitectPrompt:
    def test_includes_framework_name(self):
        fw = get_framework(FrameworkName.COSTAR)
        prompt = build_architect_prompt(fw)
        assert "CO-STAR" in prompt

    def test_includes_section_descriptions(self):
        fw = get_framework(FrameworkName.COSTAR)
        prompt = build_architect_prompt(fw)
        assert "context" in prompt.lower()
        assert "objective" in prompt.lower()
        assert "audience" in prompt.lower()

    def test_includes_best_practices_when_provided(self):
        fw = get_framework(FrameworkName.COSTAR)
        practices = "- Always specify the language version"
        prompt = build_architect_prompt(fw, best_practices=practices)
        assert "Always specify the language version" in prompt

    def test_no_revision_section_without_critique(self):
        fw = get_framework(FrameworkName.COSTAR)
        prompt = build_architect_prompt(fw)
        assert "REVISION MODE" not in prompt

    def test_includes_revision_section_with_critique(self):
        fw = get_framework(FrameworkName.COSTAR)
        critique = "The prompt lacks specificity in the context section."
        prompt = build_architect_prompt(fw, critique=critique)
        assert "REVISION MODE" in prompt
        assert "lacks specificity" in prompt

    def test_race_framework(self):
        fw = get_framework(FrameworkName.RACE)
        prompt = build_architect_prompt(fw)
        assert "RACE" in prompt
        assert "role" in prompt.lower()
        assert "action" in prompt.lower()

    def test_requests_json_output(self):
        fw = get_framework(FrameworkName.COSTAR)
        prompt = build_architect_prompt(fw)
        assert "JSON" in prompt


# ── _parse_response tests ────────────────────────────────────────────────

class TestParseResponse:
    def test_clean_json_costar(self):
        result = ArchitectAgent._parse_response(COSTAR_RESPONSE, FrameworkName.COSTAR.value)
        assert isinstance(result, PromptSchema)
        assert result.framework == FrameworkName.COSTAR.value
        assert result.context == "A Python web application using Flask for a small business"
        assert result.objective == "Create a user registration endpoint with email validation"

    def test_fenced_json(self):
        result = ArchitectAgent._parse_response(COSTAR_FENCED, FrameworkName.COSTAR.value)
        assert isinstance(result, PromptSchema)
        assert "Flask" in result.context

    def test_json_with_preamble(self):
        result = ArchitectAgent._parse_response(COSTAR_WITH_PREAMBLE, FrameworkName.COSTAR.value)
        assert isinstance(result, PromptSchema)
        assert "Flask" in result.context

    def test_race_response(self):
        result = ArchitectAgent._parse_response(RACE_RESPONSE, FrameworkName.RACE.value)
        assert result.framework == FrameworkName.RACE.value
        assert result.sections["role"] == "Senior Python developer specializing in REST APIs"
        assert result.sections["action"] == "Design and implement a rate-limiting middleware"

    def test_ape_response(self):
        result = ArchitectAgent._parse_response(APE_RESPONSE, FrameworkName.APE.value)
        assert result.framework == FrameworkName.APE.value
        assert result.sections["action"] == "Generate a SQL query to find duplicate records"
        assert result.sections["purpose"] == "Clean up a customer database before migration"

    def test_crispe_response(self):
        result = ArchitectAgent._parse_response(CRISPE_RESPONSE, FrameworkName.CRISPE.value)
        assert result.framework == FrameworkName.CRISPE.value
        assert result.sections["role"] == "Senior data engineer with ETL expertise"
        assert "migration pipeline" in result.sections["instruction"]
        assert result.sections["crispe_examples"] == "Include sample transformation rules for common data types"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            ArchitectAgent._parse_response("This has no JSON.", FrameworkName.COSTAR.value)

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            ArchitectAgent._parse_response("{bad: json???}", FrameworkName.COSTAR.value)


# ── _build_user_message tests ────────────────────────────────────────────

class TestBuildUserMessage:
    def test_includes_intent_sections(self):
        fw = get_framework(FrameworkName.COSTAR)
        intent = {
            "sections": {
                "objective": "Write a sorting algorithm",
                "context": "Python homework assignment",
            }
        }
        msg = ArchitectAgent._build_user_message(intent, fw)
        assert "sorting algorithm" in msg
        assert "homework" in msg

    def test_includes_constraints(self):
        fw = get_framework(FrameworkName.COSTAR)
        intent = {
            "sections": {"objective": "Test"},
            "constraints": ["no recursion", "O(n log n)"],
        }
        msg = ArchitectAgent._build_user_message(intent, fw)
        assert "no recursion" in msg
        assert "O(n log n)" in msg

    def test_includes_missing_variables(self):
        fw = get_framework(FrameworkName.COSTAR)
        intent = {
            "sections": {"objective": "Test"},
            "missing_variables": ["language", "framework"],
        }
        msg = ArchitectAgent._build_user_message(intent, fw)
        assert "language" in msg
        assert "framework" in msg

    def test_includes_critique_in_revision(self):
        fw = get_framework(FrameworkName.COSTAR)
        intent = {"sections": {"objective": "Test"}}
        critique = "Context section is too vague."
        msg = ArchitectAgent._build_user_message(intent, fw, critique=critique)
        assert "too vague" in msg

    def test_fallback_to_top_level_keys(self):
        fw = get_framework(FrameworkName.COSTAR)
        intent = {
            "objective": "Direct top-level objective",
            "context": "Direct context",
        }
        msg = ArchitectAgent._build_user_message(intent, fw)
        assert "Direct top-level objective" in msg
        assert "Direct context" in msg


# ── Async draft() with mocked LLM ───────────────────────────────────────

class TestDraftAsync:
    @pytest.mark.asyncio
    async def test_draft_costar(self):
        agent = _make_architect()
        agent.llm.ainvoke.return_value = _mock_llm_response(COSTAR_RESPONSE)

        intent = {
            "sections": {
                "objective": "Create a registration endpoint",
                "context": "Flask web app",
            }
        }
        result = await agent.draft(intent, framework=FrameworkName.COSTAR.value)

        assert isinstance(result, PromptSchema)
        assert result.framework == FrameworkName.COSTAR.value
        assert "registration" in result.objective.lower()
        agent.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_draft_race(self):
        agent = _make_architect()
        agent.llm.ainvoke.return_value = _mock_llm_response(RACE_RESPONSE)

        intent = {
            "sections": {
                "role": "Developer",
                "action": "Build rate limiter",
            }
        }
        result = await agent.draft(intent, framework=FrameworkName.RACE.value)

        assert result.framework == FrameworkName.RACE.value
        assert "rate-limiting" in result.sections["action"].lower()

    @pytest.mark.asyncio
    async def test_draft_with_critique(self):
        agent = _make_architect()
        agent.llm.ainvoke.return_value = _mock_llm_response(REVISED_COSTAR_RESPONSE)

        intent = {"sections": {"objective": "Create a registration endpoint"}}
        critique = "Add security considerations and GDPR compliance."

        result = await agent.draft(
            intent,
            framework=FrameworkName.COSTAR.value,
            critique=critique,
        )

        assert isinstance(result, PromptSchema)
        assert "GDPR" in result.context or "security" in result.sections.get("style", "").lower()

        # Verify the system prompt included revision mode
        call_args = agent.llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "REVISION MODE" in system_msg

    @pytest.mark.asyncio
    async def test_draft_with_knowledge_base(self):
        kb = load_knowledge_base("knowledge-base")
        agent = _make_architect(knowledge_base=kb)
        agent.llm.ainvoke.return_value = _mock_llm_response(COSTAR_RESPONSE)

        intent = {"sections": {"objective": "Write a function"}}
        await agent.draft(intent, domain="Software Development")

        # Verify the system prompt included best practices
        call_args = agent.llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "best practices" in system_msg.lower()

    @pytest.mark.asyncio
    async def test_draft_passes_intent_in_user_message(self):
        agent = _make_architect()
        agent.llm.ainvoke.return_value = _mock_llm_response(COSTAR_RESPONSE)

        intent = {
            "sections": {"objective": "My unique test objective xyz123"},
        }
        await agent.draft(intent)

        call_args = agent.llm.ainvoke.call_args[0][0]
        user_msg = call_args[1].content
        assert "xyz123" in user_msg


# ── Sync draft_sync() with mocked LLM ───────────────────────────────────

class TestDraftSync:
    def test_draft_sync_costar(self):
        agent = _make_architect()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = _mock_llm_response(COSTAR_RESPONSE)

        intent = {"sections": {"objective": "Create a registration endpoint"}}
        result = agent.draft_sync(intent, framework=FrameworkName.COSTAR.value)

        assert isinstance(result, PromptSchema)
        assert result.framework == FrameworkName.COSTAR.value
        agent.llm.invoke.assert_called_once()

    def test_draft_sync_race(self):
        agent = _make_architect()
        agent.llm = MagicMock()
        agent.llm.invoke.return_value = _mock_llm_response(RACE_RESPONSE)

        intent = {"sections": {"role": "Developer", "action": "Build API"}}
        result = agent.draft_sync(intent, framework=FrameworkName.RACE.value)

        assert result.framework == FrameworkName.RACE.value
        assert result.sections["role"] == "Senior Python developer specializing in REST APIs"


# ── Compile round-trip test ──────────────────────────────────────────────

class TestCompileRoundTrip:
    def test_drafted_prompt_compiles(self):
        """Verify that a drafted PromptSchema can compile_prompt() without error."""
        result = ArchitectAgent._parse_response(COSTAR_RESPONSE, FrameworkName.COSTAR.value)
        compiled = result.compile_prompt()
        assert "Context" in compiled
        assert "Objective" in compiled
        assert "Flask" in compiled

    def test_race_drafted_prompt_compiles(self):
        result = ArchitectAgent._parse_response(RACE_RESPONSE, FrameworkName.RACE.value)
        compiled = result.compile_prompt()
        assert "Role" in compiled
        assert "Action" in compiled
        assert "rate-limiting" in compiled
