"""Unit tests for Phase 2B — IntentExtractor with mocked LLM."""

import sys
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.intent_extractor import (
    ExtractedIntent,
    IntentExtractor,
    build_extraction_prompt,
)
from src.schemas.frameworks import FrameworkName, get_framework


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm_response(content: str):
    """Create a mock LLM response object with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_extractor(framework: str = FrameworkName.COSTAR.value):
    """Create an IntentExtractor with mocked LLM (no real Ollama needed)."""
    ext = IntentExtractor.__new__(IntentExtractor)
    ext.framework = framework
    ext.framework_def = get_framework(framework)
    ext.system_prompt = build_extraction_prompt(ext.framework_def)
    ext.llm = AsyncMock()
    return ext


# ── Mock LLM responses ────────────────────────────────────────────────────

CLEAN_JSON = """{
  "objective": "Create an email validation function",
  "context": "User needs input validation for a web form",
  "audience": "intermediate Python developer",
  "missing_variables": ["validation_rules", "return_type"],
  "constraints": ["no external dependencies"],
  "response_format": "code_block"
}"""

FENCED_JSON = """```json
{
  "objective": "Build a REST API endpoint",
  "context": "FastAPI project for e-commerce",
  "audience": "senior backend developer",
  "missing_variables": [],
  "constraints": ["Python 3.8+"],
  "response_format": "code_block"
}
```"""

JSON_WITH_PREAMBLE = """Sure! Here is the extracted intent:
{
  "objective": "Write a haiku about coding",
  "context": "User wants creative writing from an AI",
  "audience": "general audience",
  "missing_variables": ["theme"],
  "constraints": [],
  "response_format": "plain_text"
}
I hope this helps!"""

MINIMAL_JSON = """{
  "objective": "Summarize a document",
  "context": "User has a long report",
  "audience": "executive"
}"""

RACE_JSON = """{
  "role": "Senior security analyst",
  "action": "Review the authentication module for vulnerabilities",
  "context": "A Flask web app handling OAuth2 login flows for enterprise customers",
  "expectation": "Bulleted list of vulnerabilities with severity ratings",
  "missing_variables": ["codebase_url"],
  "constraints": ["focus on OWASP Top 10"],
  "response_format": "bulleted_list"
}"""


# ── _parse_response tests (no LLM needed) ─────────────────────────────────

class TestParseResponse:
    def test_clean_json(self):
        result = IntentExtractor._parse_response(CLEAN_JSON)
        assert isinstance(result, ExtractedIntent)
        assert result.objective == "Create an email validation function"
        assert result.audience == "intermediate Python developer"
        assert result.response_format == "code_block"
        assert "validation_rules" in result.missing_variables
        assert "no external dependencies" in result.constraints

    def test_fenced_json(self):
        result = IntentExtractor._parse_response(FENCED_JSON)
        assert result.objective == "Build a REST API endpoint"
        assert result.constraints == ["Python 3.8+"]

    def test_json_with_preamble(self):
        result = IntentExtractor._parse_response(JSON_WITH_PREAMBLE)
        assert result.objective == "Write a haiku about coding"
        assert result.audience == "general audience"

    def test_minimal_json_uses_defaults(self):
        result = IntentExtractor._parse_response(MINIMAL_JSON)
        assert result.missing_variables == []
        assert result.constraints == []
        assert result.response_format == "plain_text"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            IntentExtractor._parse_response("This is just plain text with no JSON.")

    def test_invalid_json_raises(self):
        with pytest.raises(Exception):
            IntentExtractor._parse_response("{bad json: ???}")

    def test_sections_populated(self):
        result = IntentExtractor._parse_response(CLEAN_JSON)
        assert result.sections["objective"] == "Create an email validation function"
        assert result.sections["context"] == "User needs input validation for a web form"
        assert result.sections["audience"] == "intermediate Python developer"


# ── Async extract() with mocked LLM ───────────────────────────────────────

class TestExtractAsync:
    @pytest.mark.asyncio
    async def test_extract_returns_intent(self):
        extractor = _make_extractor()
        extractor.llm.ainvoke.return_value = _mock_llm_response(CLEAN_JSON)

        result = await extractor.extract("Create a function to validate emails")

        assert isinstance(result, ExtractedIntent)
        assert result.objective == "Create an email validation function"
        extractor.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_handles_fenced_response(self):
        extractor = _make_extractor()
        extractor.llm.ainvoke.return_value = _mock_llm_response(FENCED_JSON)

        result = await extractor.extract("Build a REST API")

        assert result.objective == "Build a REST API endpoint"

    @pytest.mark.asyncio
    async def test_extract_passes_user_input_in_message(self):
        extractor = _make_extractor()
        extractor.llm.ainvoke.return_value = _mock_llm_response(CLEAN_JSON)

        await extractor.extract("my custom request")

        call_args = extractor.llm.ainvoke.call_args[0][0]
        assert "my custom request" in call_args[1].content


# ── Sync extract_sync() with mocked LLM ───────────────────────────────────

class TestExtractSync:
    def test_extract_sync_returns_intent(self):
        extractor = _make_extractor()
        extractor.llm = MagicMock()
        extractor.llm.invoke.return_value = _mock_llm_response(CLEAN_JSON)

        result = extractor.extract_sync("Validate emails")

        assert isinstance(result, ExtractedIntent)
        assert result.objective == "Create an email validation function"
        extractor.llm.invoke.assert_called_once()


# ── ExtractedIntent model tests ────────────────────────────────────────────

class TestExtractedIntentModel:
    def test_roundtrip(self):
        intent = ExtractedIntent(
            sections={
                "objective": "Test objective",
                "context": "Test context",
                "audience": "tester",
            },
            missing_variables=["x"],
            constraints=["y"],
            response_format="json",
        )
        data = intent.model_dump()
        restored = ExtractedIntent.model_validate(data)
        assert restored.objective == "Test objective"
        assert restored.missing_variables == ["x"]

    def test_backward_compat_properties(self):
        intent = ExtractedIntent(
            sections={
                "objective": "Do something",
                "context": "Some context",
                "audience": "devs",
            },
        )
        assert intent.objective == "Do something"
        assert intent.context == "Some context"
        assert intent.audience == "devs"

    def test_missing_property_returns_empty(self):
        intent = ExtractedIntent(sections={})
        assert intent.objective == ""
        assert intent.context == ""
        assert intent.audience == ""


# ── Multi-framework extraction ─────────────────────────────────────────────

class TestMultiFrameworkExtraction:
    def test_race_parse_response(self):
        race_def = get_framework(FrameworkName.RACE)
        result = IntentExtractor._parse_response(
            RACE_JSON,
            framework=FrameworkName.RACE.value,
            framework_def=race_def,
        )
        assert result.framework == FrameworkName.RACE.value
        assert result.sections["role"] == "Senior security analyst"
        assert result.sections["action"] == "Review the authentication module for vulnerabilities"
        assert result.sections["expectation"] == "Bulleted list of vulnerabilities with severity ratings"
        assert "codebase_url" in result.missing_variables
        assert result.response_format == "bulleted_list"

    def test_build_extraction_prompt_includes_framework_name(self):
        race_def = get_framework(FrameworkName.RACE)
        prompt = build_extraction_prompt(race_def)
        assert "RACE" in prompt
        assert "role" in prompt
        assert "action" in prompt
        assert "context" in prompt
        assert "expectation" in prompt

    def test_build_extraction_prompt_costar(self):
        costar_def = get_framework(FrameworkName.COSTAR)
        prompt = build_extraction_prompt(costar_def)
        assert "CO-STAR" in prompt
        assert "objective" in prompt
        assert "audience" in prompt

    @pytest.mark.asyncio
    async def test_race_extract_async(self):
        extractor = _make_extractor(framework=FrameworkName.RACE.value)
        extractor.llm.ainvoke.return_value = _mock_llm_response(RACE_JSON)

        result = await extractor.extract("Review auth module for security issues")

        assert result.framework == FrameworkName.RACE.value
        assert result.sections["role"] == "Senior security analyst"
