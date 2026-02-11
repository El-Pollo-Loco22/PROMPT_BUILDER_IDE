"""Unit tests for Phase 2D — SimulationNode with mocked LLM."""

import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.simulator import (
    SimulationNode,
    check_format_compliance,
    check_required_content,
    check_unwanted_content,
    _check_json_format,
    _check_code_format,
    _check_markdown_format,
    _check_list_format,
)
from src.schemas.prompt import PromptSchema, PromptTestResult, ResponseFormat
from src.schemas.frameworks import FrameworkName


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm_response(content: str):
    """Create a mock LLM response object with .content attribute."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_simulator():
    """Create a SimulationNode with mocked LLM (no real Ollama needed)."""
    node = SimulationNode.__new__(SimulationNode)
    node.model = "llama3:8b"
    node.base_url = "http://localhost:11434"
    node.llm = AsyncMock()
    return node


def _make_prompt(**kwargs):
    """Create a minimal CO-STAR PromptSchema for testing."""
    defaults = {
        "context": "Test context for simulation",
        "objective": "Test objective for simulation",
        "audience": "Test audience",
    }
    defaults.update(kwargs)
    return PromptSchema(**defaults)


# ── Format Compliance: JSON ──────────────────────────────────────────────

class TestJsonFormat:
    def test_valid_json(self):
        assert _check_json_format('{"key": "value"}')

    def test_valid_json_array(self):
        assert _check_json_format('[1, 2, 3]')

    def test_fenced_json(self):
        assert _check_json_format('```json\n{"key": "value"}\n```')

    def test_invalid_json(self):
        assert not _check_json_format("This is plain text")

    def test_empty_string(self):
        assert not _check_json_format("")


class TestCodeFormat:
    def test_fenced_code_block(self):
        assert _check_code_format("```python\ndef hello():\n    pass\n```")

    def test_python_def(self):
        assert _check_code_format("def calculate_sum(a, b):\n    return a + b")

    def test_python_class(self):
        assert _check_code_format("class MyClass:\n    pass")

    def test_python_import(self):
        assert _check_code_format("import os\nimport sys")

    def test_javascript_function(self):
        assert _check_code_format("function greet(name) {\n    return name;\n}")

    def test_plain_text_no_code(self):
        assert not _check_code_format("This is just a regular sentence about coding.")


class TestMarkdownFormat:
    def test_headers_and_bold(self):
        assert _check_markdown_format("# Title\n\nSome **bold** text")

    def test_headers_and_list(self):
        assert _check_markdown_format("## Section\n- item 1\n- item 2")

    def test_table_and_code(self):
        assert _check_markdown_format("| Col1 | Col2 |\n```code```")

    def test_plain_text(self):
        assert not _check_markdown_format("Just a simple sentence.")


class TestListFormat:
    def test_bullet_dash(self):
        assert _check_list_format("- item one\n- item two\n- item three")

    def test_bullet_asterisk(self):
        assert _check_list_format("* item one\n* item two")

    def test_numbered_list_dot(self):
        assert _check_list_format("1. First\n2. Second\n3. Third")

    def test_numbered_list_paren(self):
        assert _check_list_format("1) First\n2) Second")

    def test_single_item_not_enough(self):
        assert not _check_list_format("- just one item")

    def test_plain_text(self):
        assert not _check_list_format("No list items here at all.")


# ── check_format_compliance (dispatcher) ─────────────────────────────────

class TestCheckFormatCompliance:
    def test_plain_text_always_passes(self):
        assert check_format_compliance("anything", ResponseFormat.TEXT.value)

    def test_json_format_valid(self):
        assert check_format_compliance('{"a": 1}', ResponseFormat.JSON.value)

    def test_json_format_invalid(self):
        assert not check_format_compliance("not json", ResponseFormat.JSON.value)

    def test_code_format_valid(self):
        assert check_format_compliance("```py\npass\n```", ResponseFormat.CODE.value)

    def test_list_format_valid(self):
        assert check_format_compliance("- a\n- b", ResponseFormat.LIST.value)

    def test_unknown_format_defaults_to_pass(self):
        assert check_format_compliance("anything", "unknown_format")

    def test_empty_text_fails(self):
        assert not check_format_compliance("", ResponseFormat.JSON.value)
        assert not check_format_compliance("   ", ResponseFormat.TEXT.value)


# ── Content Detection ────────────────────────────────────────────────────

class TestRequiredContent:
    def test_all_found(self):
        found, missing = check_required_content(
            "This response includes error handling and type hints.",
            ["error handling", "type hints"],
        )
        assert found == ["error handling", "type hints"]
        assert missing == []

    def test_partial_found(self):
        found, missing = check_required_content(
            "This has error handling but nothing else.",
            ["error handling", "type hints", "docstring"],
        )
        assert found == ["error handling"]
        assert set(missing) == {"type hints", "docstring"}

    def test_none_found(self):
        found, missing = check_required_content(
            "Completely unrelated text.",
            ["error handling", "type hints"],
        )
        assert found == []
        assert missing == ["error handling", "type hints"]

    def test_case_insensitive(self):
        found, _ = check_required_content(
            "This has Error Handling.",
            ["error handling"],
        )
        assert found == ["error handling"]

    def test_empty_requirements(self):
        found, missing = check_required_content("Any text", [])
        assert found == []
        assert missing == []


class TestUnwantedContent:
    def test_no_unwanted_found(self):
        result = check_unwanted_content(
            "Clean response with good content.",
            ["profanity", "placeholder"],
        )
        assert result == []

    def test_unwanted_detected(self):
        result = check_unwanted_content(
            "This contains a TODO placeholder for later.",
            ["placeholder", "hardcoded"],
        )
        assert result == ["placeholder"]

    def test_multiple_unwanted(self):
        result = check_unwanted_content(
            "Contains placeholder and hardcoded values.",
            ["placeholder", "hardcoded"],
        )
        assert result == ["placeholder", "hardcoded"]

    def test_case_insensitive(self):
        result = check_unwanted_content(
            "Contains PLACEHOLDER text.",
            ["placeholder"],
        )
        assert result == ["placeholder"]

    def test_empty_blacklist(self):
        result = check_unwanted_content("Any text", [])
        assert result == []


# ── SimulationNode async simulate() ──────────────────────────────────────

class TestSimulateAsync:
    @pytest.mark.asyncio
    async def test_simulate_returns_test_result(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response(
            "Here is a function:\n```python\ndef hello():\n    pass\n```"
        )

        prompt = _make_prompt()
        result = await node.simulate(prompt, expected_format=ResponseFormat.CODE.value)

        assert isinstance(result, PromptTestResult)
        assert result.model_used == "llama3:8b"
        assert result.token_count > 0
        assert result.execution_time_ms >= 0
        assert "hello" in result.model_response
        node.llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_format_compliance_pass(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response('{"result": "success"}')

        prompt = _make_prompt()
        result = await node.simulate(prompt, expected_format=ResponseFormat.JSON.value)

        assert result.follows_format is True

    @pytest.mark.asyncio
    async def test_simulate_format_compliance_fail(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response("This is plain text, not JSON.")

        prompt = _make_prompt()
        result = await node.simulate(prompt, expected_format=ResponseFormat.JSON.value)

        assert result.follows_format is False

    @pytest.mark.asyncio
    async def test_simulate_required_content_found(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response(
            "Here is error handling with type hints included."
        )

        prompt = _make_prompt(must_include=["error handling", "type hints"])
        result = await node.simulate(prompt)

        assert "error handling" in result.includes_required
        assert "type hints" in result.includes_required
        assert result.missing_required == []

    @pytest.mark.asyncio
    async def test_simulate_required_content_missing(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response(
            "Simple function without much detail."
        )

        prompt = _make_prompt(must_include=["error handling", "docstring"])
        result = await node.simulate(prompt)

        assert "error handling" in result.missing_required
        assert "docstring" in result.missing_required

    @pytest.mark.asyncio
    async def test_simulate_unwanted_content_detected(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response(
            "TODO: implement this placeholder later."
        )

        prompt = _make_prompt(must_not_include=["TODO", "placeholder"])
        result = await node.simulate(prompt)

        assert "TODO" in result.unwanted_content
        assert "placeholder" in result.unwanted_content

    @pytest.mark.asyncio
    async def test_simulate_unwanted_content_clean(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response(
            "Clean, production-ready code here."
        )

        prompt = _make_prompt(must_not_include=["TODO", "placeholder"])
        result = await node.simulate(prompt)

        assert result.unwanted_content == []

    @pytest.mark.asyncio
    async def test_simulate_captures_compiled_prompt(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response("Response text")

        prompt = _make_prompt()
        result = await node.simulate(prompt)

        # prompt_used should be the compiled version
        assert "Context" in result.prompt_used
        assert "Objective" in result.prompt_used

    @pytest.mark.asyncio
    async def test_simulate_default_format_is_text(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response("Any text")

        prompt = _make_prompt()
        result = await node.simulate(prompt)

        # Default format (plain_text) always passes
        assert result.follows_format is True


# ── SimulationNode sync simulate_sync() ──────────────────────────────────

class TestSimulateSync:
    def test_simulate_sync_returns_test_result(self):
        node = _make_simulator()
        node.llm = MagicMock()
        node.llm.invoke.return_value = _mock_llm_response("Sync response text")

        prompt = _make_prompt()
        result = node.simulate_sync(prompt)

        assert isinstance(result, PromptTestResult)
        assert result.model_response == "Sync response text"
        node.llm.invoke.assert_called_once()

    def test_simulate_sync_format_check(self):
        node = _make_simulator()
        node.llm = MagicMock()
        node.llm.invoke.return_value = _mock_llm_response("- item 1\n- item 2\n- item 3")

        prompt = _make_prompt()
        result = node.simulate_sync(prompt, expected_format=ResponseFormat.LIST.value)

        assert result.follows_format is True


# ── Multi-framework simulation ───────────────────────────────────────────

class TestMultiFrameworkSimulation:
    @pytest.mark.asyncio
    async def test_simulate_race_prompt(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response("RACE framework response")

        prompt = PromptSchema(
            framework=FrameworkName.RACE.value,
            sections={
                "role": "Security analyst",
                "action": "Review code for vulnerabilities",
                "context": "Flask web application",
                "expectation": "Bulleted list of findings",
            },
        )
        result = await node.simulate(prompt)

        assert isinstance(result, PromptTestResult)
        # The compiled prompt should contain RACE sections
        assert "Role" in result.prompt_used
        assert "Action" in result.prompt_used

    @pytest.mark.asyncio
    async def test_simulate_ape_prompt(self):
        node = _make_simulator()
        node.llm.ainvoke.return_value = _mock_llm_response("APE response")

        prompt = PromptSchema(
            framework=FrameworkName.APE.value,
            sections={
                "action": "Generate unit tests",
                "purpose": "Ensure code coverage",
                "expectation": "pytest test functions",
            },
        )
        result = await node.simulate(prompt)

        assert isinstance(result, PromptTestResult)
        assert "Action" in result.prompt_used
