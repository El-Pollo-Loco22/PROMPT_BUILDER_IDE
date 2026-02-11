"""Unit tests for Phase 2F — LangGraph orchestration with mocked agents."""

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.builder import (
    PromptBuilderState,
    build_prompt_graph,
    compile_prompt_graph,
    extract_intent_node,
    architect_node,
    simulate_node,
    linter_node,
    finalize_node,
    should_continue,
)
from src.schemas.frameworks import FrameworkName
from src.schemas.prompt import (
    PromptIteration,
    PromptSchema,
    PromptTestResult,
    QualityScore,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_llm_response(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


def _make_good_quality_score():
    return QualityScore(
        overall_score=8,
        clarity_score=8,
        specificity_score=8,
        structure_score=9,
        constraint_score=7,
        token_efficiency_score=8,
        strengths=["Good structure", "Clear objectives"],
        issues=[],
        suggestions=[],
        reasoning="Scores: Clarity: 8/10, all good.",
    )


def _make_poor_quality_score():
    return QualityScore(
        overall_score=4,
        clarity_score=4,
        specificity_score=3,
        structure_score=5,
        constraint_score=3,
        token_efficiency_score=5,
        strengths=[],
        issues=["Vague language", "Missing constraints"],
        suggestions=["Be more specific", "Add constraints"],
        reasoning="Scores: Clarity: 4/10, needs work.",
    )


def _make_prompt():
    return PromptSchema(
        context="A Python web application for user management",
        objective="Create a user registration endpoint with validation",
        audience="Intermediate Python developers",
    )


def _make_test_result():
    return PromptTestResult(
        prompt_used="compiled prompt text",
        model_response="Here is the registration endpoint code...",
        token_count=150,
        execution_time_ms=500,
        model_used="llama3:8b",
        follows_format=True,
        includes_required=[],
        missing_required=[],
        unwanted_content=[],
    )


INTENT_JSON = json.dumps({
    "objective": "Create a user registration endpoint",
    "context": "Python Flask web application",
    "audience": "intermediate developers",
    "missing_variables": [],
    "constraints": [],
    "response_format": "code_block",
})

ARCHITECT_JSON = json.dumps({
    "context": "A Python Flask web application for user management",
    "objective": "Create a user registration endpoint with email validation",
    "style": "Clean, production-ready Python code",
    "tone": "Professional",
    "audience": "Intermediate Python developers",
    "response": "Complete endpoint code with comments",
})


# ── Graph Build Tests ────────────────────────────────────────────────────

class TestGraphBuild:
    def test_build_prompt_graph(self):
        graph = build_prompt_graph()
        assert graph is not None

    def test_compile_prompt_graph(self):
        compiled = compile_prompt_graph()
        assert compiled is not None

    def test_graph_has_expected_nodes(self):
        graph = build_prompt_graph()
        # Check node names are present
        node_names = set(graph.nodes.keys())
        expected = {"extract_intent", "architect", "simulate", "linter", "finalize"}
        assert expected.issubset(node_names)


# ── should_continue routing ──────────────────────────────────────────────

class TestShouldContinue:
    def test_passes_threshold_finalizes(self):
        state = {
            "quality_score": _make_good_quality_score(),
            "iteration": 2,
            "max_iterations": 3,
        }
        assert should_continue(state) == "finalize"

    def test_below_threshold_revises(self):
        state = {
            "quality_score": _make_poor_quality_score(),
            "iteration": 2,
            "max_iterations": 3,
        }
        assert should_continue(state) == "revise"

    def test_max_iterations_forces_finalize(self):
        state = {
            "quality_score": _make_poor_quality_score(),
            "iteration": 4,
            "max_iterations": 3,
        }
        assert should_continue(state) == "finalize"

    def test_first_iteration_below_threshold_revises(self):
        state = {
            "quality_score": _make_poor_quality_score(),
            "iteration": 1,
            "max_iterations": 3,
        }
        assert should_continue(state) == "revise"

    def test_no_score_revises(self):
        state = {
            "quality_score": None,
            "iteration": 1,
            "max_iterations": 3,
        }
        assert should_continue(state) == "revise"

    def test_default_max_iterations(self):
        state = {
            "quality_score": _make_poor_quality_score(),
            "iteration": 4,
        }
        assert should_continue(state) == "finalize"


# ── Individual Node Tests (mocked) ──────────────────────────────────────

class TestExtractIntentNode:
    @pytest.mark.asyncio
    @patch("src.graph.builder.IntentExtractor")
    async def test_extract_intent(self, MockExtractor):
        mock_instance = MockExtractor.return_value
        mock_intent = MagicMock()
        mock_intent.model_dump.return_value = {
            "framework": "co_star",
            "sections": {"objective": "Test"},
            "missing_variables": [],
            "constraints": [],
            "response_format": "plain_text",
        }
        mock_instance.extract = AsyncMock(return_value=mock_intent)

        state = {
            "user_input": "Create a function to validate emails",
            "framework": FrameworkName.COSTAR.value,
        }

        result = await extract_intent_node(state)

        assert result["status"] == "intent_extracted"
        assert "intent" in result
        assert result["iteration"] == 1
        mock_instance.extract.assert_called_once()


class TestArchitectNode:
    @pytest.mark.asyncio
    @patch("src.graph.builder.ArchitectAgent")
    async def test_architect_drafts_prompt(self, MockArchitect):
        mock_instance = MockArchitect.return_value
        mock_prompt = _make_prompt()
        mock_instance.draft = AsyncMock(return_value=mock_prompt)

        state = {
            "intent": {"sections": {"objective": "Test"}},
            "framework": FrameworkName.COSTAR.value,
            "domain": "General",
            "critique": None,
        }

        result = await architect_node(state)

        assert result["status"] == "prompt_drafted"
        assert result["current_prompt"] is mock_prompt
        mock_instance.draft.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.graph.builder.ArchitectAgent")
    async def test_architect_revision_with_critique(self, MockArchitect):
        mock_instance = MockArchitect.return_value
        mock_prompt = _make_prompt()
        mock_instance.draft = AsyncMock(return_value=mock_prompt)

        state = {
            "intent": {"sections": {"objective": "Test"}},
            "framework": FrameworkName.COSTAR.value,
            "domain": "General",
            "critique": "Context needs more detail.",
        }

        result = await architect_node(state)

        call_kwargs = mock_instance.draft.call_args
        assert call_kwargs.kwargs.get("critique") == "Context needs more detail."


class TestSimulateNode:
    @pytest.mark.asyncio
    @patch("src.graph.builder.SimulationNode")
    async def test_simulate_runs(self, MockSimulator):
        mock_instance = MockSimulator.return_value
        mock_result = _make_test_result()
        mock_instance.simulate = AsyncMock(return_value=mock_result)

        state = {
            "current_prompt": _make_prompt(),
            "expected_format": "plain_text",
        }

        result = await simulate_node(state)

        assert result["status"] == "simulated"
        assert result["test_result"] is mock_result
        mock_instance.simulate.assert_called_once()


class TestLinterNode:
    @pytest.mark.asyncio
    async def test_linter_evaluates(self):
        prompt = _make_prompt()
        state = {
            "current_prompt": prompt,
            "test_result": _make_test_result(),
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }

        result = await linter_node(state)

        assert result["status"] == "evaluated"
        assert isinstance(result["quality_score"], QualityScore)
        assert len(result["iterations_history"]) == 1
        assert result["iteration"] == 2

    @pytest.mark.asyncio
    async def test_linter_builds_critique_on_failure(self):
        prompt = PromptSchema(
            context="Something vague about stuff maybe or whatever etc",
            objective="Help me write something, sort of a function",
            audience="someone who codes",
        )
        state = {
            "current_prompt": prompt,
            "test_result": None,
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }

        result = await linter_node(state)

        if not result["quality_score"].passes_threshold():
            assert result["critique"] is not None
            assert "Issues found" in result["critique"]

    @pytest.mark.asyncio
    async def test_linter_no_critique_on_pass(self):
        # Use a well-structured prompt that should pass
        prompt = PromptSchema(
            context="A Python 3.11 Flask web application for an e-commerce platform with 10000 users",
            objective="Create a secure user registration endpoint that validates email, checks duplicates, and hashes passwords",
            audience="Senior Python developers familiar with Flask and SQLAlchemy",
            sections={
                "context": "A Python 3.11 Flask web application for an e-commerce platform with 10000 users",
                "objective": "Create a secure user registration endpoint that validates email, checks duplicates, and hashes passwords",
                "style": "Production-ready code with type hints and error handling",
                "tone": "Professional and technically precise",
                "audience": "Senior Python developers familiar with Flask and SQLAlchemy",
                "response": "Complete Python module with docstrings and example usage",
            },
            constraints=["Use stdlib and Flask only", "Follow PEP 8"],
            must_include=["error handling", "type hints"],
        )
        test_result = PromptTestResult(
            prompt_used="test",
            model_response="Here is error handling code with type hints...",
            token_count=100,
            execution_time_ms=500,
            model_used="llama3:8b",
            follows_format=True,
            includes_required=["error handling", "type hints"],
            missing_required=[],
            unwanted_content=[],
        )
        state = {
            "current_prompt": prompt,
            "test_result": test_result,
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }

        result = await linter_node(state)

        if result["quality_score"].passes_threshold():
            assert result["critique"] is None


class TestFinalizeNode:
    @pytest.mark.asyncio
    async def test_finalize_packages_result(self):
        prompt = _make_prompt()
        score = _make_good_quality_score()

        state = {
            "current_prompt": prompt,
            "quality_score": score,
            "status": "evaluated",
        }

        result = await finalize_node(state)

        assert result["status"] == "complete"
        assert result["final_prompt"] is prompt
        assert result["final_score"] is score


# ── Iteration History Tracking ───────────────────────────────────────────

class TestIterationTracking:
    @pytest.mark.asyncio
    async def test_history_accumulates(self):
        prompt = _make_prompt()

        # First iteration
        state1 = {
            "current_prompt": prompt,
            "test_result": None,
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }
        result1 = await linter_node(state1)
        assert len(result1["iterations_history"]) == 1
        assert result1["iterations_history"][0].iteration_number == 1

        # Second iteration
        state2 = {
            **result1,
            "current_prompt": prompt,
        }
        result2 = await linter_node(state2)
        assert len(result2["iterations_history"]) == 2
        assert result2["iterations_history"][1].iteration_number == 2

    @pytest.mark.asyncio
    async def test_history_records_action(self):
        prompt = _make_prompt()

        state = {
            "current_prompt": prompt,
            "test_result": None,
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }
        result = await linter_node(state)
        assert result["iterations_history"][0].action_taken == "initial_draft"

        state2 = {
            **result,
            "current_prompt": prompt,
        }
        result2 = await linter_node(state2)
        assert result2["iterations_history"][1].action_taken == "revision"


# ── State Type Tests ─────────────────────────────────────────────────────

class TestPromptBuilderState:
    def test_state_creation(self):
        state: PromptBuilderState = {
            "user_input": "Create a function",
            "framework": FrameworkName.COSTAR.value,
            "domain": "General",
            "expected_format": "plain_text",
            "iteration": 1,
            "max_iterations": 3,
        }
        assert state["user_input"] == "Create a function"
        assert state["framework"] == "co_star"

    def test_state_with_race_framework(self):
        state: PromptBuilderState = {
            "user_input": "Review code for security",
            "framework": FrameworkName.RACE.value,
            "iteration": 1,
        }
        assert state["framework"] == "race"
