"""
End-to-End Integration Tests — Live Ollama

These tests run the FULL pipeline with NO mocks:
  user input → IntentExtractor → ArchitectAgent → SimulationNode → LinterAgent
  → conditional routing (revise or finalize) → final output

They require a running Ollama instance with the llama3:8b model pulled.

Usage:
    # Run only e2e tests:
    pytest tests/test_e2e_live.py -v

    # Skip e2e tests during normal development:
    pytest -m "not e2e"

    # Run everything including e2e:
    pytest -v
"""

import sys
import os
import asyncio

import pytest
import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import settings
from src.schemas.frameworks import FrameworkName, get_framework
from src.schemas.prompt import (
    PromptIteration,
    PromptSchema,
    PromptTestResult,
    QualityScore,
    ResponseFormat,
)
from src.agents.intent_extractor import IntentExtractor, ExtractedIntent
from src.agents.architect import ArchitectAgent
from src.agents.simulator import SimulationNode
from src.agents.linter import LinterAgent
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ollama_is_available() -> bool:
    """Check if Ollama is reachable and has the required model."""
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        return settings.default_model in models
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# Skip the entire module if Ollama is not available
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not ollama_is_available(),
        reason=f"Ollama not available at {settings.ollama_base_url} "
               f"or model '{settings.default_model}' not pulled",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Individual Live Agent Calls
# ═══════════════════════════════════════════════════════════════════════════
# These test each agent in isolation with real Ollama calls, so if the
# full pipeline fails you can see exactly WHICH agent broke.

class TestLiveAgents:
    """Test each agent individually against live Ollama."""

    @pytest.mark.asyncio
    async def test_intent_extractor_live(self):
        """
        IntentExtractor sends the user's raw text to Ollama and gets back
        structured JSON with CO-STAR sections filled in.

        This proves: Ollama can understand our system prompt and return
        parseable JSON with the right keys.
        """
        extractor = IntentExtractor(framework=FrameworkName.COSTAR.value)
        intent = await extractor.extract(
            "Write a Python function that validates email addresses "
            "and returns detailed error messages for invalid formats"
        )

        # The LLM should return a valid ExtractedIntent
        assert isinstance(intent, ExtractedIntent)
        assert intent.framework == FrameworkName.COSTAR.value

        # It should have filled in at least the core CO-STAR sections
        assert intent.sections, "LLM returned empty sections"
        assert "objective" in intent.sections, f"Missing 'objective', got: {list(intent.sections.keys())}"

        # The objective should relate to what we asked
        obj = intent.sections["objective"].lower()
        assert "email" in obj or "valid" in obj, (
            f"Objective doesn't seem related to email validation: {obj[:100]}"
        )

    @pytest.mark.asyncio
    async def test_architect_live(self):
        """
        ArchitectAgent takes structured intent and builds a full prompt
        with all CO-STAR sections populated by the LLM.

        This proves: Ollama can take partial intent and flesh it out into
        a complete, compilable PromptSchema.
        """
        intent = {
            "sections": {
                "objective": "Create a Python function for email validation",
                "context": "Backend utility module for a Flask web app",
                "audience": "Junior Python developers",
            },
            "constraints": ["Python 3.10+", "No third-party libraries"],
        }

        architect = ArchitectAgent()
        prompt = await architect.draft(
            intent=intent,
            framework=FrameworkName.COSTAR.value,
            domain="Software Development",
        )

        assert isinstance(prompt, PromptSchema)
        assert prompt.framework == FrameworkName.COSTAR.value

        # Should have meaningful content in sections
        assert len(prompt.sections) >= 3, f"Only {len(prompt.sections)} sections filled"

        # Should compile without error
        compiled = prompt.compile_prompt()
        assert len(compiled) > 50, "Compiled prompt is suspiciously short"

    @pytest.mark.asyncio
    async def test_simulator_live(self):
        """
        SimulationNode takes a compiled prompt, sends it to Ollama, and
        captures the response along with metrics (token count, time, format).

        This proves: Ollama can respond to our compiled prompts and the
        metrics capture works end-to-end.
        """
        prompt = PromptSchema(
            framework=FrameworkName.COSTAR.value,
            sections={
                "context": "Python utility module",
                "objective": "Write a function that checks if a string is a palindrome",
                "style": "Clean, readable code with type hints",
                "tone": "Educational",
                "audience": "Beginner Python learners",
                "response": "A single Python function with a docstring",
            },
        )

        simulator = SimulationNode()
        result = await simulator.simulate(
            prompt_schema=prompt,
            expected_format=ResponseFormat.CODE.value,
        )

        assert isinstance(result, PromptTestResult)
        assert result.model_used == settings.default_model
        assert result.token_count > 0
        assert result.execution_time_ms > 0
        assert len(result.model_response) > 20, "Model response is too short"

        # The response should contain something code-related
        resp_lower = result.model_response.lower()
        assert "def " in resp_lower or "palindrome" in resp_lower, (
            f"Response doesn't look like code: {result.model_response[:200]}"
        )

    def test_linter_on_live_quality(self):
        """
        LinterAgent is heuristic-based (no LLM), but here we verify it
        produces sensible scores on realistic prompt content — the kind
        of prompt the live architect would actually produce.

        This proves: The scoring logic works on real-world prompt quality.
        """
        # A well-structured prompt (what a good architect call would produce)
        good_prompt = PromptSchema(
            framework=FrameworkName.COSTAR.value,
            sections={
                "context": "A Python 3.11 Flask web application serving an e-commerce platform with PostgreSQL database and Redis caching layer",
                "objective": "Create a secure REST API endpoint for user registration that validates email format, checks for duplicate accounts, hashes passwords with bcrypt, and returns appropriate HTTP status codes",
                "style": "Production-ready code with comprehensive error handling, type hints, and PEP 8 compliance",
                "tone": "Professional and technically precise",
                "audience": "Senior backend developers familiar with Flask, SQLAlchemy, and web security best practices",
                "response": "Complete Python module with type annotations, docstrings, and example usage comments",
            },
        )

        linter = LinterAgent()
        score = linter.evaluate(good_prompt)

        assert isinstance(score, QualityScore)
        assert score.overall_score >= 6, (
            f"Well-structured prompt scored too low: {score.overall_score}/10. "
            f"Issues: {score.issues}"
        )
        assert score.structure_score >= 8, "All CO-STAR sections are present"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Chained Pipeline (No Graph, Sequential Agent Calls)
# ═══════════════════════════════════════════════════════════════════════════
# Calls each agent in sequence, passing real outputs forward.
# This isolates data-flow issues from graph-routing issues.

class TestLivePipelineChained:
    """
    Run the full agent chain manually: intent → architect → simulate → lint.
    No LangGraph involved — just sequential calls to prove data flows.
    """

    @pytest.mark.asyncio
    async def test_costar_pipeline_live(self):
        """
        Full CO-STAR pipeline with live Ollama:

        1. IntentExtractor parses "build me a REST API" into CO-STAR sections
        2. ArchitectAgent expands that into a full PromptSchema
        3. SimulationNode sends the compiled prompt to Ollama and gets a response
        4. LinterAgent scores the prompt quality

        Each step uses the REAL output from the previous step.
        """
        # ── Step 1: Extract Intent ──
        extractor = IntentExtractor(framework=FrameworkName.COSTAR.value)
        intent = await extractor.extract(
            "Build a REST API endpoint in Python Flask for user login "
            "with JWT token authentication and rate limiting"
        )

        assert isinstance(intent, ExtractedIntent)
        assert len(intent.sections) >= 2, (
            f"Intent extraction returned too few sections: {intent.sections}"
        )

        # ── Step 2: Architect Drafts Prompt ──
        architect = ArchitectAgent()
        prompt = await architect.draft(
            intent=intent.model_dump(),
            framework=FrameworkName.COSTAR.value,
            domain="Software Development",
        )

        assert isinstance(prompt, PromptSchema)
        compiled = prompt.compile_prompt()
        assert len(compiled) > 100, f"Compiled prompt too short: {len(compiled)} chars"

        # ── Step 3: Simulate ──
        simulator = SimulationNode()
        test_result = await simulator.simulate(
            prompt_schema=prompt,
            expected_format=ResponseFormat.CODE.value,
        )

        assert isinstance(test_result, PromptTestResult)
        assert test_result.token_count > 0
        assert test_result.execution_time_ms > 0
        assert len(test_result.model_response) > 50, (
            f"Simulation response too short: {test_result.model_response[:100]}"
        )

        # ── Step 4: Lint ──
        linter = LinterAgent()
        score = linter.evaluate(prompt, test_result=test_result)

        assert isinstance(score, QualityScore)
        assert 1 <= score.overall_score <= 10
        assert score.reasoning, "Linter should provide reasoning"

        # Record results for visibility
        print(f"\n{'='*60}")
        print(f"  E2E CO-STAR Pipeline Results")
        print(f"{'='*60}")
        print(f"  Intent sections:  {list(intent.sections.keys())}")
        print(f"  Prompt length:    {len(compiled)} chars")
        print(f"  Sim response:     {len(test_result.model_response)} chars")
        print(f"  Sim time:         {test_result.execution_time_ms}ms")
        print(f"  Format OK:        {test_result.follows_format}")
        print(f"  Quality score:    {score.overall_score}/10")
        print(f"  Structure:        {score.structure_score}/10")
        print(f"  Clarity:          {score.clarity_score}/10")
        print(f"  Strengths:        {score.strengths[:2]}")
        print(f"  Issues:           {score.issues[:2]}")
        print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Full LangGraph Execution (The Real Deal)
# ═══════════════════════════════════════════════════════════════════════════
# This is the ultimate test: compile the LangGraph and invoke it.
# The graph handles routing, iteration, and finalization automatically.

class TestLiveGraphExecution:
    """
    Run the compiled LangGraph with live Ollama.
    This is the closest thing to what the real product does.
    """

    @pytest.mark.asyncio
    async def test_full_graph_costar(self):
        """
        Compile the LangGraph and run it end-to-end.

        The graph will:
        1. extract_intent — parse user input via Ollama
        2. architect — draft a CO-STAR prompt via Ollama
        3. simulate — run the prompt against Ollama
        4. linter — score it (heuristic)
        5. Route: if score >= 7 → finalize, else → back to architect (max 3 loops)
        6. finalize — package the result

        We don't control how many iterations happen — the graph decides
        based on the actual quality score from the linter.
        """
        graph = compile_prompt_graph()

        initial_state: PromptBuilderState = {
            "user_input": (
                "Write a Python function that takes a list of dictionaries "
                "and groups them by a specified key, handling missing keys gracefully"
            ),
            "framework": FrameworkName.COSTAR.value,
            "domain": "Software Development",
            "expected_format": ResponseFormat.CODE.value,
            "max_iterations": 3,
        }

        # Run the graph — this will make multiple real Ollama calls
        result = await graph.ainvoke(initial_state)

        # ── Verify final state ──

        # The graph should have completed
        assert result["status"] == "complete", (
            f"Graph did not reach 'complete' status: {result['status']}"
        )

        # Should have a final prompt
        assert result["final_prompt"] is not None, "No final prompt produced"
        assert isinstance(result["final_prompt"], PromptSchema)

        final_compiled = result["final_prompt"].compile_prompt()
        assert len(final_compiled) > 50, "Final prompt is too short"

        # Should have a final score
        assert result["final_score"] is not None, "No final score produced"
        assert isinstance(result["final_score"], QualityScore)
        assert 1 <= result["final_score"].overall_score <= 10

        # Should have iteration history
        history = result.get("iterations_history", [])
        assert len(history) >= 1, "Should have at least 1 iteration recorded"
        assert len(history) <= 3, f"Should not exceed max_iterations, got {len(history)}"

        # Each iteration should be well-formed
        for i, entry in enumerate(history):
            assert isinstance(entry, PromptIteration), f"Iteration {i} is not a PromptIteration"
            assert entry.iteration_number == i + 1
            assert entry.prompt_version is not None
            assert entry.quality_score is not None

        # Print results for visibility
        print(f"\n{'='*60}")
        print(f"  Full Graph Execution Results")
        print(f"{'='*60}")
        print(f"  Iterations:       {len(history)}")
        print(f"  Final score:      {result['final_score'].overall_score}/10")
        print(f"  Structure:        {result['final_score'].structure_score}/10")
        print(f"  Clarity:          {result['final_score'].clarity_score}/10")
        print(f"  Prompt length:    {len(final_compiled)} chars")
        if history:
            scores = [h.quality_score.overall_score for h in history]
            print(f"  Score progression: {' → '.join(str(s) for s in scores)}")
        print(f"  Strengths:        {result['final_score'].strengths[:2]}")
        print(f"  Issues:           {result['final_score'].issues[:2]}")
        print(f"{'='*60}\n")

    @pytest.mark.asyncio
    async def test_full_graph_race(self):
        """
        Same test but with the RACE framework — proves the graph is
        truly framework-agnostic end-to-end, not just with mocked data.
        """
        graph = compile_prompt_graph()

        initial_state: PromptBuilderState = {
            "user_input": (
                "Review a Node.js Express API for security vulnerabilities, "
                "focusing on authentication bypass and injection attacks"
            ),
            "framework": FrameworkName.RACE.value,
            "domain": "Software Development",
            "expected_format": ResponseFormat.LIST.value,
            "max_iterations": 2,  # Keep it shorter for RACE
        }

        result = await graph.ainvoke(initial_state)

        assert result["status"] == "complete"
        assert result["final_prompt"] is not None
        assert result["final_prompt"].framework == FrameworkName.RACE.value
        assert result["final_score"] is not None

        compiled = result["final_prompt"].compile_prompt()
        # RACE uses Role/Action/Context/Expectation headers
        assert "Role" in compiled or "Action" in compiled, (
            f"RACE prompt missing expected headers: {compiled[:200]}"
        )

        history = result.get("iterations_history", [])
        assert len(history) >= 1

        print(f"\n{'='*60}")
        print(f"  RACE Graph Execution Results")
        print(f"{'='*60}")
        print(f"  Iterations:    {len(history)}")
        print(f"  Final score:   {result['final_score'].overall_score}/10")
        print(f"  Framework:     {result['final_prompt'].framework}")
        print(f"{'='*60}\n")
