"""
Integration Tests — Phases 2A through 2F working in conjunction.

These tests verify that all components interoperate correctly:
- 2A schemas (PromptSchema, QualityScore, PromptTestResult, PromptIteration)
- 2B IntentExtractor output feeds into 2C ArchitectAgent
- 2C ArchitectAgent output feeds into 2D SimulationNode
- 2D SimulationNode output feeds into 2E LinterAgent
- 2E LinterAgent output feeds back into 2C (revision loop)
- 2F LangGraph orchestration wires the full pipeline

All LLM calls are mocked — these test data flow and type compatibility,
not model quality.
"""

import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.schemas.frameworks import FrameworkName, get_framework
from src.schemas.prompt import (
    PromptIteration,
    PromptSchema,
    PromptTestResult,
    QualityScore,
    ResponseFormat,
)
from src.agents.intent_extractor import IntentExtractor, ExtractedIntent, build_extraction_prompt
from src.agents.architect import ArchitectAgent, build_architect_prompt, load_knowledge_base
from src.agents.simulator import (
    SimulationNode,
    check_format_compliance,
    check_required_content,
    check_unwanted_content,
)
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


# ── Mock Data ────────────────────────────────────────────────────────────

def _mock_resp(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


# Realistic CO-STAR pipeline data
INTENT_COSTAR_RAW = json.dumps({
    "objective": "Create a REST API endpoint for user registration with email validation",
    "context": "Python Flask web application for an e-commerce platform with PostgreSQL",
    "audience": "Senior backend developers familiar with Flask and SQLAlchemy",
    "style": "Production-ready code with comprehensive error handling",
    "tone": "Professional and technically precise",
    "response": "Complete Python module with type hints and docstrings",
    "missing_variables": ["database_schema"],
    "constraints": ["Python 3.11+", "No external validation libraries"],
    "response_format": "code_block",
})

ARCHITECT_COSTAR_RAW = json.dumps({
    "context": "Python Flask web application for an e-commerce platform with PostgreSQL database, targeting Python 3.11+",
    "objective": "Create a REST API endpoint for user registration that validates email format, checks for duplicate accounts, and stores hashed passwords",
    "style": "Production-ready code with comprehensive error handling, type hints, and PEP 8 compliance",
    "tone": "Professional and technically precise",
    "audience": "Senior backend developers familiar with Flask and SQLAlchemy ORM patterns",
    "response": "Complete Python module with type annotations, docstrings, and example usage in comments",
})

SIMULATION_RESPONSE = """\
```python
from flask import Flask, request, jsonify
from sqlalchemy import create_engine
import re
import hashlib

app = Flask(__name__)

def validate_email(email: str) -> bool:
    \"\"\"Validate email format using regex.\"\"\"
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

@app.route('/register', methods=['POST'])
def register():
    \"\"\"User registration endpoint with email validation and error handling.\"\"\"
    data = request.get_json()
    email = data.get('email', '')

    if not validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400

    # Hash password and store user
    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    return jsonify({'message': 'User registered successfully'}), 201
```
"""

# RACE pipeline data
INTENT_RACE_RAW = json.dumps({
    "role": "Senior security analyst specializing in web application security",
    "action": "Review the authentication module for SQL injection and XSS vulnerabilities",
    "context": "Flask web application handling OAuth2 login flows for enterprise customers",
    "expectation": "Bulleted list of vulnerabilities with CVSS severity ratings and remediation steps",
    "missing_variables": [],
    "constraints": ["Focus on OWASP Top 10"],
    "response_format": "bulleted_list",
})

ARCHITECT_RACE_RAW = json.dumps({
    "role": "Senior security analyst with expertise in OWASP Top 10 vulnerabilities and web application penetration testing",
    "action": "Perform a comprehensive security review of the authentication module, focusing on SQL injection, XSS, and authentication bypass vulnerabilities",
    "context": "A production Flask web application implementing OAuth2 login flows, serving 500+ enterprise customers with sensitive data",
    "expectation": "A prioritized bulleted list of identified vulnerabilities, each with CVSS v3.1 severity rating, proof-of-concept description, and specific remediation steps",
})

SIMULATION_RACE_RESPONSE = """\
- **SQL Injection in login query** (CVSS 9.8 - Critical)
  - The `username` parameter is concatenated directly into the SQL query
  - Remediation: Use parameterized queries with SQLAlchemy ORM

- **Reflected XSS in error messages** (CVSS 6.1 - Medium)
  - Error messages echo user input without sanitization
  - Remediation: Sanitize all user input before rendering in templates

- **Missing CSRF protection** (CVSS 4.3 - Medium)
  - OAuth2 callback endpoint lacks state parameter validation
  - Remediation: Implement state parameter verification in OAuth2 flow
"""

# Revised architect output after critique
REVISED_ARCHITECT_RAW = json.dumps({
    "context": "Python Flask web application for an e-commerce platform with PostgreSQL database. Must comply with GDPR and handle 10000+ daily registrations securely.",
    "objective": "Create a secure REST API endpoint for user registration that validates email format using regex, checks for duplicate accounts against the database, hashes passwords using bcrypt, and returns appropriate HTTP status codes",
    "style": "Production-ready Python 3.11+ code with comprehensive error handling, type hints, logging, and PEP 8 compliance",
    "tone": "Professional, security-focused, and technically precise",
    "audience": "Senior backend developers familiar with Flask, SQLAlchemy, and web security best practices",
    "response": "Complete Python module with: 1) Type annotations on all functions, 2) Docstrings with Args/Returns sections, 3) Example usage, 4) Unit test stubs",
})


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Full CO-STAR Pipeline (2B → 2C → 2D → 2E)
# ═══════════════════════════════════════════════════════════════════════════

class TestCostarPipelineIntegration:
    """
    Simulates the full CO-STAR data flow:
    IntentExtractor → ArchitectAgent → SimulationNode → LinterAgent
    """

    @pytest.mark.asyncio
    async def test_intent_to_architect(self):
        """2B output (ExtractedIntent) feeds correctly into 2C (ArchitectAgent)."""
        # Step 1: Parse intent (mocked LLM)
        extractor = IntentExtractor.__new__(IntentExtractor)
        extractor.framework = FrameworkName.COSTAR.value
        extractor.framework_def = get_framework(FrameworkName.COSTAR)
        extractor.system_prompt = build_extraction_prompt(extractor.framework_def)
        extractor.llm = AsyncMock()
        extractor.llm.ainvoke.return_value = _mock_resp(INTENT_COSTAR_RAW)

        intent = await extractor.extract("Create a user registration API")

        # Verify intent has framework-compatible structure
        assert isinstance(intent, ExtractedIntent)
        assert "objective" in intent.sections
        assert "context" in intent.sections

        # Step 2: Feed intent into architect (mocked LLM)
        architect = ArchitectAgent.__new__(ArchitectAgent)
        architect.llm = AsyncMock()
        architect.knowledge_base = load_knowledge_base("knowledge-base")
        architect.llm.ainvoke.return_value = _mock_resp(ARCHITECT_COSTAR_RAW)

        prompt = await architect.draft(
            intent=intent.model_dump(),
            framework=FrameworkName.COSTAR.value,
            domain="Software Development",
        )

        # Verify architect produced a valid PromptSchema
        assert isinstance(prompt, PromptSchema)
        assert prompt.framework == FrameworkName.COSTAR.value
        assert prompt.context != ""
        assert prompt.objective != ""

        # Verify it compiles without error
        compiled = prompt.compile_prompt()
        assert "Context" in compiled
        assert "Objective" in compiled
        assert len(compiled) > 100

    @pytest.mark.asyncio
    async def test_architect_to_simulator(self):
        """2C output (PromptSchema) feeds correctly into 2D (SimulationNode)."""
        # Create a prompt via architect parse
        prompt = ArchitectAgent._parse_response(ARCHITECT_COSTAR_RAW, FrameworkName.COSTAR.value)

        # Add constraints to test content detection
        prompt = PromptSchema(
            framework=prompt.framework,
            sections=prompt.sections,
            must_include=["error handling", "type hints"],
            must_not_include=["TODO", "hardcoded"],
        )

        # Simulate (mocked LLM)
        simulator = SimulationNode.__new__(SimulationNode)
        simulator.model = "llama3:8b"
        simulator.base_url = "http://localhost:11434"
        simulator.llm = AsyncMock()
        simulator.llm.ainvoke.return_value = _mock_resp(SIMULATION_RESPONSE)

        result = await simulator.simulate(prompt, expected_format=ResponseFormat.CODE.value)

        # Verify test result is well-formed
        assert isinstance(result, PromptTestResult)
        assert result.follows_format is True  # Has code fences
        assert result.model_used == "llama3:8b"
        assert result.token_count > 0
        assert "validate_email" in result.model_response

        # Content detection worked
        assert "error handling" in result.includes_required
        assert result.unwanted_content == []  # No TODO or hardcoded

    @pytest.mark.asyncio
    async def test_simulator_to_linter(self):
        """2D output (PromptTestResult) feeds correctly into 2E (LinterAgent)."""
        # Build a good prompt
        prompt = PromptSchema(
            context="Python Flask web application for e-commerce with PostgreSQL, targeting Python 3.11+",
            objective="Create a REST API endpoint for user registration with email validation, duplicate checking, and password hashing",
            audience="Senior backend developers familiar with Flask and SQLAlchemy",
            sections={
                "context": "Python Flask web application for e-commerce with PostgreSQL, targeting Python 3.11+",
                "objective": "Create a REST API endpoint for user registration with email validation, duplicate checking, and password hashing",
                "style": "Production-ready code with error handling and type hints",
                "tone": "Professional and technically precise",
                "audience": "Senior backend developers familiar with Flask and SQLAlchemy",
                "response": "Complete Python module with docstrings and example usage",
            },
            constraints=["Python 3.11+", "No external validation libraries"],
            must_include=["error handling", "type hints"],
            must_not_include=["TODO"],
        )

        # Simulate a test result
        test_result = PromptTestResult(
            prompt_used=prompt.compile_prompt(),
            model_response=SIMULATION_RESPONSE,
            token_count=200,
            execution_time_ms=1500,
            model_used="llama3:8b",
            follows_format=True,
            includes_required=["error handling", "type hints"],
            missing_required=[],
            unwanted_content=[],
        )

        # Feed into linter
        linter = LinterAgent()
        score = linter.evaluate(prompt, test_result=test_result)

        # Verify QualityScore is well-formed
        assert isinstance(score, QualityScore)
        assert 1 <= score.overall_score <= 10
        assert score.reasoning != ""
        assert isinstance(score.strengths, list)
        assert isinstance(score.issues, list)

        # Good prompt + good test result should score decently
        assert score.structure_score >= 7
        assert not score.has_injection_risk

        # Simulation feedback should be integrated
        assert any("conforming" in s.lower() for s in score.strengths)

    @pytest.mark.asyncio
    async def test_linter_critique_feeds_back_to_architect(self):
        """2E critique output feeds back into 2C for revision (the reflection loop)."""
        # Start with a mediocre prompt
        mediocre_prompt = PromptSchema(
            context="Some code project with stuff and things etc",
            objective="Help me write something, maybe a function or whatever for validation",
            audience="someone who programs",
        )

        # Lint it
        linter = LinterAgent()
        score = linter.evaluate(mediocre_prompt)

        # Should have issues
        assert len(score.issues) > 0

        # Build critique string (like the graph does)
        critique = "Issues found:\n" + "\n".join(f"- {issue}" for issue in score.issues)
        if score.suggestions:
            critique += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in score.suggestions)

        # Feed critique into architect revision (mocked LLM)
        architect = ArchitectAgent.__new__(ArchitectAgent)
        architect.llm = AsyncMock()
        architect.knowledge_base = {}
        architect.llm.ainvoke.return_value = _mock_resp(REVISED_ARCHITECT_RAW)

        intent = {"sections": mediocre_prompt.sections}
        revised = await architect.draft(
            intent=intent,
            framework=FrameworkName.COSTAR.value,
            critique=critique,
        )

        # Verify the system prompt included revision mode
        call_args = architect.llm.ainvoke.call_args[0][0]
        system_msg = call_args[0].content
        assert "REVISION MODE" in system_msg
        assert "issues" in system_msg.lower() or any(issue[:20] in system_msg for issue in score.issues[:2])

        # Verify revised prompt is valid
        assert isinstance(revised, PromptSchema)
        assert revised.compile_prompt() != mediocre_prompt.compile_prompt()

    def test_full_data_types_chain(self):
        """Verify all data types chain correctly: schema types are compatible across components."""
        # PromptSchema (2A) → compile_prompt() works
        prompt = PromptSchema(
            context="Test context for integration verification",
            objective="Verify data type compatibility across all phases",
            audience="Integration test suite",
            constraints=["must be type-safe"],
            must_include=["validation"],
        )
        compiled = prompt.compile_prompt()
        assert isinstance(compiled, str)
        assert len(compiled) > 50

        # PromptTestResult (2A) → accepts correct types
        test_result = PromptTestResult(
            prompt_used=compiled,
            model_response="Response with validation logic included",
            token_count=50,
            execution_time_ms=100,
            model_used="llama3:8b",
            follows_format=True,
            includes_required=["validation"],
            missing_required=[],
            unwanted_content=[],
        )
        assert isinstance(test_result, PromptTestResult)

        # QualityScore (2A) → LinterAgent produces it
        linter = LinterAgent()
        score = linter.evaluate(prompt, test_result=test_result)
        assert isinstance(score, QualityScore)

        # PromptIteration (2A) → bundles everything
        iteration = PromptIteration(
            iteration_number=1,
            prompt_version=prompt,
            test_result=test_result,
            quality_score=score,
            critique="",
            action_taken="initial_draft",
        )
        assert iteration.prompt_version is prompt
        assert iteration.quality_score is score
        assert iteration.test_result is test_result


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: Full RACE Pipeline (2B → 2C → 2D → 2E)
# ═══════════════════════════════════════════════════════════════════════════

class TestRacePipelineIntegration:
    """Same pipeline but using RACE framework — verifies framework-agnostic design."""

    @pytest.mark.asyncio
    async def test_race_full_pipeline(self):
        """RACE: intent → architect → simulate → linter in sequence."""
        # 2B: Extract intent
        extractor = IntentExtractor.__new__(IntentExtractor)
        extractor.framework = FrameworkName.RACE.value
        extractor.framework_def = get_framework(FrameworkName.RACE)
        extractor.system_prompt = build_extraction_prompt(extractor.framework_def)
        extractor.llm = AsyncMock()
        extractor.llm.ainvoke.return_value = _mock_resp(INTENT_RACE_RAW)

        intent = await extractor.extract("Review auth module for security issues")
        assert intent.framework == FrameworkName.RACE.value
        assert "role" in intent.sections

        # 2C: Architect drafts prompt
        architect = ArchitectAgent.__new__(ArchitectAgent)
        architect.llm = AsyncMock()
        architect.knowledge_base = {}
        architect.llm.ainvoke.return_value = _mock_resp(ARCHITECT_RACE_RAW)

        prompt = await architect.draft(
            intent=intent.model_dump(),
            framework=FrameworkName.RACE.value,
        )
        assert prompt.framework == FrameworkName.RACE.value
        assert "role" in prompt.sections
        assert "action" in prompt.sections

        compiled = prompt.compile_prompt()
        assert "Role" in compiled
        assert "Action" in compiled

        # 2D: Simulate
        simulator = SimulationNode.__new__(SimulationNode)
        simulator.model = "llama3:8b"
        simulator.base_url = "http://localhost:11434"
        simulator.llm = AsyncMock()
        simulator.llm.ainvoke.return_value = _mock_resp(SIMULATION_RACE_RESPONSE)

        # Add content requirements
        prompt_with_reqs = PromptSchema(
            framework=FrameworkName.RACE.value,
            sections=prompt.sections,
            must_include=["SQL injection", "XSS"],
        )
        test_result = await simulator.simulate(
            prompt_with_reqs,
            expected_format=ResponseFormat.LIST.value,
        )
        assert test_result.follows_format is True  # Response is a bulleted list
        assert "SQL Injection" in test_result.model_response
        assert "SQL injection" in test_result.includes_required

        # 2E: Lint
        linter = LinterAgent()
        score = linter.evaluate(prompt_with_reqs, test_result=test_result)
        assert isinstance(score, QualityScore)
        assert score.structure_score >= 7  # RACE sections all present


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Graph Orchestration Integration (2F with mocked agents)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphIntegration:
    """
    Tests the LangGraph orchestration (2F) wiring all components together.
    Mocks at the LLM level so real agent logic executes.
    """

    def test_graph_compiles_and_has_correct_structure(self):
        """Graph builds correctly with all expected nodes and edges."""
        compiled = compile_prompt_graph()
        assert compiled is not None

        graph = build_prompt_graph()
        nodes = set(graph.nodes.keys())
        assert "extract_intent" in nodes
        assert "architect" in nodes
        assert "simulate" in nodes
        assert "linter" in nodes
        assert "finalize" in nodes

    @pytest.mark.asyncio
    async def test_graph_node_sequence_costar(self):
        """Walk through nodes manually to verify state flows correctly."""
        # Step 1: extract_intent
        with patch("src.graph.builder.IntentExtractor") as MockExtractor:
            mock_ext = MockExtractor.return_value
            mock_intent = MagicMock()
            mock_intent.model_dump.return_value = {
                "framework": "co_star",
                "sections": {
                    "objective": "Create registration API",
                    "context": "Flask web app",
                    "audience": "Senior devs",
                },
                "missing_variables": [],
                "constraints": [],
                "response_format": "code_block",
            }
            mock_ext.extract = AsyncMock(return_value=mock_intent)

            state = {
                "user_input": "Create a registration API",
                "framework": FrameworkName.COSTAR.value,
            }
            state = await extract_intent_node(state)

        assert state["status"] == "intent_extracted"
        assert "sections" in state["intent"]

        # Step 2: architect
        with patch("src.graph.builder.ArchitectAgent") as MockArchitect:
            mock_arch = MockArchitect.return_value
            mock_prompt = PromptSchema(
                context="Flask web application for e-commerce platform",
                objective="Create a user registration endpoint with email validation",
                audience="Senior Python developers",
            )
            mock_arch.draft = AsyncMock(return_value=mock_prompt)

            state = await architect_node(state)

        assert state["status"] == "prompt_drafted"
        assert isinstance(state["current_prompt"], PromptSchema)

        # Step 3: simulate
        with patch("src.graph.builder.SimulationNode") as MockSim:
            mock_sim = MockSim.return_value
            mock_result = PromptTestResult(
                prompt_used="compiled",
                model_response="Code with error handling and type hints",
                token_count=100,
                execution_time_ms=500,
                model_used="llama3:8b",
                follows_format=True,
                includes_required=[],
                missing_required=[],
                unwanted_content=[],
            )
            mock_sim.simulate = AsyncMock(return_value=mock_result)

            state = await simulate_node(state)

        assert state["status"] == "simulated"
        assert isinstance(state["test_result"], PromptTestResult)

        # Step 4: linter (uses real LinterAgent — no LLM needed)
        state["iterations_history"] = []
        state["critique"] = ""
        state = await linter_node(state)

        assert state["status"] == "evaluated"
        assert isinstance(state["quality_score"], QualityScore)
        assert len(state["iterations_history"]) == 1

        # Step 5: routing decision
        decision = should_continue(state)
        assert decision in ("finalize", "revise")

        # Step 6: finalize
        state = await finalize_node(state)
        assert state["status"] == "complete"
        assert state["final_prompt"] is not None
        assert state["final_score"] is not None

    @pytest.mark.asyncio
    async def test_revision_loop_flow(self):
        """Verify that revision correctly passes critique back to architect."""
        # Start with a poor-quality prompt
        poor_prompt = PromptSchema(
            context="Something vague about maybe doing stuff or things etc",
            objective="Help me write something, sort of a function for whatever",
            audience="some person who codes things",
        )

        # Linter evaluates and builds critique
        state = {
            "current_prompt": poor_prompt,
            "test_result": None,
            "iteration": 1,
            "iterations_history": [],
            "critique": "",
        }
        state = await linter_node(state)

        # Should want revision
        if not state["quality_score"].passes_threshold():
            assert state["critique"] is not None
            assert "Issues found" in state["critique"]

            # Now architect receives the critique
            with patch("src.graph.builder.ArchitectAgent") as MockArchitect:
                mock_arch = MockArchitect.return_value
                revised_prompt = PromptSchema(
                    context="A Python 3.11 Flask web application for e-commerce handling 10000 daily registrations",
                    objective="Create a secure REST API endpoint for user registration with email validation and bcrypt password hashing",
                    audience="Senior backend developers familiar with Flask and SQLAlchemy",
                )
                mock_arch.draft = AsyncMock(return_value=revised_prompt)

                state["intent"] = {"sections": poor_prompt.sections}
                state = await architect_node(state)

                # Verify critique was passed through
                call_kwargs = mock_arch.draft.call_args
                assert call_kwargs.kwargs.get("critique") is not None

            # Verify state evolved
            assert state["current_prompt"] is revised_prompt
            assert state["iteration"] == 2

    def test_iteration_cap_prevents_infinite_loop(self):
        """Max iterations forces finalize even with low score."""
        poor_score = QualityScore(
            overall_score=3,
            clarity_score=3,
            specificity_score=3,
            structure_score=3,
            constraint_score=3,
            token_efficiency_score=3,
            strengths=[],
            issues=["Everything is vague"],
            suggestions=["Be specific"],
            reasoning="Low score across the board.",
        )

        # Iteration 1, 2, 3 — should still revise
        for i in [1, 2, 3]:
            state = {
                "quality_score": poor_score,
                "iteration": i,
                "max_iterations": 3,
            }
            assert should_continue(state) == "revise"

        # Iteration 4 — must finalize
        state = {
            "quality_score": poor_score,
            "iteration": 4,
            "max_iterations": 3,
        }
        assert should_continue(state) == "finalize"


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Cross-Framework Compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossFrameworkCompatibility:
    """Verify that all 4 frameworks work through the full component chain."""

    @pytest.mark.parametrize("framework,sections", [
        (
            FrameworkName.COSTAR.value,
            {
                "context": "Python web application for managing user accounts securely",
                "objective": "Create a password reset endpoint with email verification",
                "style": "Clean, well-documented code",
                "tone": "Professional",
                "audience": "Mid-level Python developers",
                "response": "Complete implementation with tests",
            },
        ),
        (
            FrameworkName.RACE.value,
            {
                "role": "Senior database architect with PostgreSQL expertise",
                "action": "Design a database schema for a multi-tenant SaaS application",
                "context": "Startup building a project management tool for 100+ organizations",
                "expectation": "ERD diagram description with table definitions and indexes",
            },
        ),
        (
            FrameworkName.APE.value,
            {
                "action": "Generate comprehensive unit tests for the authentication module",
                "purpose": "Ensure 90% code coverage and catch edge cases before release",
                "expectation": "pytest test functions covering happy path, error cases, and boundary conditions",
            },
        ),
        (
            FrameworkName.CRISPE.value,
            {
                "context": "Healthcare data platform processing patient records under HIPAA",
                "role": "Senior compliance engineer with healthcare IT experience",
                "instruction": "Audit the data pipeline for HIPAA compliance violations",
                "schema": "Structured report with violation categories, risk levels, and remediation timelines",
                "persona": "Thorough, regulation-aware, detail-oriented",
                "crispe_examples": "Example: PHI exposed in logs → HIGH risk → Mask all PII in log output within 2 weeks",
            },
        ),
    ])
    def test_framework_through_linter(self, framework, sections):
        """Each framework creates a valid prompt that the linter can evaluate."""
        prompt = PromptSchema(framework=framework, sections=sections)

        # Compiles
        compiled = prompt.compile_prompt()
        assert len(compiled) > 50

        # Linter evaluates without error
        linter = LinterAgent()
        score = linter.evaluate(prompt)

        assert isinstance(score, QualityScore)
        assert 1 <= score.overall_score <= 10
        assert score.reasoning != ""

        # Structure score should be decent (all required sections filled)
        fw_def = get_framework(framework)
        missing = fw_def.required_keys - set(sections.keys())
        if not missing:
            assert score.structure_score >= 7

    @pytest.mark.parametrize("framework,sections", [
        (
            FrameworkName.COSTAR.value,
            {
                "context": "Python web application for managing user accounts",
                "objective": "Create a password reset endpoint with email verification",
                "audience": "Mid-level Python developers",
            },
        ),
        (
            FrameworkName.RACE.value,
            {
                "role": "Senior database architect with PostgreSQL expertise",
                "action": "Design a normalized database schema for multi-tenant SaaS",
                "context": "A startup building a project management tool for organizations",
                "expectation": "Table definitions with indexes and foreign key relationships",
            },
        ),
    ])
    def test_framework_format_compliance(self, framework, sections):
        """Compiled prompts from any framework work with format compliance checking."""
        prompt = PromptSchema(framework=framework, sections=sections)
        compiled = prompt.compile_prompt()

        # The compiled prompt itself is plain text
        assert check_format_compliance(compiled, ResponseFormat.TEXT.value) is True

    @pytest.mark.parametrize("framework", [
        FrameworkName.COSTAR.value,
        FrameworkName.RACE.value,
        FrameworkName.APE.value,
        FrameworkName.CRISPE.value,
    ])
    def test_framework_graph_state_compatible(self, framework):
        """PromptBuilderState accepts any framework."""
        state: PromptBuilderState = {
            "user_input": "Test input",
            "framework": framework,
            "iteration": 1,
            "max_iterations": 3,
        }
        assert state["framework"] == framework


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Knowledge Base Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseIntegration:
    """Verify knowledge base loads and integrates with architect and linter."""

    def test_knowledge_base_loads_and_provides_practices(self):
        """Knowledge base files load correctly and feed into architect prompts."""
        kb = load_knowledge_base("knowledge-base")
        assert "Software Development" in kb
        assert "General" in kb

        # Build architect prompt with knowledge base
        fw = get_framework(FrameworkName.COSTAR)
        from src.agents.architect import _format_best_practices
        practices = _format_best_practices(kb, "Software Development")
        assert "programming language" in practices.lower()

        prompt = build_architect_prompt(fw, best_practices=practices)
        assert "best practices" in prompt.lower()
        assert "programming language" in prompt.lower()
