"""
LangGraph Orchestration (Phase 2F)

Reflection loop: extract_intent -> architect -> simulate -> linter
-> conditional re-draft (score >= 7 passes, else revise with critique,
max 3 iterations).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.agents.architect import ArchitectAgent
from src.agents.intent_extractor import IntentExtractor
from src.agents.linter import LinterAgent
from src.agents.simulator import SimulationNode
from src.config import settings
from src.schemas.frameworks import FrameworkName
from src.schemas.prompt import (
    PromptIteration,
    PromptSchema,
    PromptTestResult,
    QualityScore,
    ResponseFormat,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class PromptBuilderState(TypedDict, total=False):
    """State flowing through the LangGraph reflection loop."""
    # Input
    user_input: str
    framework: str
    domain: str
    expected_format: str
    user_overrides: Optional[Dict[str, Any]]  # Phase 3: form overrides
    quality_threshold: int  # Phase 3: configurable (default 7)

    # Intermediate
    intent: Dict[str, Any]
    current_prompt: Optional[PromptSchema]
    test_result: Optional[PromptTestResult]
    quality_score: Optional[QualityScore]
    critique: Optional[str]

    # Tracking
    iteration: int
    max_iterations: int
    iterations_history: List[PromptIteration]

    # Output
    final_prompt: Optional[PromptSchema]
    final_score: Optional[QualityScore]
    status: str


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

async def extract_intent_node(state: PromptBuilderState) -> PromptBuilderState:
    """Parse user input into structured intent."""
    framework = state.get("framework", FrameworkName.COSTAR.value)
    extractor = IntentExtractor(framework=framework)
    intent = await extractor.extract(state["user_input"])

    return {
        **state,
        "intent": intent.model_dump(),
        "iteration": 1,
        "iterations_history": [],
        "status": "intent_extracted",
    }


async def architect_node(state: PromptBuilderState) -> PromptBuilderState:
    """Build or revise a prompt from intent."""
    framework = state.get("framework", FrameworkName.COSTAR.value)
    domain = state.get("domain", "General")
    critique = state.get("critique")

    architect = ArchitectAgent()
    prompt = await architect.draft(
        intent=state["intent"],
        framework=framework,
        domain=domain,
        critique=critique,
    )

    # Phase 3: Merge user overrides onto architect output
    overrides = state.get("user_overrides") or {}
    if overrides:
        override_fields = {}
        for key in ("must_include", "must_not_include", "constraints", "variables", "examples"):
            if key in overrides and overrides[key]:
                override_fields[key] = overrides[key]
        # Apply target_model and temperature from overrides
        if "target_model" in overrides and overrides["target_model"]:
            override_fields["target_model"] = overrides["target_model"]
        if "temperature" in overrides and overrides["temperature"] is not None:
            override_fields["temperature"] = overrides["temperature"]
        if "max_tokens" in overrides and overrides["max_tokens"] is not None:
            override_fields["max_tokens"] = overrides["max_tokens"]
        if override_fields:
            # Rebuild PromptSchema with merged overrides
            data = prompt.model_dump()
            data.update(override_fields)
            prompt = PromptSchema(**data)

    return {
        **state,
        "current_prompt": prompt,
        "status": "prompt_drafted",
    }


async def simulate_node(state: PromptBuilderState) -> PromptBuilderState:
    """Run the compiled prompt against Ollama."""
    prompt = state["current_prompt"]
    # Phase 3: Construct SimulationNode per-invocation with prompt's model/temp
    simulator = SimulationNode(
        model=prompt.target_model.value,
        temperature=prompt.temperature,
    )
    expected_format = state.get("expected_format", ResponseFormat.TEXT.value)

    test_result = await simulator.simulate(
        prompt_schema=prompt,
        expected_format=expected_format,
    )

    return {
        **state,
        "test_result": test_result,
        "status": "simulated",
    }


async def linter_node(state: PromptBuilderState) -> PromptBuilderState:
    """Evaluate prompt quality and produce a score."""
    linter = LinterAgent()
    score = linter.evaluate(
        prompt_schema=state["current_prompt"],
        test_result=state.get("test_result"),
    )

    # Record this iteration
    iteration_num = state.get("iteration", 1)
    history = list(state.get("iterations_history", []))
    history.append(PromptIteration(
        iteration_number=iteration_num,
        prompt_version=state["current_prompt"],
        test_result=state.get("test_result"),
        quality_score=score,
        critique=state.get("critique") or "",
        action_taken="initial_draft" if iteration_num == 1 else "revision",
    ))

    # Build critique from issues for potential next iteration
    critique = None
    if not score.passes_threshold():
        critique = "Issues found:\n" + "\n".join(f"- {issue}" for issue in score.issues)
        if score.suggestions:
            critique += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in score.suggestions)

    return {
        **state,
        "quality_score": score,
        "critique": critique,
        "iteration": iteration_num + 1,
        "iterations_history": history,
        "status": "evaluated",
    }


async def finalize_node(state: PromptBuilderState) -> PromptBuilderState:
    """Package the final result."""
    return {
        **state,
        "final_prompt": state["current_prompt"],
        "final_score": state["quality_score"],
        "status": "complete",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_continue(state: PromptBuilderState) -> str:
    """Decide whether to refine or finalize."""
    score = state.get("quality_score")
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", 3)
    # Phase 3: Use configurable quality threshold
    threshold = state.get("quality_threshold", 7)

    # Pass if score meets threshold
    if score is not None and score.passes_threshold(threshold=threshold):
        return "finalize"

    # Stop if max iterations reached
    if iteration > max_iter:
        return "finalize"

    return "revise"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_prompt_graph() -> StateGraph:
    """
    Build the LangGraph for the prompt reflection loop.

    Flow: extract_intent -> architect -> simulate -> linter
          -> conditional: score >= 7 -> finalize
                          score < 7 -> architect (with critique)
                          max iterations -> finalize
    """
    graph = StateGraph(PromptBuilderState)

    # Add nodes
    graph.add_node("extract_intent", extract_intent_node)
    graph.add_node("architect", architect_node)
    graph.add_node("simulate", simulate_node)
    graph.add_node("linter", linter_node)
    graph.add_node("finalize", finalize_node)

    # Linear flow: extract -> architect -> simulate -> linter
    graph.add_edge("extract_intent", "architect")
    graph.add_edge("architect", "simulate")
    graph.add_edge("simulate", "linter")

    # Conditional: linter -> finalize or architect (revision)
    graph.add_conditional_edges(
        "linter",
        should_continue,
        {
            "finalize": "finalize",
            "revise": "architect",
        },
    )

    # Finalize -> END
    graph.add_edge("finalize", END)

    # Entry point
    graph.set_entry_point("extract_intent")

    return graph


def compile_prompt_graph():
    """Build and compile the graph, ready for invocation."""
    graph = build_prompt_graph()
    return graph.compile()
