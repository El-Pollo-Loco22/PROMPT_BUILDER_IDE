"""
Simulation Node (Phase 2D)

Runs compiled prompts against local Ollama, captures response text,
token count, execution time, and evaluates format compliance and
required/unwanted content detection.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import settings
from src.schemas.prompt import PromptSchema, PromptTestResult, ResponseFormat


# ---------------------------------------------------------------------------
# Format Compliance Checkers
# ---------------------------------------------------------------------------

def _check_json_format(text: str) -> bool:
    """Check if the response is valid JSON."""
    text = text.strip()
    # Strip markdown fences
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:]
        if "```" in text:
            text = text[:text.rindex("```")]
        text = text.strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _check_code_format(text: str) -> bool:
    """Check if the response contains a code block."""
    # Accept fenced code blocks or content with common code indicators
    if "```" in text:
        return True
    # Heuristic: lines starting with def/class/import/function/const/let/var
    code_patterns = [
        r"^\s*(def |class |import |from |function |const |let |var |#include)",
        r"^\s*(public |private |protected |static )",
        r"^\s*\{",
    ]
    for pattern in code_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def _check_markdown_format(text: str) -> bool:
    """Check if the response uses markdown formatting."""
    md_indicators = ["# ", "## ", "**", "- ", "* ", "```", "| "]
    count = sum(1 for indicator in md_indicators if indicator in text)
    return count >= 2


def _check_list_format(text: str) -> bool:
    """Check if the response is a bulleted/numbered list."""
    lines = text.strip().split("\n")
    list_lines = 0
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-*•]\s", stripped) or re.match(r"^\d+[\.)]\s", stripped):
            list_lines += 1
    # At least 2 list items
    return list_lines >= 2


def check_format_compliance(text: str, expected_format: str) -> bool:
    """
    Check if the response matches the expected format.

    Args:
        text: The model's response text
        expected_format: One of ResponseFormat values or a string format name
    """
    if not text.strip():
        return False

    checkers = {
        ResponseFormat.JSON.value: _check_json_format,
        ResponseFormat.CODE.value: _check_code_format,
        ResponseFormat.MARKDOWN.value: _check_markdown_format,
        ResponseFormat.LIST.value: _check_list_format,
        ResponseFormat.TEXT.value: lambda _: True,  # plain text always passes
    }

    checker = checkers.get(expected_format)
    if checker is None:
        # Unknown format — default to pass
        return True
    return checker(text)


# ---------------------------------------------------------------------------
# Content Detection
# ---------------------------------------------------------------------------

def check_required_content(
    text: str, must_include: List[str]
) -> tuple[List[str], List[str]]:
    """
    Check which required elements are present/missing in the response.

    Returns (found, missing) lists.
    """
    text_lower = text.lower()
    found = []
    missing = []
    for item in must_include:
        if item.lower() in text_lower:
            found.append(item)
        else:
            missing.append(item)
    return found, missing


def check_unwanted_content(text: str, must_not_include: List[str]) -> List[str]:
    """
    Check which unwanted elements are present in the response.

    Returns list of detected unwanted items.
    """
    text_lower = text.lower()
    detected = []
    for item in must_not_include:
        if item.lower() in text_lower:
            detected.append(item)
    return detected


# ---------------------------------------------------------------------------
# Simulation Node
# ---------------------------------------------------------------------------

class SimulationNode:
    """
    Runs a compiled prompt against local Ollama and captures results.
    Produces a PromptTestResult with response text, metrics, and
    format/content compliance checks.
    """

    def __init__(
        self,
        model: str = settings.default_model,
        base_url: str = settings.ollama_base_url,
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    async def simulate(
        self,
        prompt_schema: PromptSchema,
        expected_format: str = ResponseFormat.TEXT.value,
    ) -> PromptTestResult:
        """
        Run the compiled prompt against Ollama and evaluate the result.

        Args:
            prompt_schema: The prompt to test
            expected_format: Expected response format for compliance checking
        """
        compiled = prompt_schema.compile_prompt()

        start_time = time.perf_counter()
        response = await self.llm.ainvoke([HumanMessage(content=compiled)])
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        response_text = response.content
        token_count = _estimate_tokens(response_text)

        # Format compliance
        follows_format = check_format_compliance(response_text, expected_format)

        # Content detection
        found, missing = check_required_content(
            response_text, prompt_schema.must_include
        )
        unwanted = check_unwanted_content(
            response_text, prompt_schema.must_not_include
        )

        return PromptTestResult(
            prompt_used=compiled,
            model_response=response_text,
            token_count=token_count,
            execution_time_ms=elapsed_ms,
            model_used=self.model,
            follows_format=follows_format,
            includes_required=found,
            missing_required=missing,
            unwanted_content=unwanted,
        )

    def simulate_sync(
        self,
        prompt_schema: PromptSchema,
        expected_format: str = ResponseFormat.TEXT.value,
    ) -> PromptTestResult:
        """Synchronous version of simulate()."""
        compiled = prompt_schema.compile_prompt()

        start_time = time.perf_counter()
        response = self.llm.invoke([HumanMessage(content=compiled)])
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        response_text = response.content
        token_count = _estimate_tokens(response_text)

        follows_format = check_format_compliance(response_text, expected_format)
        found, missing = check_required_content(
            response_text, prompt_schema.must_include
        )
        unwanted = check_unwanted_content(
            response_text, prompt_schema.must_not_include
        )

        return PromptTestResult(
            prompt_used=compiled,
            model_response=response_text,
            token_count=token_count,
            execution_time_ms=elapsed_ms,
            model_used=self.model,
            follows_format=follows_format,
            includes_required=found,
            missing_required=missing,
            unwanted_content=unwanted,
        )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, len(text) // 4)
