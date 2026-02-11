"""
Intent Extraction Node (Phase 2B)

Pre-processing agent that parses free-form user input and maps it
to structured framework sections for downstream prompt construction.
Supports CO-STAR (default), RACE, APE, CRISPE, and custom frameworks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import settings
from src.schemas.frameworks import FrameworkDef, FrameworkName, get_framework


def _try_repair_json(fragment: str) -> Optional[Any]:
    """
    Attempt to repair truncated JSON by progressively trimming from the end
    and closing open structures. Returns parsed dict or None.
    """
    # Strategy 1: strip trailing comma/whitespace, close open brackets/braces
    cleaned = fragment.rstrip(",\n\t ")
    open_braces = cleaned.count("{") - cleaned.count("}")
    open_brackets = cleaned.count("[") - cleaned.count("]")
    attempt = cleaned + "]" * open_brackets + "}" * open_braces
    try:
        return json.loads(attempt)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the last complete key-value pair and truncate there
    # Look for the last complete "key": "value" or "key": [...] or "key": number
    # by finding the last comma that separates complete entries
    for i in range(len(cleaned) - 1, 0, -1):
        if cleaned[i] == ",":
            truncated = cleaned[:i]
            ob = truncated.count("{") - truncated.count("}")
            ol = truncated.count("[") - truncated.count("]")
            attempt = truncated + "]" * ol + "}" * ob
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                continue

    return None


def build_extraction_prompt(framework_def: FrameworkDef) -> str:
    """
    Dynamically build a system prompt that instructs the LLM to extract
    the sections defined by the given framework.
    """
    # Build the JSON template from framework section definitions
    fields = {}
    for section in framework_def.sections:
        fields[section.key] = f"<{section.description}>"
    fields["missing_variables"] = ["list of unknowns or placeholders the user hasn't specified"]
    fields["constraints"] = ["any implicit or explicit limitations"]
    fields["response_format"] = "plain_text | json | code_block | markdown | bulleted_list"

    json_template = json.dumps(fields, indent=2)

    return f"""\
You are an expert prompt analyst.

Given a user's description of what they want an AI to do, extract fields \
for the {framework_def.display_name} framework and return them as a JSON \
object — nothing else, no markdown fences, just raw JSON:

{json_template}

Rules:
- If a field is unclear, make a reasonable inference and note it.
- "missing_variables" should list things the user probably needs to specify but didn't.
- "response_format" should be one of the exact enum values listed above.
- Return ONLY valid JSON. No explanation, no preamble.\
"""


class ExtractedIntent(BaseModel):
    """Structured output from intent extraction — framework-agnostic."""

    framework: str = Field(default=FrameworkName.COSTAR.value)
    sections: Dict[str, str] = Field(default_factory=dict)
    missing_variables: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    response_format: str = "plain_text"

    # Backward-compatible property accessors for CO-STAR fields
    @property
    def objective(self) -> str:
        return self.sections.get("objective", "")

    @property
    def context(self) -> str:
        return self.sections.get("context", "")

    @property
    def audience(self) -> str:
        return self.sections.get("audience", "")


class IntentExtractor:
    """
    Parses free-form user input via a local Ollama model and returns
    a structured ExtractedIntent suitable for feeding into the Architect agent.
    """

    def __init__(
        self,
        model: str = settings.default_model,
        base_url: str = settings.ollama_base_url,
        temperature: float = 0.3,
        framework: str = FrameworkName.COSTAR.value,
    ):
        self.framework = framework
        # Resolve framework name to definition
        self.framework_def = get_framework(framework)
        self.system_prompt = build_extraction_prompt(self.framework_def)
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    async def extract(self, user_input: str) -> ExtractedIntent:
        """Parse user input into structured framework intent."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User request:\n{user_input}"),
        ]

        response = await self.llm.ainvoke(messages)
        return self._parse_response(response.content, self.framework, self.framework_def)

    def extract_sync(self, user_input: str) -> ExtractedIntent:
        """Synchronous version for non-async contexts."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User request:\n{user_input}"),
        ]

        response = self.llm.invoke(messages)
        return self._parse_response(response.content, self.framework, self.framework_def)

    @staticmethod
    def _parse_response(
        raw: str,
        framework: str = FrameworkName.COSTAR.value,
        framework_def: FrameworkDef | None = None,
    ) -> ExtractedIntent:
        """
        Parse LLM output into an ExtractedIntent.

        Handles common LLM quirks:
        - Markdown code fences wrapping JSON
        - Trailing text after the JSON object
        """
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
            if "```" in text:
                text = text[:text.rindex("```")]
            text = text.strip()

        # Extract JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            # LLM may return truncated JSON — try adding closing brace
            if start != -1:
                text = text + "}"
                end = len(text)
            else:
                raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")

        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            # Truncated JSON — attempt to salvage by closing open structures
            fragment = text[start:]
            data = _try_repair_json(fragment)
            if data is None:
                raise ValueError(f"Could not parse JSON from LLM response: {raw[:200]}")

        # Resolve framework def if not provided
        if framework_def is None:
            framework_def = get_framework(framework)

        # Separate framework sections from meta-fields
        # Normalize keys to lowercase (LLMs often capitalize)
        meta_keys = {"missing_variables", "constraints", "response_format"}
        section_keys = framework_def.all_keys
        sections = {}
        for key, value in data.items():
            norm_key = key.lower()
            if norm_key in meta_keys:
                continue
            if norm_key in section_keys:
                sections[norm_key] = str(value)

        return ExtractedIntent(
            framework=framework,
            sections=sections,
            missing_variables=data.get("missing_variables", []),
            constraints=data.get("constraints", []),
            response_format=data.get("response_format", "plain_text"),
        )
