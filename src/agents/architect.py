"""
Architect Agent (Phase 2C)

Builds and revises prompts using the selected framework's structure.
Incorporates domain-specific knowledge base best practices and
handles critique-driven revision from the Linter agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.config import settings
from src.schemas.frameworks import FrameworkDef, FrameworkName, get_framework
from src.schemas.prompt import PromptSchema


# ---------------------------------------------------------------------------
# Knowledge Base Loader
# ---------------------------------------------------------------------------

def load_knowledge_base(kb_dir: str | Path = "knowledge-base") -> Dict[str, Any]:
    """
    Load all JSON knowledge base files from a directory.
    Returns a dict keyed by category name.
    """
    kb: Dict[str, Any] = {}
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return kb
    for f in kb_path.glob("*.json"):
        with open(f) as fh:
            data = json.load(fh)
            kb[data.get("category", f.stem)] = data
    return kb


def _format_best_practices(kb: Dict[str, Any], domain: str, limit: int = 5) -> str:
    """Extract best practices for a domain from the knowledge base."""
    domain_data = kb.get(domain, kb.get("General", {}))
    practices = domain_data.get("bestPractices", [])[:limit]
    if not practices:
        return ""
    lines = [f"- {p['rule']}: {p['why']}" for p in practices]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System Prompt Builder
# ---------------------------------------------------------------------------

def build_architect_prompt(
    framework_def: FrameworkDef,
    best_practices: str = "",
    critique: str | None = None,
) -> str:
    """Build the system prompt for the Architect agent."""
    sections_desc = "\n".join(
        f"- **{s.label}** ({s.key}): {s.description}"
        + (f" [default: {s.default}]" if s.default else " [REQUIRED]")
        for s in framework_def.sections
    )

    prompt = f"""\
You are an expert prompt engineer. Your job is to build high-quality prompts \
using the {framework_def.display_name} framework.

The {framework_def.display_name} framework has these sections:
{sections_desc}

You must return a JSON object with a key for each section listed above. \
Fill in every required section with substantive content (not just a placeholder). \
For optional sections, include them if you can improve the prompt quality.

Return ONLY valid JSON. No markdown fences, no explanation, just the JSON object.\
"""

    if best_practices:
        prompt += f"""

Domain best practices to incorporate:
{best_practices}\
"""

    if critique:
        prompt += f"""

IMPORTANT — REVISION MODE:
The previous version of this prompt was reviewed and found lacking.
Address these specific issues in your revision:
{critique}

Make targeted improvements. Do not rewrite from scratch unless necessary.\
"""

    return prompt


# ---------------------------------------------------------------------------
# Architect Agent
# ---------------------------------------------------------------------------

class ArchitectAgent:
    """
    Builds and revises prompts using the selected framework's structure.
    Loads domain knowledge for best-practice injection.
    """

    def __init__(
        self,
        model: str = settings.default_model,
        base_url: str = settings.ollama_base_url,
        temperature: float = 0.7,
        knowledge_base: Dict[str, Any] | None = None,
        kb_dir: str | Path = "knowledge-base",
    ):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self.knowledge_base = knowledge_base if knowledge_base is not None else load_knowledge_base(kb_dir)

    async def draft(
        self,
        intent: Dict[str, Any],
        framework: str = FrameworkName.COSTAR.value,
        domain: str = "General",
        critique: str | None = None,
    ) -> PromptSchema:
        """
        Generate or revise a PromptSchema from extracted intent.

        Args:
            intent: Dict with at least a 'sections' key (from ExtractedIntent)
                    or raw CO-STAR fields (objective, context, audience, etc.)
            framework: Framework name to use
            domain: Knowledge base domain for best practices
            critique: If provided, the agent revises instead of drafting fresh
        """
        framework_def = get_framework(framework)
        best_practices = _format_best_practices(self.knowledge_base, domain)
        system_prompt = build_architect_prompt(framework_def, best_practices, critique)

        # Build the user message from intent
        user_content = self._build_user_message(intent, framework_def, critique)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        response = await self.llm.ainvoke(messages)
        return self._parse_response(response.content, framework)

    def draft_sync(
        self,
        intent: Dict[str, Any],
        framework: str = FrameworkName.COSTAR.value,
        domain: str = "General",
        critique: str | None = None,
    ) -> PromptSchema:
        """Synchronous version of draft()."""
        framework_def = get_framework(framework)
        best_practices = _format_best_practices(self.knowledge_base, domain)
        system_prompt = build_architect_prompt(framework_def, best_practices, critique)

        user_content = self._build_user_message(intent, framework_def, critique)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        response = self.llm.invoke(messages)
        return self._parse_response(response.content, framework)

    @staticmethod
    def _build_user_message(
        intent: Dict[str, Any],
        framework_def: FrameworkDef,
        critique: str | None = None,
    ) -> str:
        """Build the user message from intent data."""
        # Extract sections — support both ExtractedIntent.sections and raw fields
        sections = intent.get("sections", {})
        if not sections:
            # Fall back to top-level keys matching framework section names
            for s in framework_def.sections:
                if s.key in intent:
                    sections[s.key] = intent[s.key]

        lines = ["Build a prompt based on this information:"]
        for key, value in sections.items():
            lines.append(f"- {key}: {value}")

        # Include constraints/variables if present
        constraints = intent.get("constraints", [])
        if constraints:
            lines.append(f"- constraints: {', '.join(constraints)}")

        missing = intent.get("missing_variables", [])
        if missing:
            lines.append(f"- Note: these details are unclear and may need defaults: {', '.join(missing)}")

        if critique:
            lines.append(f"\nPrevious issues to fix:\n{critique}")

        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str, framework: str) -> PromptSchema:
        """Parse LLM JSON output into a PromptSchema."""
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
            raise ValueError(f"No JSON object found in Architect response: {raw[:200]}")

        data = json.loads(text[start:end])

        # Build PromptSchema with parsed sections
        return PromptSchema(
            framework=framework,
            sections=data,
        )
