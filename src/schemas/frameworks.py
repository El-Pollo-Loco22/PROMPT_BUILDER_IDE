"""
Framework Registry (Multi-Framework Support)

Defines prompt engineering frameworks as pure data. Each framework
specifies its sections (ordered), which are required, and a compile
template for rendering the final prompt string.

Built-in frameworks: CO-STAR, RACE, APE, CRISPE.
Custom frameworks can be registered at runtime via register_framework().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class FrameworkName(str, Enum):
    """Built-in prompt engineering frameworks."""

    COSTAR = "co_star"
    RACE = "race"
    APE = "ape"
    CRISPE = "crispe"


@dataclass(frozen=True)
class SectionDef:
    """Definition of a single section within a framework."""

    key: str
    label: str
    description: str
    required: bool = True
    default: Optional[str] = None
    min_length: int = 0


@dataclass(frozen=True)
class FrameworkDef:
    """
    Complete definition of a prompt engineering framework.
    Pure data — no Pydantic model needed per framework.
    """

    name: str  # FrameworkName enum for built-ins, any string for custom
    display_name: str
    sections: tuple[SectionDef, ...]
    compile_template: str

    @property
    def required_keys(self) -> frozenset[str]:
        return frozenset(s.key for s in self.sections if s.required)

    @property
    def all_keys(self) -> frozenset[str]:
        return frozenset(s.key for s in self.sections)

    @property
    def defaults(self) -> Dict[str, str]:
        return {s.key: s.default for s in self.sections if s.default is not None}

    @property
    def min_lengths(self) -> Dict[str, int]:
        return {s.key: s.min_length for s in self.sections if s.min_length > 0}


# ---------------------------------------------------------------------------
# Built-in framework definitions
# ---------------------------------------------------------------------------

COSTAR = FrameworkDef(
    name=FrameworkName.COSTAR,
    display_name="CO-STAR",
    sections=(
        SectionDef("context", "Context", "Background information the AI needs", required=True, min_length=20),
        SectionDef("objective", "Objective", "Clear, specific goal", required=True, min_length=10),
        SectionDef("style", "Style", "Writing style and approach", default="professional and technical"),
        SectionDef("tone", "Tone", "Emotional quality of the response", default="helpful and informative"),
        SectionDef("audience", "Audience", "Who the response is for and their expertise level", required=True),
        SectionDef("response", "Response", "Expected output format", default="plain_text"),
    ),
    compile_template=(
        "**Context:**\n{context}\n\n"
        "**Objective:**\n{objective}\n\n"
        "{constraints_block}"
        "{required_elements_block}"
        "{examples_block}"
        "**Response Format:**\n{response}\n\n"
        "**Style & Tone:**\n"
        "Write for {audience} in a {style} style with a {tone} tone.\n"
    ),
)

RACE = FrameworkDef(
    name=FrameworkName.RACE,
    display_name="RACE",
    sections=(
        SectionDef("role", "Role", "Who the AI should act as", required=True),
        SectionDef("action", "Action", "What specifically the AI should do", required=True, min_length=10),
        SectionDef("context", "Context", "Background information and situation", required=True, min_length=20),
        SectionDef("expectation", "Expectation", "Expected output and quality criteria", required=True),
    ),
    compile_template=(
        "**Role:**\n{role}\n\n"
        "**Action:**\n{action}\n\n"
        "**Context:**\n{context}\n\n"
        "{constraints_block}"
        "{required_elements_block}"
        "{examples_block}"
        "**Expectation:**\n{expectation}\n"
    ),
)

APE = FrameworkDef(
    name=FrameworkName.APE,
    display_name="APE",
    sections=(
        SectionDef("action", "Action", "What the AI should do", required=True, min_length=10),
        SectionDef("purpose", "Purpose", "Why this task matters and the goal", required=True),
        SectionDef("expectation", "Expectation", "Expected output format and quality", required=True),
    ),
    compile_template=(
        "**Action:**\n{action}\n\n"
        "**Purpose:**\n{purpose}\n\n"
        "{constraints_block}"
        "{required_elements_block}"
        "{examples_block}"
        "**Expectation:**\n{expectation}\n"
    ),
)

CRISPE = FrameworkDef(
    name=FrameworkName.CRISPE,
    display_name="CRISPE",
    sections=(
        SectionDef("context", "Context", "Background info and situation", required=True, min_length=20),
        SectionDef("role", "Role", "Who the AI should be", required=True),
        SectionDef("instruction", "Instruction", "Specific task instructions", required=True, min_length=10),
        SectionDef("schema", "Schema", "Output structure and format", required=True),
        SectionDef("persona", "Persona", "Communication style and personality", default="professional"),
        SectionDef("crispe_examples", "Examples", "Example inputs and outputs", required=False, default=""),
    ),
    compile_template=(
        "**Context:**\n{context}\n\n"
        "**Role:**\n{role}\n\n"
        "**Instruction:**\n{instruction}\n\n"
        "{constraints_block}"
        "{required_elements_block}"
        "{examples_block}"
        "**Schema:**\n{schema}\n\n"
        "**Persona:**\n{persona}\n"
    ),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, FrameworkDef] = {
    FrameworkName.COSTAR.value: COSTAR,
    FrameworkName.RACE.value: RACE,
    FrameworkName.APE.value: APE,
    FrameworkName.CRISPE.value: CRISPE,
}


def get_framework(name: str) -> FrameworkDef:
    """Look up a framework definition by name (enum value or string)."""
    # Accept both FrameworkName enum and raw strings
    key = name.value if isinstance(name, FrameworkName) else name
    return _REGISTRY[key]


def register_framework(defn: FrameworkDef) -> None:
    """Register a custom framework at runtime."""
    key = defn.name.value if isinstance(defn.name, FrameworkName) else defn.name
    _REGISTRY[key] = defn


def list_frameworks() -> list[str]:
    """Return all registered framework names."""
    return list(_REGISTRY.keys())
