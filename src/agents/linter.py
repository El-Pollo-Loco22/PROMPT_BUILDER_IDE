"""
Linter Agent (Phase 2E)

Multi-dimensional quality evaluation of prompts. Scores clarity,
specificity, structure, constraints, and token efficiency. Detects
risks (prompt injection, token bloat, ambiguity). Produces a
QualityScore with human-readable feedback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.schemas.frameworks import FrameworkDef, get_framework
from src.schemas.prompt import PromptSchema, PromptTestResult, QualityScore


# ---------------------------------------------------------------------------
# Heuristic Scoring Functions
# ---------------------------------------------------------------------------

def score_clarity(sections: Dict[str, str]) -> tuple[int, List[str], List[str]]:
    """
    Score how clear and unambiguous the prompt sections are.
    Returns (score, strengths, issues).
    """
    strengths = []
    issues = []

    all_text = " ".join(sections.values())
    word_count = len(all_text.split())

    # Vague language detection
    vague_words = ["maybe", "somehow", "stuff", "things", "whatever", "etc",
                   "kind of", "sort of", "something like"]
    vague_found = [w for w in vague_words if w in all_text.lower()]
    if vague_found:
        issues.append(f"Vague language detected: {', '.join(vague_found)}")

    # Check for specificity indicators
    specific_indicators = ["must", "should", "exactly", "specifically",
                          "ensure", "require", "always", "never"]
    specific_found = [w for w in specific_indicators if w in all_text.lower()]
    if len(specific_found) >= 2:
        strengths.append("Uses clear directive language")

    # Check for well-defined sections (not too short)
    short_sections = [k for k, v in sections.items() if len(v.strip()) < 15]
    if short_sections:
        issues.append(f"Very short sections (may lack detail): {', '.join(short_sections)}")

    # Score calculation
    score = 7  # baseline
    score -= len(vague_found)
    score -= min(len(short_sections), 2)
    if len(specific_found) >= 2:
        score += 1
    if word_count > 20:
        score += 1

    return max(1, min(10, score)), strengths, issues


def score_specificity(sections: Dict[str, str]) -> tuple[int, List[str], List[str]]:
    """
    Score how specific and actionable the prompt is.
    Returns (score, strengths, issues).
    """
    strengths = []
    issues = []

    all_text = " ".join(sections.values())

    # Check for concrete details (numbers, names, formats)
    has_numbers = bool(re.search(r"\d+", all_text))
    has_quotes = '"' in all_text or "'" in all_text
    has_examples = any(w in all_text.lower() for w in ["example", "e.g.", "such as", "for instance"])
    has_format_spec = any(w in all_text.lower() for w in ["json", "csv", "markdown", "html",
                                                          "xml", "yaml", "code", "list",
                                                          "bullet", "table"])

    if has_numbers:
        strengths.append("Includes specific quantities or constraints")
    if has_examples:
        strengths.append("References examples for clarity")
    if has_format_spec:
        strengths.append("Specifies output format")

    # Penalty for being too generic
    generic_phrases = ["do something", "help me", "write something",
                       "make it good", "be creative"]
    generic_found = [p for p in generic_phrases if p in all_text.lower()]
    if generic_found:
        issues.append(f"Generic phrasing detected: {', '.join(generic_found)}")

    score = 6  # baseline
    score += int(has_numbers) + int(has_examples) + int(has_format_spec) + int(has_quotes)
    score -= len(generic_found)

    return max(1, min(10, score)), strengths, issues


def score_structure(
    prompt_schema: PromptSchema,
    framework_def: FrameworkDef,
) -> tuple[int, List[str], List[str]]:
    """
    Score structural completeness against the framework definition.
    Returns (score, strengths, issues).
    """
    strengths = []
    issues = []

    required = framework_def.required_keys
    all_keys = framework_def.all_keys
    present = set(prompt_schema.sections.keys())

    # Check required sections filled
    required_filled = required & present
    required_missing = required - present

    if not required_missing:
        strengths.append("All required framework sections are present")
    else:
        issues.append(f"Missing required sections: {', '.join(required_missing)}")

    # Check optional sections used
    optional_keys = all_keys - required
    optional_filled = optional_keys & present
    if optional_filled:
        strengths.append(f"Uses optional sections: {', '.join(optional_filled)}")

    # Check for substantive content (not just defaults or very short)
    thin_sections = []
    for key in present:
        val = prompt_schema.sections[key]
        if len(val.strip()) < 10 and key in required:
            thin_sections.append(key)
    if thin_sections:
        issues.append(f"Required sections with minimal content: {', '.join(thin_sections)}")

    # Score
    total_sections = len(all_keys)
    filled_ratio = len(present) / total_sections if total_sections else 0
    required_ratio = len(required_filled) / len(required) if required else 1.0

    score = int(required_ratio * 6 + filled_ratio * 4)
    if thin_sections:
        score -= 1

    return max(1, min(10, score)), strengths, issues


def score_constraints(prompt_schema: PromptSchema) -> tuple[int, List[str], List[str]]:
    """
    Score how well constraints and boundaries are defined.
    Returns (score, strengths, issues).
    """
    strengths = []
    issues = []

    has_constraints = bool(prompt_schema.constraints)
    has_must_include = bool(prompt_schema.must_include)
    has_must_not_include = bool(prompt_schema.must_not_include)

    # Check for implicit constraints in section text
    all_text = " ".join(prompt_schema.sections.values())
    constraint_words = ["must", "should not", "avoid", "never", "always",
                       "limit", "maximum", "minimum", "only", "do not"]
    implicit_constraints = [w for w in constraint_words if w in all_text.lower()]

    if has_constraints:
        strengths.append(f"Explicit constraints defined ({len(prompt_schema.constraints)})")
    if has_must_include:
        strengths.append(f"Required elements specified ({len(prompt_schema.must_include)})")
    if has_must_not_include:
        strengths.append(f"Exclusions defined ({len(prompt_schema.must_not_include)})")
    if implicit_constraints:
        strengths.append("Inline constraints in prompt text")

    if not has_constraints and not implicit_constraints:
        issues.append("No constraints defined — output may be unpredictable")
    if not has_must_include and not has_must_not_include:
        issues.append("No required/excluded elements — harder to validate output")

    score = 5  # baseline
    score += int(has_constraints) * 2
    score += int(has_must_include)
    score += int(has_must_not_include)
    score += min(len(implicit_constraints), 2)

    return max(1, min(10, score)), strengths, issues


def score_token_efficiency(prompt_schema: PromptSchema) -> tuple[int, List[str], List[str]]:
    """
    Score token efficiency — penalize bloat, reward conciseness.
    Returns (score, strengths, issues).
    """
    strengths = []
    issues = []

    compiled = prompt_schema.compile_prompt()
    token_est = prompt_schema.token_estimate()
    word_count = len(compiled.split())

    # Check for repetition
    sentences = [s.strip() for s in compiled.split(".") if s.strip()]
    unique_sentences = set(s.lower() for s in sentences)
    if len(sentences) > 2 and len(unique_sentences) < len(sentences) * 0.7:
        issues.append("Significant repetition detected in prompt text")

    # Check for filler words
    filler_words = ["basically", "essentially", "actually", "literally",
                    "in order to", "for the purpose of", "as a matter of fact"]
    all_text = compiled.lower()
    fillers_found = [w for w in filler_words if w in all_text]
    if fillers_found:
        issues.append(f"Filler words detected: {', '.join(fillers_found)}")

    # Token budget
    if token_est > 1000:
        issues.append(f"Prompt is very long ({token_est} estimated tokens) — consider condensing")
    elif token_est > 500:
        issues.append(f"Prompt is moderately long ({token_est} estimated tokens)")
    elif token_est < 30:
        issues.append("Prompt may be too short to provide adequate context")
    else:
        strengths.append(f"Good token efficiency ({token_est} estimated tokens)")

    score = 7  # baseline
    if token_est > 1000:
        score -= 3
    elif token_est > 500:
        score -= 1
    elif token_est < 30:
        score -= 2
    score -= len(fillers_found)
    if len(sentences) > 2 and len(unique_sentences) < len(sentences) * 0.7:
        score -= 2

    return max(1, min(10, score)), strengths, issues


# ---------------------------------------------------------------------------
# Risk Detection
# ---------------------------------------------------------------------------

def detect_injection_risk(prompt_schema: PromptSchema) -> bool:
    """Check for potential prompt injection patterns."""
    all_text = " ".join(prompt_schema.sections.values()).lower()

    injection_patterns = [
        r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"disregard\s+(everything|all|previous)",
        r"you\s+are\s+now\s+",
        r"new\s+instructions?:",
        r"system\s*:\s*",
        r"<\s*/?\s*system\s*>",
        r"forget\s+(everything|what|your)",
        r"pretend\s+you\s+are",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    ]

    for pattern in injection_patterns:
        if re.search(pattern, all_text):
            return True
    return False


def detect_token_bloat(prompt_schema: PromptSchema) -> bool:
    """Check if the prompt is excessively long."""
    return prompt_schema.token_estimate() > 800


def detect_ambiguity(prompt_schema: PromptSchema) -> bool:
    """Check for contradictory or ambiguous instructions."""
    all_text = " ".join(prompt_schema.sections.values()).lower()

    # Contradiction patterns
    contradiction_pairs = [
        ("be concise", "provide detailed"),
        ("be brief", "comprehensive"),
        ("short", "exhaustive"),
        ("simple", "complex"),
    ]

    for a, b in contradiction_pairs:
        if a in all_text and b in all_text:
            return True

    # Excessive hedging
    hedge_words = ["maybe", "perhaps", "possibly", "might", "could be",
                   "not sure", "uncertain"]
    hedge_count = sum(1 for w in hedge_words if w in all_text)
    if hedge_count >= 3:
        return True

    return False


# ---------------------------------------------------------------------------
# Linter Agent
# ---------------------------------------------------------------------------

class LinterAgent:
    """
    Multi-dimensional quality evaluator for prompts.
    Produces a QualityScore with component scores, feedback, and risk flags.
    All scoring is heuristic-based (no LLM call needed).
    """

    def evaluate(
        self,
        prompt_schema: PromptSchema,
        test_result: Optional[PromptTestResult] = None,
    ) -> QualityScore:
        """
        Evaluate a prompt and produce a QualityScore.

        Args:
            prompt_schema: The prompt to evaluate
            test_result: Optional simulation result for additional context
        """
        framework_def = get_framework(prompt_schema.framework)

        # Collect all scores and feedback
        all_strengths: List[str] = []
        all_issues: List[str] = []
        all_suggestions: List[str] = []

        # Score each dimension
        clarity, s, i = score_clarity(prompt_schema.sections)
        all_strengths.extend(s)
        all_issues.extend(i)

        specificity, s, i = score_specificity(prompt_schema.sections)
        all_strengths.extend(s)
        all_issues.extend(i)

        structure, s, i = score_structure(prompt_schema, framework_def)
        all_strengths.extend(s)
        all_issues.extend(i)

        constraint, s, i = score_constraints(prompt_schema)
        all_strengths.extend(s)
        all_issues.extend(i)

        efficiency, s, i = score_token_efficiency(prompt_schema)
        all_strengths.extend(s)
        all_issues.extend(i)

        # Risk detection
        has_injection = detect_injection_risk(prompt_schema)
        has_bloat = detect_token_bloat(prompt_schema)
        has_ambiguity = detect_ambiguity(prompt_schema)

        if has_injection:
            all_issues.append("RISK: Potential prompt injection pattern detected")
        if has_bloat:
            all_issues.append("RISK: Token bloat — prompt exceeds recommended length")
        if has_ambiguity:
            all_issues.append("RISK: Ambiguous or contradictory instructions detected")

        # Incorporate test result feedback
        if test_result is not None:
            if not test_result.follows_format:
                all_issues.append("Simulation: output did not match expected format")
                all_suggestions.append("Add clearer format instructions to the prompt")
            if test_result.missing_required:
                items = ", ".join(test_result.missing_required)
                all_issues.append(f"Simulation: missing required elements: {items}")
                all_suggestions.append("Emphasize required elements more strongly in the prompt")
            if test_result.unwanted_content:
                items = ", ".join(test_result.unwanted_content)
                all_issues.append(f"Simulation: unwanted content detected: {items}")
                all_suggestions.append("Add explicit exclusion instructions for unwanted content")
            if test_result.follows_format and not test_result.missing_required:
                all_strengths.append("Simulation produced conforming output")

        # Generate suggestions from issues
        all_suggestions.extend(self._generate_suggestions(all_issues, prompt_schema))

        # Overall score (weighted average)
        overall = self._compute_overall(
            clarity, specificity, structure, constraint, efficiency,
            has_injection, has_bloat, has_ambiguity,
        )

        # Build reasoning
        reasoning = self._build_reasoning(
            clarity, specificity, structure, constraint, efficiency,
            has_injection, has_bloat, has_ambiguity,
        )

        return QualityScore(
            overall_score=overall,
            clarity_score=clarity,
            specificity_score=specificity,
            structure_score=structure,
            constraint_score=constraint,
            token_efficiency_score=efficiency,
            strengths=all_strengths,
            issues=all_issues,
            suggestions=all_suggestions,
            has_injection_risk=has_injection,
            has_token_bloat=has_bloat,
            has_ambiguity=has_ambiguity,
            reasoning=reasoning,
        )

    @staticmethod
    def _compute_overall(
        clarity: int,
        specificity: int,
        structure: int,
        constraint: int,
        efficiency: int,
        has_injection: bool,
        has_bloat: bool,
        has_ambiguity: bool,
    ) -> int:
        """Compute weighted overall score."""
        weighted = (
            clarity * 0.25
            + specificity * 0.25
            + structure * 0.25
            + constraint * 0.15
            + efficiency * 0.10
        )
        score = round(weighted)

        # Risk penalties
        if has_injection:
            score = max(1, score - 3)
        if has_ambiguity:
            score = max(1, score - 1)

        return max(1, min(10, score))

    @staticmethod
    def _generate_suggestions(issues: List[str], prompt_schema: PromptSchema) -> List[str]:
        """Generate actionable suggestions based on detected issues."""
        suggestions = []

        for issue in issues:
            if "vague language" in issue.lower():
                suggestions.append("Replace vague terms with specific, measurable language")
            elif "very short sections" in issue.lower():
                suggestions.append("Expand short sections with more context and detail")
            elif "generic phrasing" in issue.lower():
                suggestions.append("Replace generic phrases with specific, actionable instructions")
            elif "no constraints" in issue.lower():
                suggestions.append("Add explicit constraints to bound the expected output")
            elif "repetition" in issue.lower():
                suggestions.append("Remove duplicate content to improve token efficiency")
            elif "filler words" in issue.lower():
                suggestions.append("Remove filler words for more direct communication")
            elif "too long" in issue.lower() or "very long" in issue.lower():
                suggestions.append("Condense the prompt — focus on essential instructions only")
            elif "too short" in issue.lower():
                suggestions.append("Add more context to help the model understand the task")

        return suggestions

    @staticmethod
    def _build_reasoning(
        clarity: int,
        specificity: int,
        structure: int,
        constraint: int,
        efficiency: int,
        has_injection: bool,
        has_bloat: bool,
        has_ambiguity: bool,
    ) -> str:
        """Build a human-readable reasoning string."""
        parts = [
            f"Clarity: {clarity}/10",
            f"Specificity: {specificity}/10",
            f"Structure: {structure}/10",
            f"Constraints: {constraint}/10",
            f"Token Efficiency: {efficiency}/10",
        ]

        risks = []
        if has_injection:
            risks.append("prompt injection risk")
        if has_bloat:
            risks.append("token bloat")
        if has_ambiguity:
            risks.append("ambiguous instructions")

        reasoning = "Scores: " + ", ".join(parts)
        if risks:
            reasoning += f". Risks detected: {', '.join(risks)}"

        return reasoning
