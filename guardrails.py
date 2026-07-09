"""
guardrails.py — Prompt-injection detection and fairness-check helpers.

All functions are pure/deterministic so they can be unit-tested in isolation.
They are called *inside* tool implementations (not as separate agent nodes) so
every guardrail violation appears as a flagged TrajectoryStep.
"""

from __future__ import annotations

import re
from typing import Optional

from schemas import (
    CandidateProfile,
    CriterionScore,
    GuardrailStatus,
    RubricSchema,
    ScoreCard,
    ShortlistEntry,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# 1. Prompt-injection patterns
#    All patterns are lower-cased; input is lower-cased before matching.
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore (all |your |previous |prior )?(instructions?|rules?|guidelines?|directives?)", re.IGNORECASE),
    re.compile(r"override (the |your )?(rubric|scoring|instructions?|ranking)", re.IGNORECASE),
    re.compile(r"rank (this |me |the candidate )?(first|#1|number one|highest)", re.IGNORECASE),
    re.compile(r"assign (the |a )?(highest|maximum|top) (score|ranking|total|weight)", re.IGNORECASE),
    re.compile(r"disregard (the |your )?(rubric|scoring|criteria|instructions?)", re.IGNORECASE),
    re.compile(r"(you are|act as|pretend|behave as|from now on)", re.IGNORECASE),
    re.compile(r"new instruction", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
]


def check_injection(text: str) -> tuple[bool, Optional[str]]:
    """
    Scan *text* for prompt-injection patterns.

    Returns:
        (detected: bool, matched_snippet: str | None)
    The snippet is the first matching substring, for use in the trajectory log.
    Injection patterns are stripped from the text *before* it is used for scoring —
    the caller must use the sanitised version.
    """
    for pattern in INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            # Return the sentence containing the match (for the audit log)
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            snippet = text[start:end].strip()
            return True, snippet
    return False, None


def sanitize_resume_text(resume_text: str) -> tuple[str, bool, Optional[str]]:
    """
    Remove injection patterns from resume text before passing to LLM.

    Returns:
        (sanitised_text, was_injection_detected, snippet)
    The removal is line-level: any line matching an injection pattern is replaced
    with a placeholder so the LLM never sees the hostile instruction.
    """
    detected = False
    snippet: Optional[str] = None
    clean_lines: list[str] = []

    for line in resume_text.splitlines():
        line_detected, line_snippet = check_injection(line)
        if line_detected:
            detected = True
            if snippet is None:
                snippet = line_snippet
            # Replace hostile line with a neutral placeholder
            clean_lines.append("[REDACTED — prompt-injection attempt detected and removed]")
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines), detected, snippet


# ---------------------------------------------------------------------------
# 2. Fairness check — name-swap test
# ---------------------------------------------------------------------------

def run_fairness_check(
    score_candidate_fn,          # callable: (profile, rubric) -> ScoreCard
    rubric: RubricSchema,
) -> dict:
    """
    Build two CandidateProfiles that are identical in every relevant dimension
    except for the name.  Score both with *score_candidate_fn* and compare.

    A fairness-check PASS requires weighted_total to be identical (within a
    floating-point epsilon of 0.01) for both profiles.

    Returns a dict with keys:
        passed: bool
        name_a: str, score_a: float
        name_b: str, score_b: float
        delta: float
        details: str   (human-readable explanation)
    """
    # The "base" profile — realistic mid-level ML candidate
    base_skills = [
        "Python (intermediate)",
        "scikit-learn",
        "pandas",
        "NumPy",
        "ML fundamentals: regression, classification",
    ]
    base_projects = [
        "Trained a Random Forest classifier on the UCI Adult dataset; achieved 85% accuracy using scikit-learn.",
        "Built a text classification pipeline (TF-IDF + Logistic Regression) for sentiment analysis as a coursework project.",
    ]
    base_education = ["B.Tech Computer Science, 2023"]

    profile_a = CandidateProfile(
        name="Alex Kumar",
        education=base_education,
        skills=base_skills,
        years_experience=0.5,
        projects=base_projects,
        raw_flags=[],
    )
    profile_b = CandidateProfile(
        name="Priya Singh",
        education=base_education,
        skills=base_skills,
        years_experience=0.5,
        projects=base_projects,
        raw_flags=[],
    )

    card_a: ScoreCard = score_candidate_fn(profile_a, rubric)
    card_b: ScoreCard = score_candidate_fn(profile_b, rubric)

    delta = abs(card_a.weighted_total - card_b.weighted_total)
    passed = delta < 0.01

    return {
        "passed": passed,
        "name_a": profile_a.name,
        "score_a": card_a.weighted_total,
        "name_b": profile_b.name,
        "score_b": card_b.weighted_total,
        "delta": round(delta, 4),
        "details": (
            f"{profile_a.name} scored {card_a.weighted_total:.2f}; "
            f"{profile_b.name} scored {card_b.weighted_total:.2f}. "
            f"Delta = {delta:.4f}. "
            f"Result: {'PASS ✅' if passed else 'FAIL ❌ — scores differ despite identical experience'}."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Step-cap enforcer
# ---------------------------------------------------------------------------

def check_step_cap(
    guardrail_status: GuardrailStatus,
    step_index: int,
) -> bool:
    """Return True if the step cap has been exceeded.  The caller should abort."""
    return step_index >= guardrail_status.steps_limit
