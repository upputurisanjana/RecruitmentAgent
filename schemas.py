"""
schemas.py — All Pydantic models for the TechVest Recruitment Agent.
These are the typed contracts shared by the agent graph, tools, guardrails, and UI.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Candidate Profile (output of parse_resume)
# ---------------------------------------------------------------------------

class CandidateProfile(BaseModel):
    name: str
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_experience: float = 0.0
    projects: list[str] = Field(default_factory=list)
    raw_flags: list[str] = Field(
        default_factory=list,
        description="Warning flags raised during parsing, e.g. 'possible prompt injection detected'.",
    )


# ---------------------------------------------------------------------------
# Scoring models (output of score_candidate)
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    criterion: str
    weight: float = Field(ge=0.0, le=1.0, description="Fraction of total score, sums to 1.0 across all criteria.")
    score: int = Field(ge=0, le=5, description="0–5 scale per rubric descriptor.")
    evidence: str = Field(
        description="Required: specific résumé line or section cited. Empty evidence forces score=0.",
    )


class ScoreCard(BaseModel):
    candidate: str
    criteria: list[CriterionScore] = Field(default_factory=list)
    weighted_total: float = Field(
        ge=0.0,
        le=5.0,
        description="Weight-adjusted total on a 0–5 scale: sum(weight * score) for each criterion.",
    )


# ---------------------------------------------------------------------------
# Interview slot
# ---------------------------------------------------------------------------

class InterviewSlot(BaseModel):
    day: str
    time: str


# ---------------------------------------------------------------------------
# Shortlist entry (one per candidate in the final output)
# ---------------------------------------------------------------------------

class ShortlistEntry(BaseModel):
    candidate: str
    verdict: Literal["INTERVIEW", "HOLD", "NOT A FIT"]
    weighted_score: float = Field(ge=0.0, le=5.0)
    justification: str = Field(
        description="Evidence-citing justification. Must reference specific résumé lines, never bare adjectives.",
    )
    scorecard: ScoreCard
    proposed_slot: Optional[InterviewSlot] = None
    action_status: Literal["not_applicable", "pending_approval", "approved", "rejected"] = "not_applicable"


# ---------------------------------------------------------------------------
# Trajectory step (one per agent thought / action / observation / decision)
# ---------------------------------------------------------------------------

class TrajectoryStep(BaseModel):
    step_index: int
    type: Literal["thought", "action", "observation", "decision"]
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    flagged: bool = Field(
        default=False,
        description="True if this step caught a guardrail violation (e.g. injection attempt).",
    )


# ---------------------------------------------------------------------------
# Guardrail status (live safety panel)
# ---------------------------------------------------------------------------

class GuardrailStatus(BaseModel):
    steps_used: int = 0
    steps_limit: int = 25
    human_gate: Literal["armed", "waiting_for_approval", "cleared", "rejected"] = "armed"
    injection_detected: bool = False
    injection_candidate: Optional[str] = None
    injection_snippet: Optional[str] = None
    fairness_check_run: bool = False
    fairness_check_passed: Optional[bool] = None
    fairness_score_a: Optional[float] = None
    fairness_score_b: Optional[float] = None


# ---------------------------------------------------------------------------
# Rubric schema
# ---------------------------------------------------------------------------

class RubricCriterion(BaseModel):
    name: str
    weight: float = Field(ge=0.0, le=1.0)
    descriptor: str = Field(description="Human-readable description of what this criterion measures.")
    scale: dict[int, str] = Field(
        description="Mapping of score (0–5) to a one-line descriptor for that level.",
    )


class RubricSchema(BaseModel):
    title: str
    role: str
    criteria: list[RubricCriterion]

    @property
    def total_weight(self) -> float:
        return round(sum(c.weight for c in self.criteria), 4)


# ---------------------------------------------------------------------------
# Tool confirmation stub (output of propose_interview)
# ---------------------------------------------------------------------------

class ConfirmationStub(BaseModel):
    candidate: str
    slot: InterviewSlot
    confirmation_id: str
    status: Literal["scheduled", "failed"] = "scheduled"
    message: str = ""


# ---------------------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------------------

class RunStats(BaseModel):
    step_count: int = 0
    tool_call_count: int = 0
    duration_seconds: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Top-level run result (returned by runner.py, consumed by UI)
# ---------------------------------------------------------------------------

class RunResult(BaseModel):
    run_id: str
    jd: str
    rubric: RubricSchema
    shortlist: list[ShortlistEntry] = Field(default_factory=list)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    guardrail_status: GuardrailStatus = Field(default_factory=GuardrailStatus)
    run_stats: RunStats = Field(default_factory=RunStats)


# ---------------------------------------------------------------------------
# LangGraph agent state (TypedDict for the graph)
# ---------------------------------------------------------------------------

from typing import TypedDict, Any


class AgentState(TypedDict):
    jd: str
    rubric: RubricSchema
    candidates: dict[str, str]                  # name -> raw resume text
    profiles: dict[str, CandidateProfile]        # name -> parsed profile
    scorecards: dict[str, ScoreCard]             # name -> scorecard
    availability: dict[str, list[InterviewSlot]] # name -> slots
    shortlist: list[ShortlistEntry]
    trajectory: list[TrajectoryStep]
    guardrail_status: GuardrailStatus
    pending_approval: Optional[ShortlistEntry]
    run_id: str
    step_counter: int
    tool_call_counter: int
    current_candidate: Optional[str]             # candidate being processed right now
    candidates_done: list[str]                   # names fully processed
    approval_decision: Optional[str]             # "approved" | "rejected" | None
