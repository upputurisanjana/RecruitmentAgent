"""
schemas.py — All Pydantic models for the TechVest Recruitment Agent.
These are the typed contracts shared by the agent graph, tools, guardrails, and UI.
Eval layer schemas (EvalTask, LayerResult, EvalReport, etc.) are appended at the bottom.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
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


# ===========================================================================
# EVAL LAYER SCHEMAS  (eval_layer_integration.md §2)
# ===========================================================================


class TaskInput(BaseModel):
    """Input for a single evaluation task."""
    jd_ref: str = "data/jd.md"                 # path to JD file, relative to repo root
    resume_paths: list[str] = Field(default_factory=list)   # paths to resume txt files
    rubric_ref: str = "default"                 # "default" or path to a custom rubric yaml
    # Derived text fields (populated by dataset.load_suite at load time)
    jd_text: str = ""
    resume_texts: dict[str, str] = Field(default_factory=dict)  # candidate name -> raw text

    def with_swapped_name(self) -> "TaskInput":
        """Return a copy with names swapped for the fairness sweep."""
        import copy
        swapped = copy.deepcopy(self)
        items = list(swapped.resume_texts.items())
        if len(items) >= 2:
            name_a, text_a = items[0]
            name_b, text_b = items[1]
            # Swap names in the text and keys
            swapped.resume_texts = {
                name_b: text_a.replace(name_a, name_b),
                name_a: text_b.replace(name_b, name_a),
            }
        return swapped


class ExpectedToolCall(BaseModel):
    """Expected tool call for tool-call accuracy checking."""
    tool_name: str
    order_index: int                            # 0-based position in the action sequence
    required_args: dict[str, Any] = Field(default_factory=dict)   # subset check


class ExpectedDecision(BaseModel):
    """Expected outcome for a task."""
    verdict: Optional[Literal[
        "INTERVIEW", "HOLD", "NOT A FIT", "ESCALATE", "REJECT_RETRY", "OUT_OF_SCOPE"
    ]] = None
    must_trigger_human_gate: bool = False
    must_trigger_verifier: bool = False


class PassCriteria(BaseModel):
    """Per-layer pass thresholds for a task."""
    trace_invariants_required: list[str] = Field(default_factory=list)
    tool_call_accuracy_min: float = 1.0
    faithfulness_min: float = 0.8
    relevancy_min: float = 0.8


class EvalTask(BaseModel):
    """Single evaluation task — the unit of work for the eval suite."""
    id: str
    category: Literal[
        "strong_fit", "borderline_verifier", "weak_fit",
        "injection", "missing_field", "out_of_scope", "conflicting_tools"
    ]
    description: str = ""
    input: TaskInput
    expected_trajectory: list[str] = Field(default_factory=list)
    expected_tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    expected_decision: ExpectedDecision = Field(default_factory=ExpectedDecision)
    pass_criteria: PassCriteria = Field(default_factory=PassCriteria)


class LayerResult(BaseModel):
    """Result from one layer check on one task."""
    layer: Literal["trace", "tool_calls", "output"]
    task_id: str
    passed: bool
    score: Optional[float] = None
    detail: str = ""


class RedTeamFinding(BaseModel):
    """A single red-team finding, normalised from Promptfoo or Giskard."""
    source: Literal["promptfoo", "giskard", "manual"]
    category: Literal["hijacking", "injection", "excessive_agency", "tool_misuse", "looping"]
    severity: Literal["Critical", "Medium", "Low"]
    description: str
    broke_layer: Literal["trace", "tool_calls", "output"]
    reproduced_by_task_id: Optional[str] = None
    fixed: bool = False


class EvalReport(BaseModel):
    """Full evaluation report — aggregates all layers + red-team findings."""
    run_id: str
    timestamp: str
    task_results: list[LayerResult] = Field(default_factory=list)
    tool_call_accuracy_rate: float = 0.0
    invariant_pass_rate: float = 0.0
    judge_scores: dict[str, float] = Field(default_factory=dict)        # task_id -> score
    output_scores: dict[str, dict[str, float]] = Field(default_factory=dict)  # task_id -> {f, r, tc}
    fairness_pass_rate: float = 0.0
    red_team_findings: list[RedTeamFinding] = Field(default_factory=list)
    human_gate_fire_rate: float = 0.0
    critical_findings_open: int = 0
    overall_verdict: Literal["SAFE_TO_TRUST", "NEEDS_FIXES", "NOT_SAFE"] = "NOT_SAFE"
