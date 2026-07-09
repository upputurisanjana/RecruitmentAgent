"""
agent_graph.py — LangGraph 1.x graph: nodes, conditional edges, human-approval interrupt.

Graph topology:
    START
      └─► route_node          (picks next candidate or signals done)
            ├─► parse_node    (parse_resume tool)
            │     └─► score_node   (score_candidate tool)
            │             └─► decide_node  (assigns verdict)
            │                     ├─► [INTERVIEW] ─► avail_node  (check_availability)
            │                     │                      └─► schedule_node  ← interrupt_before here
            │                     │                              └─► mark_done_node
            │                     └─► [other] ─► mark_done_node
            │                                         └─► route_node  (loop)
            └─► [all done] ─► END

Human-approval gate:
    interrupt_before=["schedule_node"] pauses the graph before that node executes.
    The Streamlit UI:
      1. Reads the paused state and shows the approval panel.
      2. Calls runner.resume("approved" | "rejected") which:
         a. Updates state with the decision via graph.update_state().
         b. Calls graph.stream(None, config) to continue.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, START, END  # type: ignore
from langgraph.checkpoint.memory import MemorySaver  # type: ignore

from schemas import (
    AgentState,
    GuardrailStatus,
    InterviewSlot,
    ShortlistEntry,
    TrajectoryStep,
)
from guardrails import check_step_cap
from tools import (
    parse_resume,
    score_candidate,
    check_availability,
    propose_interview,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECURSION_LIMIT = 25
INTERVIEW_THRESHOLD = 3.5
HOLD_THRESHOLD = 2.0


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _thought(trajectory: list[TrajectoryStep], content: str) -> None:
    trajectory.append(
        TrajectoryStep(step_index=len(trajectory), type="thought", content=content)
    )


def _decision(trajectory: list[TrajectoryStep], content: str) -> None:
    trajectory.append(
        TrajectoryStep(step_index=len(trajectory), type="decision", content=content)
    )


# ---------------------------------------------------------------------------
# Node: route_node
# ---------------------------------------------------------------------------

def route_node(state: AgentState) -> AgentState:
    """Pick the next unprocessed candidate or signal completion."""
    done = set(state.get("candidates_done") or [])
    all_names = list(state["candidates"].keys())
    remaining = [n for n in all_names if n not in done]

    state["guardrail_status"].steps_used = state.get("step_counter") or 0

    if remaining:
        next_candidate = remaining[0]
        state["current_candidate"] = next_candidate
        _thought(
            state["trajectory"],
            f"Next candidate to process: {next_candidate}. "
            f"Remaining after this: {remaining[1:] or 'none'}.",
        )
    else:
        state["current_candidate"] = None
        _thought(state["trajectory"], "All candidates processed. Finalising shortlist.")

    state["step_counter"] = (state.get("step_counter") or 0) + 1
    return state


# ---------------------------------------------------------------------------
# Node: parse_node
# ---------------------------------------------------------------------------

def parse_node(state: AgentState) -> AgentState:
    name = state.get("current_candidate")
    if not name:
        return state

    if check_step_cap(state["guardrail_status"], state.get("step_counter") or 0):
        _decision(state["trajectory"], f"⚠️ Step cap reached. Aborting.")
        state["current_candidate"] = None
        return state

    _thought(state["trajectory"], f"Parsing résumé for {name}.")

    profile = parse_resume(
        candidate_name=name,
        resume_text=state["candidates"][name],
        trajectory=state["trajectory"],
        guardrail_status=state["guardrail_status"],
    )
    state["profiles"][name] = profile
    state["tool_call_counter"] = (state.get("tool_call_counter") or 0) + 1
    state["step_counter"] = (state.get("step_counter") or 0) + 1
    state["guardrail_status"].steps_used = state["step_counter"]
    return state


# ---------------------------------------------------------------------------
# Node: score_node
# ---------------------------------------------------------------------------

def score_node(state: AgentState) -> AgentState:
    name = state.get("current_candidate")
    if not name or name not in state["profiles"]:
        return state

    _thought(state["trajectory"], f"Scoring {name} against the rubric.")

    scorecard = score_candidate(
        profile=state["profiles"][name],
        rubric=state["rubric"],
        trajectory=state["trajectory"],
    )
    state["scorecards"][name] = scorecard
    state["tool_call_counter"] = (state.get("tool_call_counter") or 0) + 1
    state["step_counter"] = (state.get("step_counter") or 0) + 1
    state["guardrail_status"].steps_used = state["step_counter"]
    return state


# ---------------------------------------------------------------------------
# Node: decide_node
# ---------------------------------------------------------------------------

def decide_node(state: AgentState) -> AgentState:
    name = state.get("current_candidate")
    if not name or name not in state["scorecards"]:
        return state

    scorecard = state["scorecards"][name]
    total = scorecard.weighted_total

    if total >= INTERVIEW_THRESHOLD:
        verdict = "INTERVIEW"
    elif total >= HOLD_THRESHOLD:
        verdict = "HOLD"
    else:
        verdict = "NOT A FIT"

    # Evidence-citing justification
    top = [cs.evidence for cs in scorecard.criteria if cs.score >= 3 and cs.evidence]
    if top:
        justification = (
            f"Weighted score {total:.2f}/5 → {verdict}. "
            f"Key evidence: {top[0]}"
            + (f"; {top[1]}" if len(top) > 1 else "") + "."
        )
    else:
        weak = [cs.evidence for cs in scorecard.criteria if cs.evidence]
        justification = (
            f"Weighted score {total:.2f}/5 → {verdict}. "
            f"Insufficient evidence for the role's core requirements. "
            + (f"Best signal: {weak[0]}" if weak else "No strong evidence found.")
        )

    entry = ShortlistEntry(
        candidate=name,
        verdict=verdict,  # type: ignore[arg-type]
        weighted_score=total,
        justification=justification,
        scorecard=scorecard,
        proposed_slot=None,
        action_status="not_applicable" if verdict != "INTERVIEW" else "pending_approval",
    )

    shortlist = [e for e in (state.get("shortlist") or []) if e.candidate != name]
    shortlist.append(entry)
    state["shortlist"] = shortlist

    if verdict == "INTERVIEW":
        state["pending_approval"] = entry
        state["guardrail_status"].human_gate = "waiting_for_approval"

    _decision(
        state["trajectory"],
        f"Decision for {name}: {verdict} (score {total:.3f}/5). {justification}",
    )

    state["step_counter"] = (state.get("step_counter") or 0) + 1
    state["guardrail_status"].steps_used = state["step_counter"]
    return state


# ---------------------------------------------------------------------------
# Node: avail_node  (check availability — runs before the interrupt)
# ---------------------------------------------------------------------------

def avail_node(state: AgentState) -> AgentState:
    """Check interview availability. Runs before the human-approval interrupt."""
    name = state.get("current_candidate")
    if not name:
        return state

    _thought(state["trajectory"], f"Checking calendar availability for {name}.")

    slots = check_availability(
        candidate_name=name,
        week="next",
        trajectory=state["trajectory"],
    )
    availability = dict(state.get("availability") or {})
    availability[name] = slots
    state["availability"] = availability
    state["tool_call_counter"] = (state.get("tool_call_counter") or 0) + 1

    proposed_slot = slots[0] if slots else InterviewSlot(day="TBD", time="TBD")

    # Stage the proposed slot on the shortlist entry
    shortlist = list(state.get("shortlist") or [])
    for entry in shortlist:
        if entry.candidate == name:
            entry.proposed_slot = proposed_slot
            entry.action_status = "pending_approval"
            break
    state["shortlist"] = shortlist

    _thought(
        state["trajectory"],
        f"Proposed slot for {name}: {proposed_slot.day} {proposed_slot.time}. "
        f"⏸ Pausing for human approval before propose_interview fires.",
    )

    state["step_counter"] = (state.get("step_counter") or 0) + 1
    state["guardrail_status"].steps_used = state["step_counter"]
    return state


# ---------------------------------------------------------------------------
# Node: schedule_node  ← graph PAUSES BEFORE this node (interrupt_before)
# The Streamlit UI resumes after updating state["approval_decision"].
# ---------------------------------------------------------------------------

def schedule_node(state: AgentState) -> AgentState:
    """
    Fire propose_interview if approved, skip if rejected.
    The graph is interrupted BEFORE this node by interrupt_before=["schedule_node"].
    The runner injects approval_decision into state before resuming.
    """
    name = state.get("current_candidate")
    if not name:
        return state

    decision = state.get("approval_decision") or "rejected"

    # Find the proposed slot from availability
    slots = (state.get("availability") or {}).get(name, [])
    proposed_slot = slots[0] if slots else InterviewSlot(day="TBD", time="TBD")

    shortlist = list(state.get("shortlist") or [])
    for entry in shortlist:
        if entry.candidate == name:
            if decision == "approved":
                entry.action_status = "approved"
                state["guardrail_status"].human_gate = "cleared"
                confirmation = propose_interview(
                    candidate_name=name,
                    slot=proposed_slot,
                    approval_actor="recruiter_session_user",
                    trajectory=state["trajectory"],
                )
                _decision(
                    state["trajectory"],
                    f"✅ Interview confirmed for {name}: {proposed_slot.day} {proposed_slot.time}. "
                    f"ID: {confirmation.confirmation_id}.",
                )
            else:
                entry.action_status = "rejected"
                state["guardrail_status"].human_gate = "rejected"
                _decision(
                    state["trajectory"],
                    f"❌ Interview rejected by reviewer for {name}. "
                    f"propose_interview was NOT called.",
                )
            break

    state["shortlist"] = shortlist
    state["approval_decision"] = None  # reset for next candidate
    state["tool_call_counter"] = (state.get("tool_call_counter") or 0) + 1
    state["step_counter"] = (state.get("step_counter") or 0) + 1
    state["guardrail_status"].steps_used = state["step_counter"]
    return state


# ---------------------------------------------------------------------------
# Node: mark_done_node
# ---------------------------------------------------------------------------

def mark_done_node(state: AgentState) -> AgentState:
    name = state.get("current_candidate")
    if name:
        done = list(state.get("candidates_done") or [])
        if name not in done:
            done.append(name)
        state["candidates_done"] = done
    state["current_candidate"] = None
    return state


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def _after_route(state: AgentState) -> str:
    return "parse" if state.get("current_candidate") else "end"


def _after_decide(state: AgentState) -> str:
    name = state.get("current_candidate")
    if not name:
        return "mark_done"
    entry = next((e for e in (state.get("shortlist") or []) if e.candidate == name), None)
    return "avail" if (entry and entry.verdict == "INTERVIEW") else "mark_done"


# ---------------------------------------------------------------------------
# Build & compile
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("route_node", route_node)
    builder.add_node("parse_node", parse_node)
    builder.add_node("score_node", score_node)
    builder.add_node("decide_node", decide_node)
    builder.add_node("avail_node", avail_node)
    builder.add_node("schedule_node", schedule_node)
    builder.add_node("mark_done_node", mark_done_node)

    builder.add_edge(START, "route_node")

    builder.add_conditional_edges(
        "route_node",
        _after_route,
        {"parse": "parse_node", "end": END},
    )

    builder.add_edge("parse_node", "score_node")
    builder.add_edge("score_node", "decide_node")

    builder.add_conditional_edges(
        "decide_node",
        _after_decide,
        {"avail": "avail_node", "mark_done": "mark_done_node"},
    )

    builder.add_edge("avail_node", "schedule_node")   # interrupt fires here
    builder.add_edge("schedule_node", "mark_done_node")
    builder.add_edge("mark_done_node", "route_node")

    checkpointer = MemorySaver()
    compiled = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["schedule_node"],  # pause BEFORE schedule_node
    )
    return compiled, checkpointer


_GRAPH: Any = None
_CHECKPOINTER: Any = None


def get_graph():
    global _GRAPH, _CHECKPOINTER
    if _GRAPH is None:
        _GRAPH, _CHECKPOINTER = build_graph()
    return _GRAPH, _CHECKPOINTER
