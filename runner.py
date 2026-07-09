"""
runner.py — Orchestrates a full agent run and returns a RunResult.

LangGraph 1.x interrupt/resume pattern:
  - graph.stream(initial_state, config)  → runs until interrupt_before fires
  - graph.get_state(config).next         → non-empty when paused
  - graph.update_state(config, patch)    → inject approval_decision before resuming
  - graph.stream(None, config)           → resume from checkpoint
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from schemas import (
    AgentState,
    GuardrailStatus,
    RunResult,
    RunStats,
)
from rubric import get_default_rubric
from agent_graph import get_graph, RECURSION_LIMIT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = DATA_DIR / "resumes"
AUDIT_DIR = BASE_DIR / "audit_log"
AUDIT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_jd() -> str:
    jd_path = DATA_DIR / "jd.md"
    if jd_path.exists():
        return jd_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"JD file not found at {jd_path}")


def load_resumes() -> dict[str, str]:
    """Return {CandidateName: raw_text} for every .txt file in resumes/."""
    resumes: dict[str, str] = {}
    if not RESUMES_DIR.exists():
        raise FileNotFoundError(f"Resumes directory not found at {RESUMES_DIR}")
    for f in sorted(RESUMES_DIR.glob("*.txt")):
        name = f.stem.capitalize()
        resumes[name] = f.read_text(encoding="utf-8")
    return resumes


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _build_initial_state(jd: str, resumes: dict[str, str], run_id: str) -> AgentState:
    return {
        "jd": jd,
        "rubric": get_default_rubric(),
        "candidates": resumes,
        "profiles": {},
        "scorecards": {},
        "availability": {},
        "shortlist": [],
        "trajectory": [],
        "guardrail_status": GuardrailStatus(steps_limit=RECURSION_LIMIT),
        "pending_approval": None,
        "run_id": run_id,
        "step_counter": 0,
        "tool_call_counter": 0,
        "current_candidate": None,
        "candidates_done": [],
        "approval_decision": None,
    }


# ---------------------------------------------------------------------------
# RunResult assembly
# ---------------------------------------------------------------------------

def _make_run_result(state: AgentState, run_id: str, start_time: float) -> RunResult:
    elapsed = time.time() - start_time
    return RunResult(
        run_id=run_id,
        jd=state.get("jd", ""),
        rubric=state.get("rubric", get_default_rubric()),
        shortlist=sorted(
            state.get("shortlist") or [],
            key=lambda e: e.weighted_score,
            reverse=True,
        ),
        trajectory=state.get("trajectory") or [],
        guardrail_status=state.get("guardrail_status") or GuardrailStatus(),
        run_stats=RunStats(
            step_count=state.get("step_counter") or 0,
            tool_call_count=state.get("tool_call_counter") or 0,
            duration_seconds=round(elapsed, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def persist_run_result(result: RunResult) -> Path:
    audit_path = AUDIT_DIR / f"{result.run_id}.json"
    audit_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return audit_path


def load_past_runs() -> list[dict]:
    summaries = []
    for f in sorted(AUDIT_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            summaries.append({
                "run_id": data.get("run_id", f.stem),
                "timestamp": data.get("run_stats", {}).get("timestamp", ""),
                "candidates": [e["candidate"] for e in data.get("shortlist", [])],
                "verdicts": {e["candidate"]: e["verdict"] for e in data.get("shortlist", [])},
                "path": str(f),
            })
        except Exception:
            pass
    return summaries


def load_run_result(run_id: str) -> Optional[RunResult]:
    audit_path = AUDIT_DIR / f"{run_id}.json"
    if not audit_path.exists():
        return None
    try:
        return RunResult.model_validate_json(audit_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Drives the LangGraph graph through one or more interrupt/resume cycles.

    Typical Streamlit flow:
        runner = AgentRunner()
        result = runner.start()             # runs to first interrupt or END
        while runner.is_paused():
            # UI shows approval panel
            result = runner.resume("approved")   # or "rejected"
        # result is now final
    """

    def __init__(self):
        self.graph, self.checkpointer = get_graph()
        self.run_id = f"run_{uuid.uuid4().hex[:12]}"
        self.thread_id = f"thread_{self.run_id}"
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }
        self._start_time: float = 0.0
        self._paused: bool = False
        self._paused_candidate: Optional[str] = None

    def start(
        self,
        jd: Optional[str] = None,
        resumes: Optional[dict[str, str]] = None,
    ) -> RunResult:
        if jd is None:
            jd = load_jd()
        if resumes is None:
            resumes = load_resumes()

        self._start_time = time.time()
        initial_state = _build_initial_state(jd, resumes, self.run_id)

        self._run_stream(initial_state)
        return self._current_result()

    def resume(self, decision: str) -> RunResult:
        """
        Resume after a human-approval interrupt.
        Injects the decision into state then continues the graph.
        """
        if not self._paused:
            raise RuntimeError("resume() called but runner is not paused.")

        # Inject the approval decision into the checkpointed state
        self.graph.update_state(
            self.config,
            {"approval_decision": decision},
        )

        self._run_stream(None)   # None = resume from checkpoint
        return self._current_result()

    def is_paused(self) -> bool:
        return self._paused

    def paused_candidate(self) -> Optional[str]:
        return self._paused_candidate

    def _run_stream(self, input_state) -> None:
        """Stream graph events and set self._paused if interrupted."""
        self._paused = False
        self._paused_candidate = None

        # Consume the stream fully
        for _ in self.graph.stream(
            input_state,
            config=self.config,
            stream_mode="values",
        ):
            pass  # state is checkpointed; we retrieve it via get_state()

        # Check whether the graph is now paused at an interrupt
        snap = self.graph.get_state(self.config)
        if snap and snap.next:
            self._paused = True
            state = snap.values
            # Find which candidate is pending
            for entry in (state.get("shortlist") or []):
                if entry.action_status == "pending_approval":
                    self._paused_candidate = entry.candidate
                    break
            if not self._paused_candidate:
                self._paused_candidate = state.get("current_candidate")

    def _current_result(self) -> RunResult:
        """Retrieve the latest state from the checkpointer and build RunResult."""
        snap = self.graph.get_state(self.config)
        state: AgentState = snap.values if snap else {}
        result = _make_run_result(state, self.run_id, self._start_time)
        if not self._paused:
            persist_run_result(result)
        return result


# ---------------------------------------------------------------------------
# Convenience one-shot runner (for tests / CLI)
# ---------------------------------------------------------------------------

def run_once(
    jd: Optional[str] = None,
    resumes: Optional[dict[str, str]] = None,
    auto_approve: bool = False,
) -> RunResult:
    runner = AgentRunner()
    result = runner.start(jd=jd, resumes=resumes)

    cycles = 0
    while runner.is_paused() and cycles < 10:
        decision = "approved" if auto_approve else "rejected"
        result = runner.resume(decision)
        cycles += 1

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    auto_approve = "--approve" in sys.argv
    print("Starting TechVest Recruitment Agent…")
    result = run_once(auto_approve=auto_approve)

    print(f"\n{'=' * 60}")
    print(f"Run ID : {result.run_id}")
    print(f"Duration: {result.run_stats.duration_seconds:.1f}s  |  "
          f"Steps: {result.run_stats.step_count}  |  "
          f"Tool calls: {result.run_stats.tool_call_count}")
    print(f"\nSHORTLIST:")
    for e in result.shortlist:
        print(f"  {e.candidate:<20} {e.verdict:<12} {e.weighted_score:.3f}/5")
    gs = result.guardrail_status
    print(f"\nGuardrails: steps {gs.steps_used}/{gs.steps_limit} | "
          f"gate: {gs.human_gate} | "
          f"injection: {'BLOCKED (' + gs.injection_candidate + ')' if gs.injection_detected else 'clean'}")
    print(f"\nAudit log: audit_log/{result.run_id}.json")
