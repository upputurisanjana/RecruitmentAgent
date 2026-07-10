"""
streamlit_app.py — TechVest Recruitment Agent UI

Layout:
  - Persistent sidebar: role/JD, rubric, guardrail status, run stats
  - Tab 1 (Shortlist): ranked candidate cards + scheduling/approval panel
  - Tab 2 (Trajectory & Guardrails): full reasoning trace, guardrail detail,
                                      fairness check, and audit log
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

# Load .env file before anything else reads environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set externally

import streamlit as st

# ── Page config must be FIRST Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="TechVest Recruitment Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (after page config) ─────────────────────────────────────
from schemas import RunResult, GuardrailStatus, ShortlistEntry, TrajectoryStep
from rubric import get_default_rubric, rubric_as_table, scale_as_table
from guardrails import run_fairness_check
from runner import AgentRunner, load_past_runs, load_run_result, load_jd, load_resumes

# ── Colour helpers ─────────────────────────────────────────────────────────
VERDICT_COLOURS = {
    "INTERVIEW": "#2ecc71",   # green
    "HOLD": "#f39c12",        # amber
    "NOT A FIT": "#e74c3c",   # red
}

STEP_ICONS = {
    "thought": "💭",
    "action": "🔧",
    "observation": "👁",
    "decision": "✅",
}


def verdict_badge(verdict: str) -> str:
    colour = VERDICT_COLOURS.get(verdict, "#95a5a6")
    return (
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:700;font-size:0.85rem;">{verdict}</span>'
    )


def status_dot(colour: str, label: str) -> str:
    return f'<span style="color:{colour};font-weight:600;">● {label}</span>'


# ── Session state initialisation ───────────────────────────────────────────

def _init_session():
    defaults = {
        "run_result": None,          # RunResult | None
        "runner": None,              # AgentRunner | None
        "running": False,            # bool
        "run_error": None,           # str | None
        "fairness_result": None,     # dict | None
        "approval_feedback": {},     # {candidate: "approved"|"rejected"}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar(run_result: Optional[RunResult]):
    with st.sidebar:
        st.markdown("## 🤖 TechVest Recruitment Agent")
        st.markdown("**Junior AI Engineer**")
        st.markdown("---")

        # ── 1. Job Description ──────────────────────────────────────────
        st.markdown("### 📋 Role Configuration")
        try:
            jd_text = load_jd()
        except Exception:
            jd_text = "(JD file not found)"
        with st.expander("Job Description", expanded=False):
            st.markdown(jd_text)

        # ── 2. Scoring Rubric ────────────────────────────────────────────
        st.markdown("### 📊 Scoring Rubric")
        rubric = get_default_rubric()
        rubric_rows = rubric_as_table(rubric)
        try:
            import pandas as pd
            df = pd.DataFrame(rubric_rows)[["Criterion", "Weight"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        except ImportError:
            for row in rubric_rows:
                st.markdown(f"- **{row['Criterion']}** — {row['Weight']}")

        with st.expander("0–5 Score Scale"):
            for row in scale_as_table():
                st.markdown(f"**{row['Score']}** — {row['Descriptor']}")

        # ── 3. Guardrail Status ──────────────────────────────────────────
        st.markdown("### 🛡 Guardrail Status")

        if run_result:
            gs: GuardrailStatus = run_result.guardrail_status

            # Step cap
            used = gs.steps_used
            limit = gs.steps_limit
            pct = int((used / max(limit, 1)) * 100)
            cap_colour = "#e74c3c" if used >= limit else ("#f39c12" if pct >= 70 else "#2ecc71")
            st.markdown(
                f"**Step cap:** "
                + status_dot(cap_colour, f"{used} / {limit} steps"),
                unsafe_allow_html=True,
            )
            st.progress(min(pct / 100, 1.0))

            # Human gate
            gate_map = {
                "armed": ("#3498db", "Armed"),
                "waiting_for_approval": ("#f39c12", "Waiting for approval"),
                "cleared": ("#2ecc71", "Cleared"),
                "rejected": ("#e74c3c", "Rejected by reviewer"),
            }
            gate_colour, gate_label = gate_map.get(gs.human_gate, ("#95a5a6", gs.human_gate))
            st.markdown(
                "**Human gate:** " + status_dot(gate_colour, gate_label),
                unsafe_allow_html=True,
            )

            # Injection
            if gs.injection_detected:
                st.markdown(
                    "**Injection defence:** "
                    + status_dot("#e74c3c", f"Blocked (in {gs.injection_candidate}'s résumé)"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "**Injection defence:** " + status_dot("#2ecc71", "Clean"),
                    unsafe_allow_html=True,
                )

            # Fairness
            fr = st.session_state.get("fairness_result")
            if fr is None:
                st.markdown(
                    "**Fairness check:** " + status_dot("#95a5a6", "Not yet run"),
                    unsafe_allow_html=True,
                )
            elif fr.get("passed"):
                st.markdown(
                    "**Fairness check:** " + status_dot("#2ecc71", "PASS"),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "**Fairness check:** " + status_dot("#e74c3c", "FAIL"),
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(status_dot("#95a5a6", "Run the agent to see live status"), unsafe_allow_html=True)

        # ── 4. Run Stats ─────────────────────────────────────────────────
        if run_result:
            st.markdown("---")
            st.markdown("### 📈 Last Run Stats")
            rs = run_result.run_stats
            st.caption(
                f"Steps: {rs.step_count}  |  "
                f"Tool calls: {rs.tool_call_count}  |  "
                f"Duration: {rs.duration_seconds:.1f}s"
            )
            if rs.timestamp:
                st.caption(f"Timestamp: {rs.timestamp[:19].replace('T', ' ')} UTC")
            st.caption(f"Run ID: `{run_result.run_id}`")

        # ── API Key check ─────────────────────────────────────────────────
        st.markdown("---")
        api_key = (
            os.environ.get("GITHUB_TOKEN", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            st.warning("⚠️ Set GITHUB_TOKEN, OPENROUTER_API_KEY, or OPENAI_API_KEY before running the agent.")
        else:
            if os.environ.get("GITHUB_TOKEN", ""):
                st.success("✅ GitHub token found (GitHub Models).")
            elif os.environ.get("OPENROUTER_API_KEY", ""):
                st.success("✅ OpenRouter API key found.")
            else:
                st.success("✅ API key found.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SHORTLIST
# ═══════════════════════════════════════════════════════════════════════════

def render_shortlist_tab(run_result: Optional[RunResult]):
    st.markdown("## 📋 Recruitment Agent — Shortlist")

    # ── Run Agent button ─────────────────────────────────────────────────
    col_btn, col_status = st.columns([2, 5])
    with col_btn:
        run_disabled = st.session_state["running"]
        if st.button("▶ Run Agent", type="primary", disabled=run_disabled):
            _start_run()

    with col_status:
        if st.session_state["running"]:
            st.info("⏳ Agent is reasoning… (check Tab 2 for live trajectory)")
        elif st.session_state["run_error"]:
            st.error(f"❌ Run failed: {st.session_state['run_error']}")
        elif run_result:
            rs = run_result.run_stats
            st.success(
                f"✅ Run complete — {len(run_result.shortlist)} candidates processed "
                f"in {rs.duration_seconds:.1f}s ({rs.step_count} steps, {rs.tool_call_count} tool calls)."
            )
        else:
            st.markdown("Click **Run Agent** to start.")

    st.markdown("---")

    if run_result is None:
        st.info("No results yet. Click **Run Agent** to begin.")
        return

    # ── Candidate cards ──────────────────────────────────────────────────
    if not run_result.shortlist:
        st.warning("The shortlist is empty — the run may have completed with no results.")
        return

    st.markdown(f"### Candidates (ranked by score)")

    for entry in run_result.shortlist:
        _render_candidate_card(entry)

    # ── Scheduling / approval panel ──────────────────────────────────────
    interview_entries = [e for e in run_result.shortlist if e.verdict == "INTERVIEW"]
    if interview_entries:
        st.markdown("---")
        st.markdown("### 📅 Scheduling — Human Approval Gate")

        runner: Optional[AgentRunner] = st.session_state.get("runner")
        paused = runner.is_paused() if runner else False

        for entry in interview_entries:
            _render_approval_panel(entry, runner, paused, run_result)


def _render_candidate_card(entry: ShortlistEntry):
    colour = VERDICT_COLOURS.get(entry.verdict, "#95a5a6")
    with st.container(border=True):
        col_name, col_badge = st.columns([5, 2])
        with col_name:
            st.markdown(f"### {entry.candidate}")
            st.markdown(
                f"**Weighted score: {entry.weighted_score:.2f} / 5**"
            )
        with col_badge:
            st.markdown(
                "<div style='text-align:right;margin-top:10px;'>"
                + verdict_badge(entry.verdict)
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"> {entry.justification}")

        # Scorecard breakdown
        if entry.scorecard and entry.scorecard.criteria:
            show_key = f"show_sc_{entry.candidate}"
            if show_key not in st.session_state:
                st.session_state[show_key] = False
            if st.button(f"{'▾' if st.session_state[show_key] else '▸'} View scorecard breakdown",
                         key=f"sc_btn_{entry.candidate}"):
                st.session_state[show_key] = not st.session_state[show_key]
            if st.session_state[show_key]:
                rows = [
                    {
                        "Criterion": cs.criterion,
                        "Weight": f"{int(cs.weight * 100)}%",
                        "Score (0–5)": cs.score,
                        "Evidence": cs.evidence,
                    }
                    for cs in entry.scorecard.criteria
                ]
                try:
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                except ImportError:
                    for row in rows:
                        st.markdown(
                            f"- **{row['Criterion']}** ({row['Weight']}): "
                            f"{row['Score (0–5)']}/5 — _{row['Evidence']}_"
                        )


def _render_approval_panel(
    entry: ShortlistEntry,
    runner: Optional[AgentRunner],
    paused: bool,
    run_result: RunResult,
):
    feedback = st.session_state["approval_feedback"]
    local_status = feedback.get(entry.candidate)

    with st.container(border=True):
        st.markdown(f"#### 📅 Scheduling — {entry.candidate}")

        slot = entry.proposed_slot
        if slot:
            st.markdown(f"**Proposed slot:** {slot.day} at {slot.time}")
        else:
            st.markdown("*Availability not yet checked.*")

        # Display current action_status
        if entry.action_status == "approved" or local_status == "approved":
            st.success("✅ Approved — interview scheduled.")

        elif entry.action_status == "rejected" or local_status == "rejected":
            st.error("❌ Rejected by reviewer — propose_interview was not called.")

        elif entry.action_status == "pending_approval":
            st.warning("⏳ Pending approval")

            col_approve, col_reject, _ = st.columns([2, 2, 5])
            with col_approve:
                if st.button(f"✅ Approve", key=f"approve_{entry.candidate}"):
                    _handle_approval(entry.candidate, "approved", runner, run_result)

            with col_reject:
                if st.button(f"❌ Reject", key=f"reject_{entry.candidate}"):
                    _handle_approval(entry.candidate, "rejected", runner, run_result)
        else:
            st.markdown("*No action required.*")


def _handle_approval(
    candidate: str,
    decision: str,
    runner: Optional[AgentRunner],
    run_result: RunResult,
):
    """Handle an Approve or Reject click — resumes the paused LangGraph."""
    st.session_state["approval_feedback"][candidate] = decision

    if runner and runner.is_paused():
        try:
            with st.spinner(f"Resuming agent after {decision}…"):
                updated_result = runner.resume(decision)
            st.session_state["run_result"] = updated_result
            st.session_state["running"] = False
        except Exception as exc:
            st.error(f"Resume failed: {exc}")
    else:
        # Runner already finished — update the entry in-place for display
        for e in run_result.shortlist:
            if e.candidate == candidate:
                e.action_status = decision  # type: ignore[assignment]
        st.session_state["run_result"] = run_result

    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAJECTORY & GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════

def render_trajectory_tab(run_result: Optional[RunResult]):
    st.markdown("## 🔍 Trajectory & Guardrails")

    if run_result is None:
        st.info("Run the agent first to see the reasoning trace.")
        return

    # ── 1. Trajectory trace ──────────────────────────────────────────────
    st.markdown("### 🗂 Agent Reasoning Trace")

    trajectory = run_result.trajectory
    if not trajectory:
        st.warning("Trajectory is empty.")
    else:
        # Group steps by candidate for easier navigation
        _render_trajectory(trajectory)

    st.markdown("---")

    # ── 2. Guardrail detail panel ────────────────────────────────────────
    st.markdown("### 🛡 Guardrail Detail")
    _render_guardrail_detail(run_result.guardrail_status)

    st.markdown("---")

    # ── 3. Fairness check ────────────────────────────────────────────────
    st.markdown("### ⚖️ Fairness Check — Name-Swap Test")
    _render_fairness_panel()

    st.markdown("---")

    # ── 4. Audit log ─────────────────────────────────────────────────────
    st.markdown("### 🗄 Decision Audit Log")
    _render_audit_log(run_result)


def _render_trajectory(trajectory: list[TrajectoryStep]):
    """Render the full trajectory as a vertical timeline."""
    for step in trajectory:
        icon = STEP_ICONS.get(step.type, "•")
        label = step.type.upper()

        if step.flagged:
            # Flagged step: red-bordered container
            with st.container(border=True):
                st.markdown(
                    f"🚫 **BLOCKED** — Step {step.step_index}  \n"
                    f"_{step.content}_",
                )
        elif step.type == "thought":
            st.markdown(
                f"<span style='color:#7f8c8d;'>{icon} **{label}** (step {step.step_index}):</span>  \n"
                f"<span style='color:#555;'>{step.content}</span>",
                unsafe_allow_html=True,
            )
        elif step.type in ("action", "observation"):
            tool_info = f" `{step.tool_name}`" if step.tool_name else ""
            st.markdown(
                f"{icon} **{label}**{tool_info} (step {step.step_index})"
            )
            st.code(step.content, language="text")
        elif step.type == "decision":
            st.markdown(
                f"{icon} **{label}** (step {step.step_index}):  \n"
                f"**{step.content}**"
            )
        else:
            st.markdown(f"{icon} **{label}**: {step.content}")


def _render_guardrail_detail(gs: GuardrailStatus):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Step / Iteration Cap**")
        st.progress(min(gs.steps_used / max(gs.steps_limit, 1), 1.0))
        st.markdown(f"{gs.steps_used} of {gs.steps_limit} steps used.")

        st.markdown("**Human Gate History**")
        gate_labels = {
            "armed": "🔵 Armed — no INTERVIEW candidates processed yet.",
            "waiting_for_approval": "🟡 Waiting for approval.",
            "cleared": "🟢 Cleared — interview approved by reviewer.",
            "rejected": "🔴 Rejected — interview rejected by reviewer.",
        }
        st.info(gate_labels.get(gs.human_gate, gs.human_gate))

    with col2:
        st.markdown("**Injection Defence**")
        if gs.injection_detected:
            st.error(
                f"🚫 Injection attempt detected in **{gs.injection_candidate}**'s résumé.  \n"
                f"Blocked snippet:  \n```\n{gs.injection_snippet}\n```  \n"
                f"Instruction was removed before LLM processing. Ranking is unaffected."
            )
        else:
            st.success("✅ No injection attempts detected.")

        st.markdown("**Fairness Check**")
        fr = st.session_state.get("fairness_result")
        if fr:
            if fr["passed"]:
                st.success(f"✅ PASS — {fr['details']}")
            else:
                st.error(f"❌ FAIL — {fr['details']}")
        else:
            st.info("Not yet run. Use the fairness check panel below.")


def _render_fairness_panel():
    """Name-swap fairness test panel."""
    fr = st.session_state.get("fairness_result")

    with st.container(border=True):
        st.markdown("**Fairness Check — Name Swap Test**")
        st.markdown(
            "Scores two profiles with identical relevant experience but different names.  \n"
            "Both profiles are scored under a neutral name — the LLM never sees demographic names.  \n"
            "A PASS requires delta = 0.00 (guaranteed when name bias is absent)."
        )

        if fr:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(fr["name_a"], f"{fr['score_a']:.2f} / 5")
            with col_b:
                st.metric(fr["name_b"], f"{fr['score_b']:.2f} / 5")

            if fr["passed"]:
                st.success(f"✅ PASS — Delta = {fr['delta']:.4f}")
            else:
                st.error(f"❌ FAIL — Delta = {fr['delta']:.4f}")

            st.caption(fr["details"])

        col_run, col_reset = st.columns([2, 5])
        with col_run:
            btn_label = "↺ Re-run fairness check" if fr else "▶ Run fairness check"
            if st.button(btn_label, key="fairness_btn"):
                _run_fairness_check()


def _run_fairness_check():
    """Run the name-swap fairness test and store result in session state."""
    rubric = get_default_rubric()
    # We need a score_candidate wrapper that doesn't append to trajectory
    from tools import score_candidate

    def _score_fn(profile, rubric):
        dummy_traj = []
        return score_candidate(profile, rubric, dummy_traj)

    try:
        with st.spinner("Running fairness check…"):
            result = run_fairness_check(_score_fn, rubric)
        st.session_state["fairness_result"] = result
        # Update guardrail status on the run result if present
        rr = st.session_state.get("run_result")
        if rr:
            rr.guardrail_status.fairness_check_run = True
            rr.guardrail_status.fairness_check_passed = result["passed"]
            rr.guardrail_status.fairness_score_a = result["score_a"]
            rr.guardrail_status.fairness_score_b = result["score_b"]
    except Exception as exc:
        st.error(f"Fairness check failed: {exc}")
    st.rerun()


def _render_audit_log(current_run: RunResult):
    """List past runs. Uses containers only — no nested expanders."""
    past_runs = load_past_runs()

    if not past_runs:
        st.info("No past runs found in audit_log/.")
        return

    st.markdown(f"Found **{len(past_runs)}** persisted run(s) in `audit_log/`.")

    # Use a selectbox to pick which run to inspect — avoids any nesting
    run_labels = []
    run_ids = []
    for run_info in past_runs:
        run_id = run_info["run_id"]
        ts = run_info["timestamp"][:19].replace("T", " ") if run_info["timestamp"] else "unknown"
        verdicts = run_info.get("verdicts", {})
        summary = ", ".join(f"{c}: {v}" for c, v in verdicts.items()) or "(no shortlist)"
        marker = " ← current" if run_id == current_run.run_id else ""
        run_labels.append(f"{ts} UTC  |  {summary}{marker}")
        run_ids.append(run_id)

    selected_label = st.selectbox("Select run to inspect:", run_labels, index=0)
    selected_run_id = run_ids[run_labels.index(selected_label)]

    with st.container(border=True):
        if selected_run_id == current_run.run_id:
            st.caption(f"Current run — audit_log/{selected_run_id}.json")
            st.markdown("**Shortlist:**")
            for e in current_run.shortlist:
                st.markdown(f"- {e.candidate}: **{e.verdict}** ({e.weighted_score:.2f}/5)")
            st.markdown("**Trajectory:**")
            with st.container(border=True):
                _render_trajectory(current_run.trajectory)
        else:
            past_result = load_run_result(selected_run_id)
            if past_result:
                st.caption(f"Stored at: audit_log/{selected_run_id}.json")
                st.markdown("**Shortlist:**")
                for e in past_result.shortlist:
                    st.markdown(f"- {e.candidate}: **{e.verdict}** ({e.weighted_score:.2f}/5)")
                st.markdown("**Trajectory:**")
                with st.container(border=True):
                    _render_trajectory(past_result.trajectory)
            else:
                st.warning("Could not load this run's data.")


# ═══════════════════════════════════════════════════════════════════════════
# AGENT RUN TRIGGER
# ═══════════════════════════════════════════════════════════════════════════

def _start_run():
    """Initialise and start a new agent run."""
    api_key = (
        os.environ.get("GITHUB_TOKEN", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        st.session_state["run_error"] = "Set GITHUB_TOKEN, OPENROUTER_API_KEY, or OPENAI_API_KEY."
        return

    st.session_state["running"] = True
    st.session_state["run_error"] = None
    st.session_state["approval_feedback"] = {}
    st.session_state["fairness_result"] = None

    runner = AgentRunner()
    st.session_state["runner"] = runner

    try:
        result = runner.start()
        st.session_state["run_result"] = result
        st.session_state["running"] = False
    except Exception as exc:
        st.session_state["run_error"] = str(exc)
        st.session_state["running"] = False
        st.session_state["runner"] = None

    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    run_result: Optional[RunResult] = st.session_state.get("run_result")

    # Sidebar (visible on both tabs)
    render_sidebar(run_result)

    # Tabs
    tab1, tab2 = st.tabs(["📋 Shortlist", "🔍 Trajectory & Guardrails"])

    with tab1:
        render_shortlist_tab(run_result)

    with tab2:
        render_trajectory_tab(run_result)


if __name__ == "__main__":
    main()
