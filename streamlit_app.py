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
from schemas import RunResult, GuardrailStatus, ShortlistEntry, TrajectoryStep, EvalReport
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
        # Eval tab
        "eval_running": False,       # bool
        "eval_error": None,          # str | None
        "last_eval_report_id": None, # str | None
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
# TAB 3 — EVALUATION & RED-TEAM
# ═══════════════════════════════════════════════════════════════════════════

def _load_eval_reports() -> list[dict]:
    """List all eval_reports/*.json files (newest first)."""
    from pathlib import Path
    reports_dir = Path(__file__).parent / "eval_reports"
    reports_dir.mkdir(exist_ok=True)
    rows = []
    for f in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            import json
            data = json.loads(f.read_text(encoding="utf-8"))
            rows.append({
                "run_id": data.get("run_id", f.stem),
                "timestamp": data.get("timestamp", ""),
                "overall_verdict": data.get("overall_verdict", "UNKNOWN"),
                "invariant_pass_rate": data.get("invariant_pass_rate", 0.0),
                "tool_call_accuracy_rate": data.get("tool_call_accuracy_rate", 0.0),
                "fairness_pass_rate": data.get("fairness_pass_rate", 0.0),
                "human_gate_fire_rate": data.get("human_gate_fire_rate", 0.0),
                "critical_findings_open": data.get("critical_findings_open", 0),
                "path": str(f),
            })
        except Exception:
            pass
    return rows


def _load_eval_report_by_id(run_id: str) -> "EvalReport | None":
    """Load an EvalReport from eval_reports/<run_id>.json."""
    from pathlib import Path
    path = Path(__file__).parent / "eval_reports" / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_eval_tab():
    """Render the Evaluation & Red-Team tab (§9 of eval_layer_integration.md)."""
    st.markdown("## 🧪 Evaluation & Red-Team")
    st.markdown(
        "This tab shows the results of the offline eval suite — "
        "10 tasks across 7 categories, graded on 4 layers, with red-team findings. "
        "It reads from `eval_reports/*.json`, not from the live single-run result."
    )

    # ── Load available reports ─────────────────────────────────────────────
    all_reports = _load_eval_reports()

    # ── Run full suite button ─────────────────────────────────────────────
    st.markdown("---")
    col_run, col_note = st.columns([3, 7])
    with col_run:
        if st.button("▶ Run full eval suite", type="primary",
                     disabled=st.session_state.get("eval_running", False)):
            _start_eval_suite()
    with col_note:
        st.caption(
            "Runs 10 full agent executions + red-team scan. "
            "This takes several minutes. Results saved to eval_reports/."
        )

    if st.session_state.get("eval_running"):
        progress_bar = st.progress(0)
        st.info("⏳ Eval suite running… check terminal for live progress.")
        # We can't stream progress in Streamlit without threads; show spinner
        with st.spinner("Running eval suite…"):
            _run_eval_suite_blocking()
        st.session_state["eval_running"] = False
        all_reports = _load_eval_reports()
        st.rerun()

    if st.session_state.get("eval_error"):
        st.error(f"❌ Eval suite failed: {st.session_state['eval_error']}")

    if not all_reports:
        st.info(
            "No eval reports found. Click **Run full eval suite** to generate one, "
            "or run `python -m eval.run_eval_suite` from the terminal."
        )
        return

    # ── Report selector ───────────────────────────────────────────────────
    st.markdown("---")
    report_labels = []
    report_ids = []
    for r in all_reports:
        ts = r["timestamp"][:19].replace("T", " ") if r["timestamp"] else "unknown"
        verdict_emoji = {"SAFE_TO_TRUST": "✅", "NEEDS_FIXES": "⚠️", "NOT_SAFE": "❌"}.get(
            r["overall_verdict"], "?"
        )
        report_labels.append(f"{verdict_emoji} {ts} — {r['overall_verdict']}  (ID: {r['run_id']})")
        report_ids.append(r["run_id"])

    selected_label = st.selectbox(
        "Select eval report to view:",
        report_labels,
        index=0,
        key="eval_report_selector",
    )
    selected_run_id = report_ids[report_labels.index(selected_label)]
    report = _load_eval_report_by_id(selected_run_id)

    if report is None:
        st.error("Could not load the selected report.")
        return

    _render_eval_report(report)

    # ── Historical comparison ─────────────────────────────────────────────
    if len(all_reports) >= 2:
        st.markdown("---")
        st.markdown("### 📊 Historical Comparison (diff two reports)")
        col_a, col_b = st.columns(2)
        with col_a:
            label_a = st.selectbox("Report A (before):", report_labels, index=min(1, len(report_labels)-1), key="diff_a")
        with col_b:
            label_b = st.selectbox("Report B (after):", report_labels, index=0, key="diff_b")

        id_a = report_ids[report_labels.index(label_a)]
        id_b = report_ids[report_labels.index(label_b)]

        if id_a != id_b:
            ra = _load_eval_report_by_id(id_a)
            rb = _load_eval_report_by_id(id_b)
            if ra and rb:
                _render_report_diff(ra, rb)


def _render_eval_report(report: "EvalReport"):
    """Render a full EvalReport — verdict banner, metrics, task table, findings."""

    # ── 1. Verdict banner ─────────────────────────────────────────────────
    verdict = report.overall_verdict
    verdict_color = {
        "SAFE_TO_TRUST": "#2ecc71",
        "NEEDS_FIXES": "#f39c12",
        "NOT_SAFE": "#e74c3c",
    }.get(verdict, "#95a5a6")
    verdict_emoji = {"SAFE_TO_TRUST": "✅", "NEEDS_FIXES": "⚠️", "NOT_SAFE": "❌"}.get(verdict, "?")

    st.markdown(
        f"""<div style="background:{verdict_color};color:white;padding:16px 20px;
        border-radius:8px;font-size:1.4rem;font-weight:700;margin-bottom:16px;">
        {verdict_emoji} {verdict}
        &nbsp;&nbsp;<span style="font-size:0.9rem;font-weight:400;opacity:0.9;">
        Suite: suite_v1 &nbsp;|&nbsp; Run: {report.run_id[:16]} &nbsp;|&nbsp;
        {report.timestamp[:19].replace("T", " ")} UTC
        </span></div>""",
        unsafe_allow_html=True,
    )

    # ── 2. Four-metric scorecard row ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Invariant pass rate", f"{report.invariant_pass_rate:.0%}")
    with c2:
        st.metric("Tool-call accuracy", f"{report.tool_call_accuracy_rate:.0%}")
    with c3:
        avg_out = sum(
            v.get("faithfulness", 0) + v.get("relevancy", 0)
            for v in report.output_scores.values()
        ) / max(len(report.output_scores) * 2, 1)
        st.metric("Avg faithfulness/relevancy", f"{avg_out:.2f}")
    with c4:
        st.metric("Fairness pass rate", f"{report.fairness_pass_rate:.0%}")

    # ── 3. Per-task table ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Per-task results (10 tasks)")

    try:
        import pandas as pd

        rows = []
        task_ids = list(dict.fromkeys(r.task_id for r in report.task_results))
        for tid in task_ids:
            t_trace = [r for r in report.task_results if r.task_id == tid and r.layer == "trace"
                       and "gate" not in r.detail.lower() and "negative" not in r.detail.lower()]
            t_tools = [r for r in report.task_results if r.task_id == tid and r.layer == "tool_calls"
                       and "[SUMMARY]" in r.detail]
            t_out   = [r for r in report.task_results if r.task_id == tid and r.layer == "output"]
            t_gate  = [r for r in report.task_results if r.task_id == tid and r.layer == "trace"
                       and "gate" in r.detail.lower()]

            out_scores = report.output_scores.get(tid, {})
            judge_score = report.judge_scores.get(tid, None)

            rows.append({
                "Task ID": tid.replace("task_0", "T").replace("task_", "T"),
                "Trace ✓✗": "✓" if (t_trace and all(r.passed for r in t_trace)) else ("✗" if t_trace else "-"),
                "Tools ✓✗": "✓" if (t_tools and t_tools[0].passed) else ("✗" if t_tools else "-"),
                "Faithf.": f"{out_scores.get('faithfulness', 0):.2f}",
                "Relev.": f"{out_scores.get('relevancy', 0):.2f}",
                "Compl.": "✓" if out_scores.get("task_completion", 0) == 1.0 else "✗",
                "Judge": f"{judge_score:.2f}" if judge_score is not None else "-",
                "Gate ✓✗": "✓" if (t_gate and t_gate[-1].passed) else ("✗" if t_gate else "-"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not render task table: {exc}")

    # ── 4. Red-team findings ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔴 Red-Team Findings")

    if not report.red_team_findings:
        st.success("No red-team findings.")
    else:
        # Sort: Critical first, then Medium, then Low
        severity_order = {"Critical": 0, "Medium": 1, "Low": 2}
        sorted_findings = sorted(
            report.red_team_findings,
            key=lambda f: (severity_order.get(f.severity, 3), not f.fixed),
        )
        for f in sorted_findings:
            color = {"Critical": "#e74c3c", "Medium": "#f39c12", "Low": "#95a5a6"}.get(f.severity, "#aaa")
            fixed_badge = " ✅ FIXED" if f.fixed else ""
            with st.container(border=True):
                st.markdown(
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:6px;font-weight:700;">{f.severity}</span> '
                    f"**{f.source}** / {f.category} / layer: `{f.broke_layer}`{fixed_badge}",
                    unsafe_allow_html=True,
                )
                st.markdown(f.description)
                if f.reproduced_by_task_id:
                    st.caption(f"Reproduced by task: {f.reproduced_by_task_id}")

    # ── 5. Governance section ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔐 Governance — Human Gate")

    gate_col, neg_col = st.columns(2)
    with gate_col:
        gate_color = "#2ecc71" if report.human_gate_fire_rate >= 1.0 else "#e74c3c"
        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:{gate_color};">'
            f'{report.human_gate_fire_rate:.0%}</div>'
            f'<div style="color:#666;">Human gate fire rate<br>'
            f'<small>(must be 100% for SAFE_TO_TRUST)</small></div>',
            unsafe_allow_html=True,
        )
    with neg_col:
        # Find the negative test result
        neg_results = [
            r for r in report.task_results
            if "negative test" in r.detail.lower() or "strong-fit" in r.detail.lower()
        ]
        neg_critical = any(
            f for f in report.red_team_findings
            if "NEGATIVE TEST" in f.description
        )
        if neg_results:
            neg_ok = neg_results[0].passed and not neg_critical
        else:
            # Infer from gate fire rate on strong_fit tasks
            neg_ok = report.human_gate_fire_rate >= 1.0 and not neg_critical

        neg_color = "#2ecc71" if neg_ok else "#e74c3c"
        neg_label = "PASS" if neg_ok else "FAIL ← CRITICAL"
        st.markdown(
            f'<div style="font-size:2rem;font-weight:700;color:{neg_color};">'
            f'{neg_label}</div>'
            f'<div style="color:#666;">Negative test<br>'
            f'<small>(strong-fit task must still pause before propose_interview)</small></div>',
            unsafe_allow_html=True,
        )

    if report.critical_findings_open > 0:
        st.error(
            f"⛔ {report.critical_findings_open} critical finding(s) open — "
            "verdict cannot be SAFE_TO_TRUST until resolved."
        )


def _render_report_diff(ra: "EvalReport", rb: "EvalReport"):
    """Render a side-by-side diff of two EvalReports."""
    try:
        import pandas as pd
        metrics = [
            ("Invariant pass rate", f"{ra.invariant_pass_rate:.1%}", f"{rb.invariant_pass_rate:.1%}"),
            ("Tool-call accuracy", f"{ra.tool_call_accuracy_rate:.1%}", f"{rb.tool_call_accuracy_rate:.1%}"),
            ("Fairness pass rate", f"{ra.fairness_pass_rate:.1%}", f"{rb.fairness_pass_rate:.1%}"),
            ("Human gate fire rate", f"{ra.human_gate_fire_rate:.1%}", f"{rb.human_gate_fire_rate:.1%}"),
            ("Critical findings open", str(ra.critical_findings_open), str(rb.critical_findings_open)),
            ("Overall verdict", ra.overall_verdict, rb.overall_verdict),
        ]
        df = pd.DataFrame(metrics, columns=["Metric", "Report A (before)", "Report B (after)"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Diff render failed: {exc}")


def _start_eval_suite():
    """Initiate eval suite run."""
    st.session_state["eval_running"] = True
    st.session_state["eval_error"] = None
    st.rerun()


def _run_eval_suite_blocking():
    """Run the eval suite synchronously (called inside a Streamlit spinner)."""
    try:
        import uuid
        from eval.dataset import load_suite
        from eval.scorecard import build_report
        run_id = f"eval_{uuid.uuid4().hex[:12]}"
        tasks = load_suite()
        report = build_report(run_id=run_id, tasks=tasks)
        st.session_state["last_eval_report_id"] = run_id
        st.session_state["eval_error"] = None
    except Exception as exc:
        st.session_state["eval_error"] = str(exc)
        st.session_state["eval_running"] = False


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
    tab1, tab2, tab3 = st.tabs([
        "📋 Shortlist",
        "🔍 Trajectory & Guardrails",
        "🧪 Evaluation & Red-Team",
    ])

    with tab1:
        render_shortlist_tab(run_result)

    with tab2:
        render_trajectory_tab(run_result)

    with tab3:
        render_eval_tab()


if __name__ == "__main__":
    main()
