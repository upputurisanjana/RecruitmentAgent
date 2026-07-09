# TechVest Recruitment Agent — Pages, Tools & Interface Working

This document describes **what each part of the app does**, **how the pieces talk to each other**, and **what needs to be built** to make it work end to end. Read alongside `spec.md` (schemas/architecture) and `ui.md` (visual/layout spec).

---

## 1. App Structure Overview

The app is a single Streamlit application with **two views (tabs)**:

1. **Shortlist tab** — the recruiter-facing output: ranked candidates, verdicts, scores, justifications, and the scheduling approval control.
2. **Trajectory & Guardrails tab** — the auditor-facing output: the full reasoning trace and guardrail evidence.

Both tabs render from the **same underlying `RunResult` object** (see `spec.md` §8) produced by one agent run. There is a persistent **sidebar** visible on both tabs.

---

## 2. Sidebar (persistent, both tabs)

**Purpose:** shows the configuration and live safety status of the current run without needing to dig into code.

**What it needs to display:**
- **Role being hired for** — the JD title (e.g. "Junior AI Engineer, TechVest") with a short JD summary or expandable full text.
- **Scoring rubric** — each criterion, its weight, and the 0–5 descriptor scale, read directly from the `RubricSchema` object (not hard-coded in the UI — if the rubric changes in code, the sidebar must reflect it).
- **Guardrail panel (live)** — one status line per guardrail from `GuardrailStatus`:
  - Step/iteration cap: current step count vs. limit
  - Human gate: armed / waiting-for-approval / cleared
  - Injection defence: clean / flagged-and-blocked (with which candidate)
  - Fairness check: last result (pass/fail) if run
- **Last-run stats** — step count, tool-call count, run duration, timestamp.

**What needs to be done:**
- A `render_sidebar(run_result)` function that reads straight from `RunResult`/`GuardrailStatus` — never recompute or fake these values in the UI layer.
- Sidebar must update automatically after each run (Streamlit session state keyed on the latest `RunResult`).

---

## 3. Tab 1 — Shortlist View

**Purpose:** the primary deliverable — a recruiter opens this and can make a decision in under a minute.

**Layout, top to bottom:**
1. **Run header** — "Run Agent" trigger button (kicks off `runner.run(jd, resumes, rubric)`), plus run metadata (candidates processed, run time).
2. **Ranked candidate cards**, ordered by `weighted_score` descending. Each card shows:
   - Candidate name
   - **Verdict badge** — `INTERVIEW` / `HOLD` / `NOT A FIT`, color-coded (see `ui.md`)
   - **Weighted score** (0–5 or %, consistent with rubric scale)
   - **Justification** — one evidence-citing paragraph, pulled from `ShortlistEntry.justification`
   - **Per-criterion scores** — expandable breakdown showing each `CriterionScore` (criterion, weight, score, evidence line)
3. **Scheduling action panel** (bottom) — for each `INTERVIEW`-verdict candidate:
   - Proposed interview slot (from `check_availability` → `propose_interview` staging)
   - Status: `pending_approval` / `approved` / `rejected`
   - **Approve** / **Reject** buttons — this is the human-in-the-loop gate. Clicking Approve is the *only* thing that allows `propose_interview` to actually fire (LangGraph `interrupt`/resume, or CrewAI `human_input` callback).

**What needs to be done:**
- Wire the "Run Agent" button to call the backend runner and store `RunResult` in `st.session_state`.
- Render candidate cards from `RunResult.shortlist` — no hard-coded sample data in the final build.
- Approve/Reject buttons must call back into the agent/runner to resume the paused graph (LangGraph) or release the `human_input` gate (CrewAI) — not just toggle a cosmetic UI flag. The action tool (`propose_interview`) must be provably un-callable until this click happens.
- After approval, update `action_status` on the relevant `ShortlistEntry` and re-render.

---

## 4. Tab 2 — Trajectory & Guardrail View

**Purpose:** lets a reviewer audit *how* the shortlist was produced — the agent's equivalent of citations in a RAG app.

**Layout, top to bottom:**
1. **Step-by-step trajectory trace** — a scrollable/expandable list of `TrajectoryStep` entries in order:
   - `thought` — what the agent decided to do next (plain text)
   - `action` — tool name + arguments called
   - `observation` — what the tool returned
   - `decision` — the final verdict for each candidate
   - Any step where `flagged = True` (e.g. the injection attempt) rendered with a distinct warning style and an explanation of what was caught and why it was ignored.
2. **Guardrail-status panel** — same data source as the sidebar summary, but expanded: e.g. exact text of the blocked injection, the step cap hit/remaining, human-gate history.
3. **Fairness check result** — a name-swap test result: two profiles with identical relevant experience but different names, their scores side by side, pass/fail.
4. **Decision audit log** — the persisted record for this run (and prior runs if available), with a pointer to where it's stored (`audit_log/<run_id>.json`) and a note that this is what Day 7's formal evaluation will build on.

**What needs to be done:**
- Trajectory renders directly from `RunResult.trajectory` in original order — do not re-sort or summarize away steps.
- Flagged steps need a lookup against `TrajectoryStep.flagged` and `tool_args`/`content` to explain *what* was blocked, in place, next to the step — not just a generic "guardrail triggered" message.
- Fairness check panel needs its own trigger (button: "Run fairness check") that runs the two-résumé (name-swapped) test live and displays a pass/fail, OR displays the last cached result with a re-run option.
- Audit log panel reads from the `audit_log/` directory (or DB) and lists past runs, each openable to re-view its trajectory (this doubles as the "Replay the trajectory" stretch goal if implemented).

---

## 5. Tools — What Each One Does and How the UI Reflects It

| Tool | Called from | UI visibility |
|---|---|---|
| `parse_resume` | `parse` node/Analyst task | Trajectory tab: shows as an `action` step per candidate; observation shows the extracted `CandidateProfile` fields |
| `score_candidate` | `score` node/Scorer task | Trajectory tab: shows rubric applied + resulting `ScoreCard`; Shortlist tab: renders as the per-criterion breakdown |
| `check_availability` | `schedule` node (conditional) | Trajectory tab: shows returned slot options; Shortlist tab: feeds the "proposed slot" shown in the scheduling panel |
| `propose_interview` | `schedule` node, gated | Only fires after Approve click; trajectory shows `pending_approval → approved → action fired` sequence explicitly, never an instant call |

**What needs to be done for correctness (not just UI polish):**
- Each tool call in the backend must append a matching `TrajectoryStep` (action + observation) so the trajectory tab is a faithful, complete log — not reconstructed after the fact from the shortlist.
- `propose_interview`'s trajectory entry must show the approval timestamp/actor so the audit log can prove a human, not the agent, authorized the write action.

---

## 6. End-to-End Flow (what happens on "Run Agent")

1. User clicks **Run Agent** in Tab 1.
2. Backend loads JD, rubric, and the three résumés (`data/`).
3. Agent graph/crew executes the plan–act–observe loop: parses all three, scores all three, checks availability for those clearing a threshold, stages `propose_interview` calls as `pending_approval`.
4. Injection guardrail runs inline during `parse_resume`/`score_candidate` — any planted hostile instruction is flagged, logged, and excluded from influencing the score.
5. Graph pauses at the human gate before any `propose_interview` actually executes.
6. `RunResult` (partial, with `pending_approval` entries) is returned to Streamlit and stored in session state.
7. Both tabs render from this `RunResult`.
8. User reviews Tab 2 if desired, then goes to Tab 1 and clicks **Approve** for shortlisted candidates.
9. Approval resumes the graph/crew, `propose_interview` fires for real, `RunResult` is updated and re-persisted to the audit log.
10. Sidebar and both tabs re-render with final `action_status`.

---

## 7. Build Checklist (interface-facing, complements spec.md §11)

- [ ] Sidebar reads live from `RunResult`/`GuardrailStatus`, not hard-coded.
- [ ] Tab 1 renders ranked shortlist with verdict badges, scores, justifications, per-criterion breakdown.
- [ ] Tab 1 Approve/Reject buttons genuinely gate `propose_interview` (backend-verified, not cosmetic).
- [ ] Tab 2 renders the full ordered trajectory with flagged/injection steps visibly called out.
- [ ] Tab 2 fairness-check panel runs or displays the name-swap test.
- [ ] Tab 2 audit log panel lists and can reopen past runs.
- [ ] Every tool call produces a corresponding trajectory entry — no silent tool calls.
- [ ] Full run is reproducible from the persisted audit log alone.
