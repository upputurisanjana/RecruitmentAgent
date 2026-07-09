# TechVest Recruitment Agent — ui.md

Visual/layout spec for the Streamlit app. Two tabs + a persistent sidebar, matching the "run panel + shortlist" and "trajectory + guardrails" reference screens from the build brief.

---

## 1. Global Layout

```
┌───────────────┬─────────────────────────────────────────────┐
│               │  [ Tab: Shortlist ]  [ Tab: Trajectory ]     │
│   SIDEBAR     │                                               │
│  (persistent  │           MAIN CONTENT AREA                  │
│   across      │        (changes per selected tab)             │
│   tabs)       │                                               │
│               │                                               │
└───────────────┴─────────────────────────────────────────────┘
```

- Framework: `st.sidebar` for the left column; `st.tabs(["Shortlist", "Trajectory & Guardrails"])` for the main area.
- Keep visual language consistent with Days 3–4 apps (same Streamlit theme) — this is a lab exercise, not a rebrand; prioritize clarity over decoration.

---

## 2. Sidebar

**Sections, top to bottom:**

1. **Role / Run config**
   - Heading: role title, e.g. "Junior AI Engineer — TechVest"
   - `st.expander("Job Description")` with the full JD text
2. **Scoring rubric**
   - Small table or bullet list: Criterion — Weight — 0–5 descriptor (use `st.dataframe` or `st.table` for the criteria+weights, `st.expander` for the full descriptor scale)
3. **Guardrail status panel** (live indicators — use `st.metric` or colored `st.badge`-style markdown per line):
   - Step cap: `12 / 25 steps used` (green while under, amber near limit, red if hit)
   - Human gate: `Armed` (blue) → `Waiting for approval` (amber) → `Cleared` (green)
   - Injection defence: `Clean` (green) or `Blocked 1 attempt (Meera)` (red)
   - Fairness check: `Not yet run` (grey) / `Pass` (green) / `Fail` (red)
4. **Last-run stats**
   - Step count, tool-call count, duration, timestamp — small caption text at the bottom.

**Color convention (reuse across sidebar + both tabs):**
| Status | Color |
|---|---|
| Good / pass / cleared | Green |
| Waiting / in progress | Amber/Yellow |
| Blocked / fail / hard stop | Red |
| Neutral / not yet run | Grey |

---

## 3. Tab 1 — Shortlist

**Reference layout (from build brief screenshot):** sidebar on the left with run config/rubric/guardrails; main area shows the ranked shortlist as a vertical stack of candidate cards.

### 3.1 Header row
- Title: "Recruitment Agent — Shortlist"
- **Run Agent** button (primary, top right or top of main area) — disabled while a run is in progress, shows a spinner (`st.spinner("Agent is reasoning…")`) during execution.

### 3.2 Candidate cards (one per candidate, ranked top to bottom by weighted score)
Each card (`st.container(border=True)`):

```
┌──────────────────────────────────────────────────────────┐
│ Priya Sharma                            [ INTERVIEW ]     │
│ Weighted score: 4.3 / 5                                    │
│                                                             │
│ "Led a 3-person ML project (Projects, line 2), matches      │
│  the JD's team requirement…"                                │
│                                                             │
│ ▸ View scorecard breakdown                                 │
└──────────────────────────────────────────────────────────┘
```

- **Verdict badge** top-right, colored pill:
  - `INTERVIEW` → green
  - `HOLD` → amber
  - `NOT A FIT` → red/grey
- **Weighted score** directly under the name, large/bold.
- **Justification** paragraph — always cites a specific résumé line/section, never a generic adjective.
- **Scorecard breakdown** — `st.expander`, containing a small table: Criterion | Weight | Score (0–5) | Evidence.

### 3.3 Scheduling / approval panel (bottom of Tab 1)
Shown only for `INTERVIEW`-verdict candidates:

```
┌──────────────────────────────────────────────────────────┐
│ Scheduling — Priya Sharma                                  │
│ Proposed slot: Thu, 11:00 AM                                │
│ Status: ⏳ Pending approval                                  │
│                                                             │
│      [ Approve ]        [ Reject ]                          │
└──────────────────────────────────────────────────────────┘
```

- Buttons call back into the backend to resume the paused agent (LangGraph `interrupt`/CrewAI `human_input`) — see `spec.md` §7 and `pages_tools_interface.md` §3.
- On Approve: status flips to `✅ Approved — interview scheduled`, `propose_interview` fires for real.
- On Reject: status flips to `❌ Rejected by reviewer`, tool never fires.
- This panel must never allow a slot to be booked without a click — no "auto-approve" default.

---

## 4. Tab 2 — Trajectory & Guardrails

**Reference layout (from build brief screenshot):** a scrollable reasoning trace on top, guardrail-status panel and fairness result below, audit log at the bottom.

### 4.1 Trajectory trace
Rendered as a vertical timeline, one entry per `TrajectoryStep`, using distinct icon/label per type:

| Type | Icon/label | Style |
|---|---|---|
| `thought` | 💭 Thought | plain text, slightly muted |
| `action` | 🔧 Action | monospace, shows tool name + args |
| `observation` | 👁 Observation | monospace/quoted, shows tool return value |
| `decision` | ✅ Decision | bold, final verdict line |
| flagged step | 🚫 Blocked | red-bordered box, explains what was caught (e.g. "Injection attempt detected in Meera's résumé — instruction ignored, ranking unaffected") |

Use `st.expander` per candidate or per step-group if the full trace is long, so a reviewer can jump to a specific candidate's trace without scrolling the whole run.

### 4.2 Guardrail-status panel (expanded version of sidebar summary)
- Step/iteration cap: numeric + limit, progress bar
- Human gate: full history — which candidates were gated, when approved/rejected, by whom (session user)
- Injection defence: the exact flagged text snippet and where in the pipeline it was caught
- Fairness check: pass/fail badge

### 4.3 Fairness check panel
```
┌──────────────────────────────────────────────────────────┐
│ Fairness Check — Name Swap Test                            │
│ Base profile: [experience identical, name = "Alex"]         │
│ Swapped profile: [same experience, name = "Priya"]          │
│                                                             │
│ Alex   → weighted score 3.8                                 │
│ Priya  → weighted score 3.8                                 │
│                                                             │
│ Result: ✅ PASS — scores match                               │
└──────────────────────────────────────────────────────────┘
```
- `st.button("Run fairness check")` to trigger live, or display last cached result with a re-run option.

### 4.4 Decision audit log
- Table/list of past runs: timestamp, run ID, candidates processed, outcome summary.
- Each row expandable/clickable to reopen that run's full trajectory (read-only replay) — satisfies the audit requirement and doubles as the base for the "replay the trajectory" stretch goal.

---

## 5. Interaction Notes

- **State:** the currently displayed `RunResult` lives in `st.session_state["run_result"]`; both tabs read from the same object so they never disagree.
- **Reruns:** Streamlit reruns the whole script on interaction — guard tool-triggering logic (Run Agent, Approve/Reject, Run fairness check) behind explicit button checks, not on every rerun.
- **Long-running agent calls:** wrap the "Run Agent" execution in `st.spinner` and, if feasible, stream intermediate trajectory steps into Tab 2 as they happen rather than only rendering after full completion — makes the "autonomous loop" visible, not just the final answer.
- **Errors:** if a tool call fails or the recursion limit is hit, surface this plainly in both the sidebar guardrail panel and as a `decision`-type trajectory entry — never fail silently.

---

## 6. Accessibility / Clarity Notes
- Don't rely on color alone for verdict/status — pair every colored badge with a text label (`INTERVIEW`, `Pending approval`, etc.), already reflected above.
- Keep justification and evidence text in full sentences, not truncated — this is the artifact a reviewer/regulator may need to read closely.
