# TechVest Recruitment Agent

An autonomous, multi-tool, auditable recruitment agent built with **LangGraph** + **Streamlit**.

## What it does

Given one job description and three candidate résumés, the agent:

1. **Plans its own next step** at each point (not a hard-coded pipeline).
2. **Calls tools** to parse résumés, score candidates against a rubric, check interview availability, and propose interviews.
3. **Holds state** across all three candidates in a single run.
4. Produces a **ranked, evidence-cited shortlist** with a full audit trail.
5. **Never books an interview** without explicit human approval (LangGraph interrupt gate).
6. **Resists prompt injection** planted inside one résumé (Meera's).
7. **Scores fairly** — name-swap fairness test built in.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...   # Windows: set OPENAI_API_KEY=sk-...

# 3. Run the Streamlit app
streamlit run streamlit_app.py
```

## File Layout

```
RecruitmentAgent/
├── data/
│   ├── jd.md                    # Junior AI Engineer JD
│   └── resumes/
│       ├── priya.txt            # Strong fit (INTERVIEW)
│       ├── rahul.txt            # Borderline (HOLD)
│       └── meera.txt            # Weak fit + planted injection (NOT A FIT)
├── schemas.py                   # All Pydantic models
├── rubric.py                    # Scoring rubric (35/30/20/15 weights)
├── tools.py                     # parse_resume, score_candidate, check_availability, propose_interview
├── guardrails.py                # Injection detection, fairness check
├── agent_graph.py               # LangGraph graph + human-approval interrupt
├── runner.py                    # AgentRunner class + audit log persistence
├── streamlit_app.py             # Streamlit UI entrypoint
├── audit_log/                   # Persisted JSON per run (auto-created)
└── requirements.txt
```

## Guardrails

| Guardrail | Implementation |
|---|---|
| Human-in-the-loop gate | LangGraph `interrupt_before=["schedule"]` — pauses before `propose_interview` fires |
| Step cap | `recursion_limit=25` on the graph |
| Prompt-injection defence | `guardrails.sanitize_resume_text()` strips hostile instructions before LLM sees them |
| Fairness check | Name-swap test in Tab 2 → Fairness Check panel |
| Decision audit log | Every run persisted to `audit_log/<run_id>.json` |

## Scoring Rubric

| Criterion | Weight |
|---|---|
| Python / ML Fundamentals | 35% |
| Relevant Projects | 30% |
| Hands-on Tooling | 20% |
| Communication | 15% |

Scores are 0–5; every score must cite a specific résumé line (no evidence → score 0).
