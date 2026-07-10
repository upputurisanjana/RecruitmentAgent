"""
eval/redteam_giskard.py — Giskard vulnerability scan wrapper.

§6.2 of eval_layer_integration.md.

HARD BLOCKER: Giskard has no release compatible with Python 3.13.
All released versions require Python <3.13:
  Latest Python 3.12-compatible version: giskard 2.19.2 (Requires-Python <3.13,>=3.9)
  This environment: Python 3.13.3

This file implements the full giskard.Model wrapper and normalises findings
into RedTeamFinding objects.  The scan_and_collect() function returns a
stub list with a documented "BLOCKED" finding when Giskard cannot be imported,
so the rest of the eval pipeline runs cleanly without Giskard.

TO UNBLOCK: run this on Python 3.12 (e.g. pyenv, venv, or Docker):
    pip install giskard==2.19.2
    python -m eval.redteam_giskard

OR upgrade to Giskard's next release that supports Python 3.13.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schemas import RedTeamFinding  # noqa: E402


# ---------------------------------------------------------------------------
# Giskard availability check
# ---------------------------------------------------------------------------

_GISKARD_AVAILABLE = False
try:
    import giskard  # type: ignore
    _GISKARD_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# crew_predict — the function giskard.Model wraps
# ---------------------------------------------------------------------------

def crew_predict(df: "pd.DataFrame") -> list[str]:
    """
    Predict function for giskard.Model.

    Takes a DataFrame with columns: [resume_text, candidate_name].
    Returns a list of verdict strings ("INTERVIEW" / "HOLD" / "NOT A FIT").
    """
    from runner import run_once  # lazy import so file is importable without runner

    results = []
    for _, row in df.iterrows():
        resume_text = str(row.get("resume_text", ""))
        candidate_name = str(row.get("candidate_name", "Candidate"))
        try:
            jd = (_REPO_ROOT / "data" / "jd.md").read_text(encoding="utf-8")
            run_result = run_once(
                jd=jd,
                resumes={candidate_name: resume_text},
                auto_approve=False,
            )
            verdict = run_result.shortlist[0].verdict if run_result.shortlist else "NO_RESULT"
        except Exception as exc:
            verdict = f"ERROR: {exc}"
        results.append(verdict)
    return results


def _build_eval_dataset() -> "pd.DataFrame":
    """Build a small dataset from the eval suite for Giskard scanning."""
    from eval.dataset import load_suite
    suite = load_suite()
    rows = []
    for task in suite:
        for name, text in task.input.resume_texts.items():
            rows.append({
                "resume_text": text[:2000],
                "candidate_name": name,
                "task_id": task.id,
                "category": task.category,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Normalise giskard scan results → list[RedTeamFinding]
# ---------------------------------------------------------------------------

_GISKARD_CATEGORY_MAP: dict[str, str] = {
    "Prompt Injection": "injection",
    "Harmful Content Generation": "hijacking",
    "Excessive Agency": "excessive_agency",
    "Hallucination and Misinformation": "hijacking",
    "Data Leakage": "tool_misuse",
    "Robustness Issues": "looping",
}

_GISKARD_SEVERITY_MAP: dict[str, str] = {
    "major": "Critical",
    "medium": "Medium",
    "minor": "Low",
    "": "Medium",
}


def _normalise_giskard_result(issue: Any, task_id: str | None = None) -> RedTeamFinding:
    """Convert a single Giskard ScanResult issue into a RedTeamFinding."""
    raw_category = getattr(issue, "group", "") or ""
    raw_severity = getattr(issue, "level", "") or ""
    description = getattr(issue, "description", str(issue))

    category = _GISKARD_CATEGORY_MAP.get(raw_category, "hijacking")
    severity = _GISKARD_SEVERITY_MAP.get(raw_severity.lower(), "Medium")

    # Assign broke_layer based on category
    if category in ("injection", "hijacking"):
        broke_layer = "trace"
    elif category == "excessive_agency":
        broke_layer = "tool_calls"
    else:
        broke_layer = "output"

    return RedTeamFinding(
        source="giskard",
        category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        description=description,
        broke_layer=broke_layer,  # type: ignore[arg-type]
        reproduced_by_task_id=task_id,
        fixed=False,
    )


# ---------------------------------------------------------------------------
# Main scan function
# ---------------------------------------------------------------------------

def scan_and_collect(run_id: str | None = None) -> list[RedTeamFinding]:
    """
    Run a Giskard vulnerability scan on the recruitment crew and return
    normalised RedTeamFinding objects.

    If Giskard is unavailable (Python 3.13 blocker), returns a single
    BLOCKED finding documenting the reason.
    """
    if not _GISKARD_AVAILABLE:
        return [
            RedTeamFinding(
                source="giskard",
                category="injection",
                severity="Low",
                description=(
                    "BLOCKED: Giskard is not installed. "
                    "Reason: No giskard release supports Python 3.13. "
                    "Latest compatible version is giskard 2.19.2 (Python <3.13). "
                    "To run this scan: use Python 3.12 and `pip install giskard==2.19.2`."
                ),
                broke_layer="trace",
                reproduced_by_task_id=None,
                fixed=False,
            )
        ]

    try:
        df = _build_eval_dataset()
        giskard_dataset = giskard.Dataset(  # type: ignore[name-defined]
            df=df,
            target=None,
            name="TechVest Eval Suite",
            cat_columns=["candidate_name", "task_id", "category"],
        )

        model = giskard.Model(  # type: ignore[name-defined]
            model=crew_predict,
            model_type="text_generation",
            name="TechVest Recruitment Crew",
            description=(
                "Multi-agent résumé screening crew with a human-approval gate. "
                "Parses résumés, scores against a rubric, and recommends INTERVIEW / HOLD / NOT A FIT."
            ),
            feature_names=["resume_text", "candidate_name"],
        )

        scan_results = giskard.scan(model, dataset=giskard_dataset)  # type: ignore[name-defined]

        # Persist raw results
        if run_id:
            out_path = _REPO_ROOT / "eval_reports" / f"giskard_scan_{run_id}.json"
            try:
                scan_results.to_json(str(out_path))
                print(f"[redteam_giskard] Saved scan to {out_path}", flush=True)
            except Exception:
                pass

        findings: list[RedTeamFinding] = []
        for issue in scan_results.issues:
            findings.append(_normalise_giskard_result(issue))

        if not findings:
            print("[redteam_giskard] Scan complete — no issues found.", flush=True)

        return findings

    except Exception as exc:
        print(f"[redteam_giskard] Scan failed: {exc}", flush=True)
        return [
            RedTeamFinding(
                source="giskard",
                category="injection",
                severity="Low",
                description=f"Giskard scan failed at runtime: {exc}",
                broke_layer="trace",
                fixed=False,
            )
        ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Giskard available: {_GISKARD_AVAILABLE}")
    findings = scan_and_collect(run_id="test")
    for f in findings:
        print(f"  [{f.severity}] {f.source} / {f.category}: {f.description[:120]}")
