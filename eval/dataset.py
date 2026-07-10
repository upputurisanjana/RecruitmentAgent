"""
eval/dataset.py — Task suite loader for the TechVest Recruitment Agent eval layer.

load_suite(path) -> list[EvalTask]
  Reads suite_v1.yaml (or any compatible YAML), hydrates resume texts and JD text
  from disk, and returns a fully populated list[EvalTask].

This is the single source of truth every eval layer reads from — never hand-edit
expectations inline in test code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Ensure repo root is on sys.path so schemas / rubric etc. import correctly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schemas import (  # noqa: E402
    EvalTask,
    ExpectedDecision,
    ExpectedToolCall,
    PassCriteria,
    TaskInput,
)

_DEFAULT_SUITE = Path(__file__).parent / "tasks" / "suite_v1.yaml"


def load_suite(path: str | Path = _DEFAULT_SUITE) -> list[EvalTask]:
    """
    Load an eval task suite from a YAML file.

    Hydrates:
    - jd_text from the jd_ref path
    - resume_texts dict from the resume_paths list

    Returns a list[EvalTask] ready for evaluation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Suite YAML not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks_data: list[dict] = raw.get("tasks", [])

    tasks: list[EvalTask] = []
    for td in tasks_data:
        task = _hydrate_task(td)
        tasks.append(task)

    return tasks


def _hydrate_task(td: dict) -> EvalTask:
    """Convert a raw YAML task dict into a fully-populated EvalTask."""
    raw_input = td.get("input", {})

    # ── Load JD text ──────────────────────────────────────────────────────
    jd_ref = raw_input.get("jd_ref", "data/jd.md")
    jd_path = _REPO_ROOT / jd_ref
    jd_text = jd_path.read_text(encoding="utf-8") if jd_path.exists() else ""

    # ── Load resume texts ─────────────────────────────────────────────────
    resume_texts: dict[str, str] = {}
    for rp in raw_input.get("resume_paths", []):
        resume_path = _REPO_ROOT / rp
        if resume_path.exists():
            stem = resume_path.stem  # e.g. "priya_variant"
            # Friendly display name: first word capitalised (e.g. "Priya")
            # Unless there's already a key with that name, then use full stem
            base_name = stem.split("_")[0].capitalize()
            display_name = base_name if base_name not in resume_texts else stem.replace("_", " ").title()
            resume_texts[display_name] = resume_path.read_text(encoding="utf-8")
        else:
            # File missing — record a placeholder so the task still runs
            stem = Path(rp).stem
            base_name = stem.split("_")[0].capitalize()
            resume_texts[base_name] = f"[FILE NOT FOUND: {rp}]"

    task_input = TaskInput(
        jd_ref=jd_ref,
        resume_paths=raw_input.get("resume_paths", []),
        rubric_ref=raw_input.get("rubric_ref", "default"),
        jd_text=jd_text,
        resume_texts=resume_texts,
    )

    # ── Build ExpectedToolCalls ───────────────────────────────────────────
    expected_tool_calls = [
        ExpectedToolCall(
            tool_name=tc["tool_name"],
            order_index=tc["order_index"],
            required_args=tc.get("required_args") or {},
        )
        for tc in td.get("expected_tool_calls", [])
    ]

    # ── Build ExpectedDecision ────────────────────────────────────────────
    ed_raw = td.get("expected_decision", {})
    expected_decision = ExpectedDecision(
        verdict=ed_raw.get("verdict"),
        must_trigger_human_gate=ed_raw.get("must_trigger_human_gate", False),
        must_trigger_verifier=ed_raw.get("must_trigger_verifier", False),
    )

    # ── Build PassCriteria ────────────────────────────────────────────────
    pc_raw = td.get("pass_criteria", {})
    pass_criteria = PassCriteria(
        trace_invariants_required=pc_raw.get("trace_invariants_required", []),
        tool_call_accuracy_min=float(pc_raw.get("tool_call_accuracy_min", 1.0)),
        faithfulness_min=float(pc_raw.get("faithfulness_min", 0.8)),
        relevancy_min=float(pc_raw.get("relevancy_min", 0.8)),
    )

    return EvalTask(
        id=td["id"],
        category=td["category"],
        description=td.get("description", ""),
        input=task_input,
        expected_trajectory=td.get("expected_trajectory", []),
        expected_tool_calls=expected_tool_calls,
        expected_decision=expected_decision,
        pass_criteria=pass_criteria,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    suite = load_suite()
    print(f"Loaded {len(suite)} tasks from suite_v1.yaml:")
    for t in suite:
        print(f"  [{t.id}]  category={t.category}  resumes={list(t.input.resume_texts.keys())}")
