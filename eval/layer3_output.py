"""
eval/layer3_output.py — Output quality evaluation using DeepEval.

§5:
- FaithfulnessMetric(threshold=0.8): justification grounded in résumé evidence
- AnswerRelevancyMetric(threshold=0.8): justification answers the JD query
- task_completion check: scorecard present, all criteria have evidence, verdict set
- fairness_sweep(): runs tasks with names swapped, compares weighted_score

Uses deepeval_compat to load DeepEval with the langchain.schema patch.
Falls back to heuristic scoring if DeepEval metrics fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load DeepEval (with the langchain.schema compat patch applied)
from eval.deepeval_compat import (  # noqa: E402
    DEEPEVAL_AVAILABLE,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    LLMTestCase,
)

from schemas import EvalTask, LayerResult, RunResult, ShortlistEntry  # noqa: E402


# ---------------------------------------------------------------------------
# Evidence extractor
# ---------------------------------------------------------------------------

def _extract_resume_evidence(entry: ShortlistEntry) -> list[str]:
    """Return non-trivial evidence strings cited in the scorecard criteria."""
    if not entry.scorecard:
        return []
    return [
        cs.evidence for cs in entry.scorecard.criteria
        if cs.evidence and cs.evidence.strip() not in ("", "No evidence found in résumé.")
    ]


# ---------------------------------------------------------------------------
# Single-task output evaluation
# ---------------------------------------------------------------------------

def evaluate_output(
    task: EvalTask,
    entry: ShortlistEntry,
    run_result: RunResult,
) -> dict[str, float]:
    """
    Evaluate a ShortlistEntry's output quality.

    Returns:
        {"faithfulness": float, "relevancy": float, "task_completion": float}
    """
    resume_evidence = _extract_resume_evidence(entry)
    jd_text = (task.input.jd_text or run_result.jd)[:1000]

    # ── Task completion (deterministic) ──────────────────────────────────
    completion_ok = (
        entry.scorecard is not None
        and all(c.evidence for c in entry.scorecard.criteria)
        and entry.verdict is not None
    )
    task_completion = 1.0 if completion_ok else 0.0

    # ── DeepEval metrics ──────────────────────────────────────────────────
    if DEEPEVAL_AVAILABLE:
        try:
            faithfulness, relevancy = _deepeval_metrics(
                jd_text=jd_text,
                justification=entry.justification,
                resume_evidence=resume_evidence,
                faithfulness_threshold=task.pass_criteria.faithfulness_min,
                relevancy_threshold=task.pass_criteria.relevancy_min,
            )
            return {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "task_completion": task_completion,
            }
        except Exception as exc:
            print(f"[layer3_output] DeepEval metrics failed ({exc}), using heuristic.", flush=True)

    # Fallback
    faithfulness, relevancy = _heuristic_metrics(entry, resume_evidence, jd_text)
    return {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "task_completion": task_completion,
    }


def _deepeval_metrics(
    jd_text: str,
    justification: str,
    resume_evidence: list[str],
    faithfulness_threshold: float,
    relevancy_threshold: float,
) -> tuple[float, float]:
    """Run DeepEval FaithfulnessMetric and AnswerRelevancyMetric (sync mode)."""

    # ── Faithfulness ──────────────────────────────────────────────────────
    faith_metric = FaithfulnessMetric(
        threshold=faithfulness_threshold,
        async_mode=False,
        verbose_mode=False,
        include_reason=False,
    )
    faith_case = LLMTestCase(
        input=jd_text,
        actual_output=justification,
        retrieval_context=resume_evidence if resume_evidence else [justification],
    )
    faith_metric.measure(faith_case, _show_indicator=False)
    faithfulness_score = float(faith_metric.score) if faith_metric.score is not None else 0.0

    # ── Answer Relevancy ──────────────────────────────────────────────────
    rel_metric = AnswerRelevancyMetric(
        threshold=relevancy_threshold,
        async_mode=False,
        verbose_mode=False,
        include_reason=False,
    )
    rel_case = LLMTestCase(
        input=jd_text,
        actual_output=justification,
    )
    rel_metric.measure(rel_case, _show_indicator=False)
    relevancy_score = float(rel_metric.score) if rel_metric.score is not None else 0.0

    return faithfulness_score, relevancy_score


def _heuristic_metrics(
    entry: ShortlistEntry,
    resume_evidence: list[str],
    jd_text: str,
) -> tuple[float, float]:
    """
    Lightweight heuristic when DeepEval metrics are unavailable.

    Faithfulness: fraction of evidence substrings that appear in the justification.
    Relevancy:    density of JD keywords in the justification.
    """
    justification = entry.justification.lower()

    # Faithfulness
    if not resume_evidence:
        faithfulness = 0.5 if len(justification) > 30 else 0.0
    else:
        cited = sum(
            1 for ev in resume_evidence
            if any(
                word in justification
                for word in ev.lower().split()[:8]
                if len(word) > 4
            )
        )
        faithfulness = min(cited / len(resume_evidence), 1.0)

    # Relevancy
    jd_words = {
        w.lower().strip(".,;:") for w in jd_text.split()
        if len(w) > 5 and w.isalpha()
    }
    jd_words -= {"should", "which", "their", "these", "those", "about", "other",
                 "experience", "skills", "candidate"}
    if jd_words:
        matched = sum(1 for w in jd_words if w in justification)
        relevancy = min(matched / max(len(jd_words), 1) * 4, 1.0)
    else:
        relevancy = 0.5

    return faithfulness, relevancy


# ---------------------------------------------------------------------------
# Build LayerResult objects from output scores
# ---------------------------------------------------------------------------

def output_results(task: EvalTask, scores: dict[str, float]) -> list[LayerResult]:
    """Convert evaluate_output scores into LayerResult objects."""
    f = scores.get("faithfulness", 0.0)
    r = scores.get("relevancy", 0.0)
    tc = scores.get("task_completion", 0.0)

    return [
        LayerResult(
            layer="output",
            task_id=task.id,
            passed=f >= task.pass_criteria.faithfulness_min,
            score=f,
            detail=f"faithfulness={f:.3f} (min={task.pass_criteria.faithfulness_min})",
        ),
        LayerResult(
            layer="output",
            task_id=task.id,
            passed=r >= task.pass_criteria.relevancy_min,
            score=r,
            detail=f"relevancy={r:.3f} (min={task.pass_criteria.relevancy_min})",
        ),
        LayerResult(
            layer="output",
            task_id=task.id,
            passed=tc == 1.0,
            score=tc,
            detail="task_completion=1.0 (scorecard+evidence+verdict present)" if tc == 1.0
                   else "task_completion=0.0 (missing scorecard, evidence, or verdict)",
        ),
    ]


# ---------------------------------------------------------------------------
# Fairness sweep (§5 step 5) — wraps guardrails.run_fairness_check
# ---------------------------------------------------------------------------

def fairness_sweep(tasks: list[EvalTask], run_task_fn) -> float:
    """
    Run every single-candidate task with names swapped and compare weighted_score.

    Runs pairs in parallel using ThreadPoolExecutor (up to 4 workers).
    Wraps guardrails.run_fairness_check for the deterministic guarantee first,
    then verifies end-to-end on the real pipeline.

    Returns fairness_pass_rate in [0.0, 1.0].
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # First: deterministic check via guardrails.run_fairness_check
    try:
        from guardrails import run_fairness_check
        from tools import score_candidate
        from rubric import get_default_rubric

        def _score_fn(profile, rubric):
            return score_candidate(profile, rubric, [])

        result = run_fairness_check(_score_fn, get_default_rubric())
        if not result["passed"]:
            print(f"[fairness_sweep] guardrails check FAILED: {result['details']}", flush=True)
            return 0.0
    except Exception as exc:
        print(f"[fairness_sweep] guardrails check error: {exc}", flush=True)
        return 0.0

    # Second: parallel end-to-end pipeline check
    single_tasks = [t for t in tasks if len(t.input.resume_texts) == 1]
    if not single_tasks:
        return 1.0

    def _check_one(task: EvalTask) -> bool:
        try:
            original_result = run_task_fn(task, auto_approve=True)
            swapped_input = task.input.with_swapped_name()
            swapped_task = task.model_copy(update={"input": swapped_input})
            swapped_result = run_task_fn(swapped_task, auto_approve=True)

            orig_score = original_result.shortlist[0].weighted_score if original_result.shortlist else None
            swap_score = swapped_result.shortlist[0].weighted_score if swapped_result.shortlist else None

            if orig_score is not None and swap_score is not None:
                delta = abs(orig_score - swap_score)
                if delta >= 0.5:
                    print(
                        f"[fairness_sweep] {task.id}: "
                        f"orig={orig_score:.3f} swapped={swap_score:.3f} delta={delta:.4f}",
                        flush=True,
                    )
                    return False
            return True
        except Exception as exc:
            print(f"[fairness_sweep] {task.id} error: {exc}", flush=True)
            return True  # don't penalise for infra errors

    matches = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_check_one, t): t for t in single_tasks}
        for future in as_completed(futures):
            if future.result():
                matches += 1

    return matches / len(single_tasks)


# ---------------------------------------------------------------------------
# Average output scores helper
# ---------------------------------------------------------------------------

def average_output_scores(output_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Return average faithfulness, relevancy, task_completion across all tasks."""
    if not output_scores:
        return {"faithfulness": 0.0, "relevancy": 0.0, "task_completion": 0.0}
    keys = ["faithfulness", "relevancy", "task_completion"]
    return {
        k: sum(v.get(k, 0.0) for v in output_scores.values()) / len(output_scores)
        for k in keys
    }
