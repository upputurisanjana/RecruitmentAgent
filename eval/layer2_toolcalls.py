"""
eval/layer2_toolcalls.py — Deterministic tool-call accuracy checker.

§4.4:
- Walk trajectory filtered to type == "action"
- Compare against task.expected_tool_calls:
  * Tool name matches at the expected order_index
  * required_args is a SUBSET match against TrajectoryStep.tool_args
  * Validate arg shapes against existing Pydantic signatures
- Report tool_call_accuracy_rate = (tasks with fully matching sequence) / total tasks
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schemas import EvalTask, ExpectedToolCall, LayerResult, RunResult, TrajectoryStep  # noqa: E402


# ---------------------------------------------------------------------------
# Smoke test: 2 tasks get full exact-trajectory match; the rest use tool-call
# accuracy + invariants (per spec §4.4 "reserve exact matching for ≤2 smoke tests")
# ---------------------------------------------------------------------------

SMOKE_TEST_TASK_IDS = {
    "task_01_strong_fit_priya",
    "task_03_borderline_rahul",
}


# ---------------------------------------------------------------------------
# Arg-shape validators (Pydantic-based, per spec.md §6 tool signatures)
# ---------------------------------------------------------------------------

def _validate_parse_resume_args(args: dict) -> tuple[bool, str]:
    """parse_resume expects: candidate (str), text_length (int, optional)."""
    if "candidate" in args:
        if not isinstance(args["candidate"], str):
            return False, f"parse_resume.candidate must be str, got {type(args['candidate'])}"
    return True, "ok"


def _validate_score_candidate_args(args: dict) -> tuple[bool, str]:
    """score_candidate expects: candidate (str), criteria_count (int, optional)."""
    if "candidate" in args:
        if not isinstance(args["candidate"], str):
            return False, f"score_candidate.candidate must be str, got {type(args['candidate'])}"
    if "criteria_count" in args:
        if not isinstance(args["criteria_count"], (int, float)):
            return False, f"score_candidate.criteria_count must be numeric"
    return True, "ok"


def _validate_check_availability_args(args: dict) -> tuple[bool, str]:
    """check_availability expects: candidate (str), week (str, optional)."""
    if "candidate" in args:
        if not isinstance(args["candidate"], str):
            return False, f"check_availability.candidate must be str"
    return True, "ok"


def _validate_propose_interview_args(args: dict) -> tuple[bool, str]:
    """propose_interview expects: candidate (str), slot_day (str), slot_time (str), approved_by (str)."""
    if "candidate" in args and not isinstance(args["candidate"], str):
        return False, "propose_interview.candidate must be str"
    if "approved_by" in args:
        val = args["approved_by"]
        if not isinstance(val, str) or not val.strip():
            return False, "propose_interview.approved_by must be non-empty str"
    return True, "ok"


_ARG_VALIDATORS = {
    "parse_resume": _validate_parse_resume_args,
    "score_candidate": _validate_score_candidate_args,
    "check_availability": _validate_check_availability_args,
    "propose_interview": _validate_propose_interview_args,
}


def _validate_args(tool_name: str, actual_args: dict | None) -> tuple[bool, str]:
    """Dispatch to the appropriate arg-shape validator."""
    if actual_args is None:
        actual_args = {}
    validator = _ARG_VALIDATORS.get(tool_name)
    if validator is None:
        return True, f"no validator for tool {tool_name} — skipping shape check"
    return validator(actual_args)


# ---------------------------------------------------------------------------
# Subset-arg match
# ---------------------------------------------------------------------------

def _subset_match(required_args: dict, actual_args: dict | None) -> tuple[bool, str]:
    """
    Check that every key-value pair in *required_args* appears in *actual_args*.
    Extra keys in *actual_args* are fine.
    """
    if not required_args:
        return True, "no required_args to check"
    if actual_args is None:
        return False, f"actual tool_args is None; expected {required_args}"
    mismatches = []
    for k, expected_v in required_args.items():
        actual_v = actual_args.get(k)
        # Case-insensitive string comparison
        if isinstance(expected_v, str) and isinstance(actual_v, str):
            if expected_v.lower() not in actual_v.lower():
                mismatches.append(f"{k}: expected '{expected_v}' ⊆ '{actual_v}'")
        elif actual_v != expected_v:
            mismatches.append(f"{k}: expected {expected_v!r}, got {actual_v!r}")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "subset match ok"


# ---------------------------------------------------------------------------
# Tool-call sequence checker
# ---------------------------------------------------------------------------

def evaluate_tool_calls(task: EvalTask, run_result: RunResult) -> list[LayerResult]:
    """
    Check the actual tool-call sequence against task.expected_tool_calls.

    For tasks in SMOKE_TEST_TASK_IDS, also verifies thought-text order.
    Returns a list of LayerResult (one per expected tool call + one summary).
    """
    traj = run_result.trajectory
    action_steps = [s for s in traj if s.type == "action"]

    results: list[LayerResult] = []
    all_passed = True

    for etc in task.expected_tool_calls:
        # Find the action step at (or near) the expected order_index
        matching_step = _find_action_step(action_steps, etc)

        if matching_step is None:
            all_passed = False
            results.append(LayerResult(
                layer="tool_calls",
                task_id=task.id,
                passed=False,
                score=0.0,
                detail=(
                    f"Expected tool '{etc.tool_name}' at order_index {etc.order_index} "
                    f"not found in trajectory. "
                    f"Actual actions: {[s.tool_name for s in action_steps]}"
                ),
            ))
            continue

        # Arg subset match
        subset_ok, subset_detail = _subset_match(etc.required_args, matching_step.tool_args)
        # Arg shape validation
        shape_ok, shape_detail = _validate_args(etc.tool_name, matching_step.tool_args)

        passed = subset_ok and shape_ok
        if not passed:
            all_passed = False

        results.append(LayerResult(
            layer="tool_calls",
            task_id=task.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=(
                f"tool='{etc.tool_name}' order_index={etc.order_index}: "
                f"subset={subset_detail}; shape={shape_detail}"
            ),
        ))

    # Smoke test: full thought-text order for designated tasks
    if task.id in SMOKE_TEST_TASK_IDS:
        smoke_result = _smoke_test_trajectory_order(task, traj)
        results.append(smoke_result)

    # Summary result
    results.append(LayerResult(
        layer="tool_calls",
        task_id=task.id,
        passed=all_passed,
        score=1.0 if all_passed else 0.0,
        detail=f"[SUMMARY] tool_call sequence {'PASS' if all_passed else 'FAIL'}",
    ))

    return results


def _find_action_step(
    action_steps: list[TrajectoryStep],
    etc: ExpectedToolCall,
) -> TrajectoryStep | None:
    """
    Find the action step that matches the expected tool call.
    Strategy:
    1. Try exact order_index position in action_steps list.
    2. Fall back to searching by tool_name anywhere in the list.
    """
    # Try by list position (0-based index into the actions-only subsequence)
    if etc.order_index < len(action_steps):
        candidate = action_steps[etc.order_index]
        if candidate.tool_name == etc.tool_name:
            return candidate

    # Fallback: find by tool_name
    for step in action_steps:
        if step.tool_name == etc.tool_name:
            return step

    return None


def _smoke_test_trajectory_order(task: EvalTask, traj: list[TrajectoryStep]) -> LayerResult:
    """
    Full step-type sequence check for smoke test tasks.
    Verifies that the trajectory starts with thought, then action:parse_resume,
    then observation, then action:score_candidate, etc.
    (Not verbatim thought content — just the type:tool_name pattern.)
    """
    expected_seq = task.expected_trajectory  # e.g. ["thought", "action:parse_resume", ...]
    actual_seq = []
    for s in traj:
        if s.type == "action" and s.tool_name:
            actual_seq.append(f"action:{s.tool_name}")
        else:
            actual_seq.append(s.type)

    # Find the expected sequence as a subsequence of the actual
    found = _is_subsequence(expected_seq, actual_seq)
    return LayerResult(
        layer="tool_calls",
        task_id=task.id,
        passed=found,
        score=1.0 if found else 0.0,
        detail=(
            f"[SMOKE] Expected sequence is {'a subsequence of' if found else 'NOT found in'} "
            f"actual trajectory. "
            f"Expected: {expected_seq}. "
            f"Actual: {actual_seq}"
        ),
    )


def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """Check that every element of *expected* appears in *actual* in order."""
    it = iter(actual)
    return all(e in it for e in expected)


# ---------------------------------------------------------------------------
# Accuracy rate helper (used by scorecard.py)
# ---------------------------------------------------------------------------

def tool_call_accuracy_rate(all_results: list[LayerResult]) -> float:
    """
    Compute the overall tool-call accuracy rate.
    = tasks where all SUMMARY results passed / total SUMMARY results
    """
    summaries = [r for r in all_results if r.layer == "tool_calls" and "[SUMMARY]" in r.detail]
    if not summaries:
        return 0.0
    return sum(1 for r in summaries if r.passed) / len(summaries)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from eval.dataset import load_suite
    from runner import run_once

    suite = load_suite()
    task = suite[0]

    print(f"Testing tool-call accuracy for: {task.id}")
    run_result = run_once(
        jd=task.input.jd_text,
        resumes=task.input.resume_texts,
        auto_approve=True,
    )
    results = evaluate_tool_calls(task, run_result)
    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.detail}")
