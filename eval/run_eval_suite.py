"""
eval/run_eval_suite.py -- CLI entry point for the TechVest Recruitment Agent eval suite.

Usage:
    python -m eval.run_eval_suite                   # full 10-task suite
    python -m eval.run_eval_suite --task task_01_strong_fit_priya  # single task
    python -m eval.run_eval_suite --fast            # skip fairness sweep + red-team

Exits:
    0  if overall_verdict == SAFE_TO_TRUST
    1  otherwise (prints a summary of what failed)

Output:
    eval_reports/<run_id>.json
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the TechVest Recruitment Agent eval suite."
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Run only the task with this ID (default: run all 10 tasks).",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Path to a custom suite YAML (default: eval/tasks/suite_v1.yaml).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip fairness sweep and red-team scan (faster, less thorough).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run ID (default: auto-generated).",
    )
    args = parser.parse_args()

    run_id = args.run_id or f"eval_{uuid.uuid4().hex[:12]}"

    # Load dataset
    from eval.dataset import load_suite
    suite_path = args.suite or None
    try:
        if suite_path:
            tasks = load_suite(suite_path)
        else:
            tasks = load_suite()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Filter to single task if requested
    if args.task:
        tasks = [t for t in tasks if t.id == args.task]
        if not tasks:
            print(f"ERROR: No task found with id '{args.task}'", file=sys.stderr)
            return 1

    print(f"\nTechVest Recruitment Agent -- Eval Suite")
    print(f"Run ID : {run_id}")
    print(f"Tasks  : {len(tasks)}")
    if args.fast:
        print("Mode   : FAST (fairness sweep + red-team disabled)")
    print()

    t_start = time.time()

    # Monkey-patch if --fast
    if args.fast:
        import eval.layer3_output as l3
        import eval.redteam_giskard as rg
        _orig_fairness = l3.fairness_sweep
        _orig_giskard = rg.scan_and_collect
        l3.fairness_sweep = lambda tasks, run_fn: 1.0
        rg.scan_and_collect = lambda run_id=None: []

    try:
        from eval.scorecard import build_report
        report = build_report(run_id=run_id, tasks=tasks)
    finally:
        if args.fast:
            l3.fairness_sweep = _orig_fairness
            rg.scan_and_collect = _orig_giskard

    elapsed = time.time() - t_start

    # Print result summary
    verdict = report.overall_verdict
    verdict_symbol = {
        "SAFE_TO_TRUST": "[OK]",
        "NEEDS_FIXES": "[WARN]",
        "NOT_SAFE": "[FAIL]",
    }.get(verdict, "?")

    print(f"\n{'='*62}")
    print(f" {verdict_symbol}  OVERALL VERDICT: {verdict}")
    print(f"{'='*62}")
    print(f"  Invariant pass rate    : {report.invariant_pass_rate:.1%}")
    print(f"  Tool-call accuracy     : {report.tool_call_accuracy_rate:.1%}")
    print(f"  Human gate fire rate   : {report.human_gate_fire_rate:.1%}  (must be 100%)")
    print(f"  Fairness pass rate     : {report.fairness_pass_rate:.1%}")
    print(f"  Critical findings open : {report.critical_findings_open}")
    print(f"  Elapsed                : {elapsed:.1f}s")
    print(f"  Report                 : eval_reports/{run_id}.json")

    # Per-task summary table
    print(f"\n  Per-task summary:")
    print(f"  {'Task ID':<42} {'Trace':>6} {'Tools':>6} {'Output':>7} {'Gate':>5}")
    print(f"  {'-'*42} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")

    task_ids = [t.id for t in tasks]
    for tid in task_ids:
        t_trace = [
            r for r in report.task_results
            if r.task_id == tid and r.layer == "trace"
            and "gate" not in r.detail.lower()
            and "negative" not in r.detail.lower()
        ]
        t_tools = [
            r for r in report.task_results
            if r.task_id == tid and r.layer == "tool_calls"
            and "[SUMMARY]" in r.detail
        ]
        t_out = [r for r in report.task_results if r.task_id == tid and r.layer == "output"]
        t_gate = [
            r for r in report.task_results
            if r.task_id == tid and r.layer == "trace"
            and "gate" in r.detail.lower()
        ]

        trace_ok = "OK" if (t_trace and all(r.passed for r in t_trace)) else ("FAIL" if t_trace else "-")
        tools_ok = "OK" if (t_tools and t_tools[0].passed) else ("FAIL" if t_tools else "-")
        out_ok   = "OK" if (t_out and all(r.passed for r in t_out)) else ("FAIL" if t_out else "-")
        gate_ok  = "OK" if (t_gate and t_gate[-1].passed) else ("FAIL" if t_gate else "-")

        print(f"  {tid:<42} {trace_ok:>6} {tools_ok:>6} {out_ok:>7} {gate_ok:>5}")

    # Show failures
    failed = [r for r in report.task_results if not r.passed]
    if failed:
        print(f"\n  Failed checks ({len(failed)}):")
        for r in failed[:20]:
            print(f"    [{r.layer}] {r.task_id}: {r.detail[:100]}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more (see report JSON)")

    # Show critical findings
    criticals = [f for f in report.red_team_findings if f.severity == "Critical" and not f.fixed]
    if criticals:
        print(f"\n  CRITICAL findings that block SAFE_TO_TRUST:")
        for f in criticals:
            print(f"    [{f.source}] {f.category}: {f.description[:120]}")

    print()

    if verdict == "SAFE_TO_TRUST":
        print("Result: SAFE_TO_TRUST -- exiting 0.")
        return 0
    else:
        print(f"Result: {verdict} -- exiting 1.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
