from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from dryrun_hardware_wrapper import (
    ROBOT_SKILL_SCHEMA_PATH,
    ActionTask,
    DryRunHardwareWrapper,
    load_schema_action_ids,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "dryrun_dispatch_results.jsonl"
SUMMARY_PATH = BASE_DIR / "dryrun_dispatch_summary.txt"


ACTION_NAMES = {
    0: "move_forward",
    1: "stop",
    2: "move_backward",
    3: "move_left",
    4: "move_right",
    5: "turn_left",
    6: "turn_right",
    7: "sit_down",
    8: "stand_up",
    9: "stretch",
    10: "roll_over",
    11: "pose",
    12: "greeting",
}


MODE_0_EMERGENCY = {1}
MODE_1_CONTINUOUS = {0, 2, 3, 4, 5, 6}
MODE_2_BLOCKING = {7, 8, 9, 10, 11, 12}


def default_duration_for_action(action_id: int) -> float:
    if action_id in MODE_0_EMERGENCY or action_id in MODE_1_CONTINUOUS:
        return 0.0
    if action_id in MODE_2_BLOCKING:
        return 3.0
    raise ValueError(f"Unknown action_id={action_id}")


def make_valid_test_tasks(trials_per_action: int = 5) -> list[ActionTask]:
    """
    Generate valid test tasks from the action_id enum in robot_skill_schema.json.
    """
    schema_action_ids = sorted(load_schema_action_ids())

    tasks: list[ActionTask] = []

    for trial in range(1, trials_per_action + 1):
        for action_id in schema_action_ids:
            tasks.append(
                ActionTask(
                    action_id=action_id,
                    duration=default_duration_for_action(action_id),
                    source=f"dryrun_trial_{trial}",
                )
            )

    return tasks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0

    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]

    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)

    if f == c:
        return xs[f]

    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    success = sum(1 for r in rows if r.get("dispatch_success") is True)
    schema_ok = sum(1 for r in rows if r.get("schema_ok") is True)

    latencies = [float(r.get("dispatch_latency", 0.0)) for r in rows]

    lines: list[str] = []
    lines.append("======== Dry-run Command Dispatch Summary ========")
    lines.append(f"Schema file: {ROBOT_SKILL_SCHEMA_PATH}")
    lines.append(f"Total tasks: {n}")
    lines.append(f"Schema-valid tasks: {schema_ok}/{n} ({schema_ok / n * 100:.2f}%)")
    lines.append(f"Dispatch success: {success}/{n} ({success / n * 100:.2f}%)")
    lines.append("")
    lines.append(f"Mean dispatch latency: {statistics.mean(latencies):.6f} s")
    lines.append(f"Median dispatch latency: {statistics.median(latencies):.6f} s")
    lines.append(f"P95 dispatch latency: {percentile(latencies, 0.95):.6f} s")
    lines.append(f"Max dispatch latency: {max(latencies):.6f} s")
    lines.append("")

    by_action: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        action_id = int(r["task"]["action_id"])
        by_action[action_id].append(r)

    lines.append("===== By action_id =====")
    lines.append(f"{'action_id':<10} {'action_name':<18} n    success%   method")
    lines.append("-" * 72)

    for action_id in sorted(by_action.keys()):
        group = by_action[action_id]
        gn = len(group)
        gs = sum(1 for r in group if r.get("dispatch_success") is True)
        methods = sorted(
            set(str(r.get("result", {}).get("method")) for r in group)
        )
        method_str = ",".join(methods)
        lines.append(
            f"{action_id:<10} {ACTION_NAMES[action_id]:<18} {gn:<4} "
            f"{gs / gn * 100:7.2f}   {method_str}"
        )

    failures = [r for r in rows if r.get("dispatch_success") is not True]
    lines.append("")
    lines.append("===== Failure cases =====")
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(
                f"- action_id={r['task']['action_id']} "
                f"action_name={r.get('action_name')} "
                f"reason={r.get('result', {}).get('message')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    wrapper = DryRunHardwareWrapper()

    tasks = make_valid_test_tasks(trials_per_action=5)

    rows: list[dict[str, Any]] = []

    for idx, task in enumerate(tasks, start=1):
        result = wrapper.dispatch(task)
        result["test_index"] = idx
        rows.append(result)

        print(
            f"[{idx}/{len(tasks)}] "
            f"action_id={task.action_id} "
            f"action_name={result.get('action_name')} "
            f"success={result.get('dispatch_success')} "
            f"method={result.get('result', {}).get('method')}"
        )

    write_jsonl(RESULTS_PATH, rows)

    summary = summarize(rows)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("")
    print(summary)
    print(f"[Done] Results written to: {RESULTS_PATH}")
    print(f"[Done] Summary written to: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())