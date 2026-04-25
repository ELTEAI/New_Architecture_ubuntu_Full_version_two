from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/ubuntu/New_Architecture")
EXECUTION_DIR = REPO_ROOT / "VLA_Agent_Core" / "execution"

sys.path.insert(0, str(EXECUTION_DIR))

from task_queue import TaskQueue  # noqa: E402
from fsm_guardian import FSMGuardian  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "queue_preemption_results.jsonl"
SUMMARY_PATH = BASE_DIR / "queue_preemption_summary.txt"


STOP_TASK = {"action_id": 1, "duration": 0.0}

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


TEST_SCENARIOS = [
    {
        "case_id": "preempt_short_mixed_standing",
        "initial_state": "STANDING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 5, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "short mixed queue from STANDING",
    },
    {
        "case_id": "preempt_locomotion_long_queue",
        "initial_state": "LOCOMOTION",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 3, "duration": 0.0},
            {"action_id": 4, "duration": 0.0},
            {"action_id": 5, "duration": 0.0},
            {"action_id": 6, "duration": 0.0},
            {"action_id": 7, "duration": 3.0},
            {"action_id": 8, "duration": 3.0},
            {"action_id": 12, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "long mixed queue while FSM is already in LOCOMOTION",
    },
    {
        "case_id": "preempt_sitting_recovery_plan",
        "initial_state": "SITTING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 8, "duration": 3.0},
            {"action_id": 0, "duration": 0.0},
            {"action_id": 11, "duration": 3.0},
            {"action_id": 7, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "recovery plan from SITTING is preempted",
    },
    {
        "case_id": "preempt_blocking_state_pending_queue",
        "initial_state": "BLOCKING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 10, "duration": 3.0},
            {"action_id": 12, "duration": 3.0},
            {"action_id": 0, "duration": 0.0},
            {"action_id": 1, "duration": 0.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "pending queue is preempted while FSM state is BLOCKING",
    },
    {
        "case_id": "preempt_idle_bootstrap_plan",
        "initial_state": "IDLE",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 8, "duration": 3.0},
            {"action_id": 9, "duration": 3.0},
            {"action_id": 6, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "bootstrap plan from IDLE is preempted",
    },
    {
        "case_id": "preempt_after_one_task_consumed",
        "initial_state": "STANDING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 3, "duration": 0.0},
            {"action_id": 4, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
        ],
        "consume_before_preempt": 1,
        "repeat_emergency": 1,
        "description": "simulate one task already popped before emergency preemption",
    },
    {
        "case_id": "preempt_after_two_tasks_consumed",
        "initial_state": "STANDING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 5, "duration": 0.0},
            {"action_id": 6, "duration": 0.0},
            {"action_id": 11, "duration": 3.0},
            {"action_id": 7, "duration": 3.0},
        ],
        "consume_before_preempt": 2,
        "repeat_emergency": 1,
        "description": "simulate two tasks already popped before emergency preemption",
    },
    {
        "case_id": "preempt_near_full_queue",
        "initial_state": "STANDING",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 2, "duration": 0.0},
            {"action_id": 3, "duration": 0.0},
            {"action_id": 4, "duration": 0.0},
            {"action_id": 5, "duration": 0.0},
            {"action_id": 6, "duration": 0.0},
            {"action_id": 8, "duration": 3.0},
            {"action_id": 9, "duration": 3.0},
            {"action_id": 10, "duration": 3.0},
            {"action_id": 11, "duration": 3.0},
            {"action_id": 12, "duration": 3.0},
            {"action_id": 7, "duration": 3.0},
            {"action_id": 0, "duration": 0.0},
            {"action_id": 3, "duration": 0.0},
            {"action_id": 6, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
            {"action_id": 11, "duration": 3.0},
            {"action_id": 9, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "near-full queue is cleared and replaced by stop",
    },
    {
        "case_id": "preempt_small_capacity_queue",
        "initial_state": "STANDING",
        "queue_max_size": 3,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 3, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
            {"action_id": 11, "duration": 3.0},
            {"action_id": 7, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 1,
        "description": "queue capacity pressure; only first max_size tasks are admitted, then cleared",
    },
    {
        "case_id": "preempt_repeated_emergency",
        "initial_state": "LOCOMOTION",
        "queue_max_size": 20,
        "normal_tasks": [
            {"action_id": 0, "duration": 0.0},
            {"action_id": 5, "duration": 0.0},
            {"action_id": 12, "duration": 3.0},
        ],
        "consume_before_preempt": 0,
        "repeat_emergency": 3,
        "description": "repeated emergency events should remain safe and idempotent",
    },
]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_set_fsm_state(guardian: FSMGuardian, state: str) -> None:
    if hasattr(guardian, "set_fsm_state_for_test"):
        guardian.set_fsm_state_for_test(state)


def safe_get_fsm_state(guardian: FSMGuardian, runtime_state: dict[str, Any]) -> str | None:
    if hasattr(guardian, "get_fsm_state"):
        return guardian.get_fsm_state()
    return runtime_state.get("fsm_state")


def drain_one_task_without_execution(q: TaskQueue) -> dict[str, Any]:
    task = q.pop_action()
    q.mark_done()
    return task


def perform_single_preemption_cycle(
    q: TaskQueue,
    guardian: FSMGuardian,
    cycle_index: int,
) -> dict[str, Any]:
    pending_before_clear = q.pending_count()

    t_clear_start = time.perf_counter()
    q.clear_queue()
    t_clear_done = time.perf_counter()

    pending_after_clear = q.pending_count()

    t_stop_push_start = time.perf_counter()
    q.push_sequence([STOP_TASK])
    t_stop_push_done = time.perf_counter()

    pending_after_stop_push = q.pending_count()

    stop_task = q.pop_action()

    t_stop_exec_start = time.perf_counter()
    exec_result = guardian._execute_single_action(
        stop_task.get("action_id", 1),
        stop_task.get("duration", 0.0),
    )
    t_stop_exec_done = time.perf_counter()

    q.mark_done()

    runtime_state = guardian.get_runtime_state()
    fsm_state = safe_get_fsm_state(guardian, runtime_state)

    queue_clear_correct = pending_after_clear == 0
    stop_injection_correct = pending_after_stop_push == 1 and int(stop_task.get("action_id", -1)) == 1

    stop_execution_correct = (
        bool(exec_result.get("accepted", True))
        and int(runtime_state.get("action_id", -1)) == 1
        and abs(float(runtime_state.get("vx", 999))) < 1e-9
        and abs(float(runtime_state.get("vy", 999))) < 1e-9
        and abs(float(runtime_state.get("omega", 999))) < 1e-9
    )

    if fsm_state is not None:
        stop_execution_correct = stop_execution_correct and fsm_state == "EMERGENCY_STOP"

    return {
        "cycle_index": cycle_index,
        "pending_before_clear": pending_before_clear,
        "pending_after_clear": pending_after_clear,
        "pending_after_stop_push": pending_after_stop_push,
        "stop_task": stop_task,
        "exec_result": exec_result,
        "runtime_state_after_stop": runtime_state,
        "fsm_state_after_stop": fsm_state,
        "queue_clear_correct": queue_clear_correct,
        "stop_injection_correct": stop_injection_correct,
        "stop_execution_correct": stop_execution_correct,
        "cycle_correct": queue_clear_correct and stop_injection_correct and stop_execution_correct,
        "clear_latency": t_clear_done - t_clear_start,
        "stop_push_latency": t_stop_push_done - t_stop_push_start,
        "stop_execution_latency": t_stop_exec_done - t_stop_exec_start,
    }


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    q = TaskQueue(max_size=int(case["queue_max_size"]))
    guardian = FSMGuardian(q)
    safe_set_fsm_state(guardian, case["initial_state"])

    normal_tasks = list(case["normal_tasks"])

    t_push_start = time.perf_counter()
    q.push_sequence(normal_tasks)
    t_push_done = time.perf_counter()

    pending_after_initial_push = q.pending_count()
    expected_admitted = min(len(normal_tasks), int(case["queue_max_size"]))

    consumed_tasks = []
    consume_n = int(case.get("consume_before_preempt", 0))
    for _ in range(min(consume_n, q.pending_count())):
        consumed_tasks.append(drain_one_task_without_execution(q))

    pending_before_preempt = q.pending_count()

    cycles = []
    repeat_emergency = int(case.get("repeat_emergency", 1))

    for cycle_index in range(1, repeat_emergency + 1):
        cycle = perform_single_preemption_cycle(q, guardian, cycle_index)
        cycles.append(cycle)

        # For repeated emergency cycles, inject a new ordinary pending task between cycles
        # to verify that clear_queue remains effective repeatedly.
        if cycle_index < repeat_emergency:
            q.push_sequence([{"action_id": 0, "duration": 0.0}])

    initial_admission_correct = pending_after_initial_push == expected_admitted
    consume_simulation_correct = len(consumed_tasks) == min(consume_n, expected_admitted)

    all_cycles_correct = all(c["cycle_correct"] for c in cycles)
    all_clear_correct = all(c["queue_clear_correct"] for c in cycles)
    all_injection_correct = all(c["stop_injection_correct"] for c in cycles)
    all_stop_execution_correct = all(c["stop_execution_correct"] for c in cycles)

    return {
        "case_id": case["case_id"],
        "description": case["description"],
        "initial_state": case["initial_state"],
        "queue_max_size": case["queue_max_size"],
        "normal_task_count": len(normal_tasks),
        "expected_admitted_count": expected_admitted,
        "pending_after_initial_push": pending_after_initial_push,
        "consume_before_preempt": consume_n,
        "consumed_tasks": consumed_tasks,
        "pending_before_preempt": pending_before_preempt,
        "repeat_emergency": repeat_emergency,
        "cycles": cycles,
        "initial_admission_correct": initial_admission_correct,
        "consume_simulation_correct": consume_simulation_correct,
        "queue_clear_correct": all_clear_correct,
        "stop_injection_correct": all_injection_correct,
        "stop_execution_correct": all_stop_execution_correct,
        "queue_preemption_correct": (
            initial_admission_correct
            and consume_simulation_correct
            and all_cycles_correct
        ),
        "initial_push_latency": t_push_done - t_push_start,
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


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

    admission_ok = sum(1 for r in rows if r["initial_admission_correct"])
    consume_ok = sum(1 for r in rows if r["consume_simulation_correct"])
    clear_ok = sum(1 for r in rows if r["queue_clear_correct"])
    inject_ok = sum(1 for r in rows if r["stop_injection_correct"])
    stop_ok = sum(1 for r in rows if r["stop_execution_correct"])
    overall_ok = sum(1 for r in rows if r["queue_preemption_correct"])

    all_cycles = [c for r in rows for c in r["cycles"]]
    clear_latencies = [float(c["clear_latency"]) for c in all_cycles]
    stop_push_latencies = [float(c["stop_push_latency"]) for c in all_cycles]
    stop_exec_latencies = [float(c["stop_execution_latency"]) for c in all_cycles]

    lines: list[str] = []
    lines.append("======== Queue Preemption Correctness Summary ========")
    lines.append(f"Total preemption scenarios: {n}")
    lines.append(f"Total emergency cycles: {len(all_cycles)}")
    lines.append(f"Initial queue admission correctness: {admission_ok}/{n} ({admission_ok / n * 100:.2f}%)")
    lines.append(f"Pre-consumption simulation correctness: {consume_ok}/{n} ({consume_ok / n * 100:.2f}%)")
    lines.append(f"Queue clear correctness: {clear_ok}/{n} ({clear_ok / n * 100:.2f}%)")
    lines.append(f"Stop injection correctness: {inject_ok}/{n} ({inject_ok / n * 100:.2f}%)")
    lines.append(f"Stop execution correctness: {stop_ok}/{n} ({stop_ok / n * 100:.2f}%)")
    lines.append(f"Overall queue preemption correctness: {overall_ok}/{n} ({overall_ok / n * 100:.2f}%)")
    lines.append("")
    lines.append(f"Mean queue clear latency: {mean(clear_latencies):.6f} s")
    lines.append(f"P95 queue clear latency: {percentile(clear_latencies, 0.95):.6f} s")
    lines.append(f"Mean stop injection latency: {mean(stop_push_latencies):.6f} s")
    lines.append(f"P95 stop injection latency: {percentile(stop_push_latencies, 0.95):.6f} s")
    lines.append(f"Mean stop execution latency: {mean(stop_exec_latencies):.6f} s")
    lines.append(f"P95 stop execution latency: {percentile(stop_exec_latencies, 0.95):.6f} s")
    lines.append("")

    lines.append("===== By scenario =====")
    lines.append(
        f"{'case_id':<36} {'state':<16} "
        f"{'maxQ':<5} {'tasks':<6} {'consumed':<9} {'cycles':<7} overall"
    )
    lines.append("-" * 102)

    for r in rows:
        lines.append(
            f"{r['case_id']:<36} {r['initial_state']:<16} "
            f"{r['queue_max_size']:<5} {r['normal_task_count']:<6} "
            f"{len(r['consumed_tasks']):<9} {r['repeat_emergency']:<7} "
            f"{r['queue_preemption_correct']}"
        )

    failures = [r for r in rows if not r["queue_preemption_correct"]]
    lines.append("")
    lines.append("===== Failure cases =====")
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(
                f"- {r['case_id']}: admission={r['initial_admission_correct']}, "
                f"consume={r['consume_simulation_correct']}, "
                f"clear={r['queue_clear_correct']}, "
                f"inject={r['stop_injection_correct']}, "
                f"stop_exec={r['stop_execution_correct']}"
            )
            for c in r["cycles"]:
                if not c["cycle_correct"]:
                    lines.append(
                        f"    cycle={c['cycle_index']} clear={c['queue_clear_correct']} "
                        f"inject={c['stop_injection_correct']} "
                        f"stop_exec={c['stop_execution_correct']} "
                        f"fsm={c['fsm_state_after_stop']} "
                        f"runtime={c['runtime_state_after_stop']}"
                    )

    return "\n".join(lines) + "\n"


def main() -> int:
    rows: list[dict[str, Any]] = []

    for case in TEST_SCENARIOS:
        print(f"[Run] {case['case_id']} | {case['description']}")
        result = run_case(case)
        rows.append(result)
        print(
            f"  admission={result['initial_admission_correct']} "
            f"consume={result['consume_simulation_correct']} "
            f"clear={result['queue_clear_correct']} "
            f"inject={result['stop_injection_correct']} "
            f"stop_exec={result['stop_execution_correct']} "
            f"overall={result['queue_preemption_correct']}"
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