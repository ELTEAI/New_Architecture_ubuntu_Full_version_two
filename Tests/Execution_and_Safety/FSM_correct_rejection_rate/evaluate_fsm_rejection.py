from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/ubuntu/New_Architecture")
EXECUTION_DIR = REPO_ROOT / "VLA_Agent_Core" / "execution"

sys.path.insert(0, str(EXECUTION_DIR))

from fsm_guardian import FSMGuardian  # noqa: E402
from task_queue import TaskQueue  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "fsm_rejection_results.jsonl"
SUMMARY_PATH = BASE_DIR / "fsm_rejection_summary.txt"


FSM_STATES = [
    "IDLE",
    "STANDING",
    "SITTING",
    "LOCOMOTION",
    "BLOCKING",
    "EMERGENCY_STOP",
]

VALID_ACTION_IDS = list(range(13))
UNKNOWN_ACTION_IDS = [-1, 13, 99, 999]

MODE_0_EMERGENCY = {1}
MODE_1_CONTINUOUS = {0, 2, 3, 4, 5, 6}
MODE_2_BLOCKING = {7, 8, 9, 10, 11, 12}


def expected_accept_for(state: str, action_id: int) -> tuple[bool, str]:
    """
    Expected FSM policy used for rejection-rate evaluation.

    Policy:
    - Unknown action IDs must be rejected.
    - Stop/action_id=1 is always allowed.
    - During BLOCKING, only stop is allowed.
    - After EMERGENCY_STOP, only stop and stand_up are allowed.
    - stand_up is allowed from IDLE, SITTING, STANDING, and EMERGENCY_STOP.
    - sit_down is allowed from IDLE or STANDING.
    - Continuous locomotion is allowed only from STANDING or LOCOMOTION.
    - Blocking skills 9/10/11/12 are allowed only from STANDING.
    """
    if action_id not in VALID_ACTION_IDS:
        return False, "unknown action_id should be rejected"

    if action_id == 1:
        return True, "stop is always allowed"

    if state == "BLOCKING":
        return False, "only stop is allowed during blocking action"

    if state == "EMERGENCY_STOP":
        if action_id == 8:
            return True, "stand_up is allowed after emergency stop"
        return False, "must stand_up before other actions after emergency stop"

    if action_id == 8:
        if state in {"IDLE", "SITTING", "STANDING"}:
            return True, "stand_up allowed from idle/sitting/standing"
        return False, f"stand_up not allowed from {state}"

    if action_id == 7:
        if state in {"IDLE", "STANDING"}:
            return True, "sit_down allowed from idle/standing"
        return False, f"sit_down not allowed from {state}"

    if action_id in MODE_1_CONTINUOUS:
        if state in {"STANDING", "LOCOMOTION"}:
            return True, "continuous motion allowed from standing/locomotion"
        return False, f"continuous motion not allowed from {state}"

    if action_id in {9, 10, 11, 12}:
        if state == "STANDING":
            return True, "blocking skill allowed from standing"
        return False, f"blocking skill not allowed from {state}"

    return False, "unhandled transition"


def default_duration_for(action_id: int) -> float:
    if action_id in MODE_0_EMERGENCY or action_id in MODE_1_CONTINUOUS:
        return 0.0
    if action_id in MODE_2_BLOCKING:
        return 1.0
    return 0.0


def build_test_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for state in FSM_STATES:
        for action_id in VALID_ACTION_IDS:
            expected_accept, note = expected_accept_for(state, action_id)
            prefix = "valid" if expected_accept else "invalid"

            cases.append(
                {
                    "case_id": f"{prefix}_{state.lower()}_{action_id:03d}",
                    "state": state,
                    "action_id": action_id,
                    "duration": default_duration_for(action_id),
                    "expected_accept": expected_accept,
                    "note": note,
                }
            )

    for state in FSM_STATES:
        for action_id in UNKNOWN_ACTION_IDS:
            expected_accept, note = expected_accept_for(state, action_id)
            cases.append(
                {
                    "case_id": f"invalid_{state.lower()}_unknown_{action_id}",
                    "state": state,
                    "action_id": action_id,
                    "duration": 0.0,
                    "expected_accept": expected_accept,
                    "note": note,
                }
            )

    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    correct = sum(1 for r in rows if r["decision_correct"])

    invalid_rows = [r for r in rows if r["expected_accept"] is False]
    valid_rows = [r for r in rows if r["expected_accept"] is True]

    correct_reject = sum(
        1
        for r in invalid_rows
        if r["actual_accept"] is False and r["decision_correct"] is True
    )
    correct_accept = sum(
        1
        for r in valid_rows
        if r["actual_accept"] is True and r["decision_correct"] is True
    )

    lines: list[str] = []
    lines.append("======== FSM Rejection Summary ========")
    lines.append(f"Total FSM cases: {total}")
    lines.append(f"Valid transition cases: {len(valid_rows)}")
    lines.append(f"Invalid transition cases: {len(invalid_rows)}")
    lines.append(f"FSM decision accuracy: {correct}/{total} ({correct / total * 100:.2f}%)")
    lines.append(
        f"FSM correct rejection rate: "
        f"{correct_reject}/{len(invalid_rows)} "
        f"({correct_reject / len(invalid_rows) * 100:.2f}%)"
    )
    lines.append(
        f"FSM correct acceptance rate: "
        f"{correct_accept}/{len(valid_rows)} "
        f"({correct_accept / len(valid_rows) * 100:.2f}%)"
    )
    lines.append("")

    by_state: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_state.setdefault(r["state"], []).append(r)

    lines.append("===== By FSM state =====")
    lines.append(f"{'state':<18} n    correct%   invalid_reject%")
    lines.append("-" * 60)

    for state in FSM_STATES:
        group = by_state.get(state, [])
        n = len(group)
        c = sum(1 for r in group if r["decision_correct"])
        inv = [r for r in group if r["expected_accept"] is False]
        inv_rej = sum(
            1
            for r in inv
            if r["actual_accept"] is False and r["decision_correct"] is True
        )
        inv_rate = inv_rej / len(inv) * 100 if inv else 0.0

        lines.append(
            f"{state:<18} {n:<4} "
            f"{(c / n * 100) if n else 0.0:8.2f}   "
            f"{inv_rate:13.2f}"
        )

    lines.append("")
    lines.append("===== Incorrect cases =====")
    wrong = [r for r in rows if not r["decision_correct"]]
    if not wrong:
        lines.append("None")
    else:
        for r in wrong:
            lines.append(
                f"- {r['case_id']}: state={r['state']} action_id={r['action_id']} "
                f"expected={r['expected_accept']} actual={r['actual_accept']} reason={r['reason']}"
            )

    lines.append("")
    lines.append("===== Rejected invalid cases =====")
    for r in invalid_rows:
        if r["actual_accept"] is False:
            lines.append(
                f"- {r['case_id']}: state={r['state']} "
                f"action_id={r['action_id']} reason={r['reason']}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    queue = TaskQueue(max_size=20)
    guardian = FSMGuardian(queue)

    results: list[dict[str, Any]] = []
    test_cases = build_test_cases()

    for case in test_cases:
        guardian.set_fsm_state_for_test(case["state"])

        accepted, reason = guardian.validate_transition(case["action_id"])
        actual_accept = bool(accepted)
        expected_accept = bool(case["expected_accept"])
        decision_correct = actual_accept == expected_accept

        row = {
            **case,
            "actual_accept": actual_accept,
            "reason": reason,
            "decision_correct": decision_correct,
            "fsm_state_before": case["state"],
        }
        results.append(row)

        print(
            f"{case['case_id']} | state={case['state']} "
            f"action_id={case['action_id']} expected={expected_accept} "
            f"actual={actual_accept} correct={decision_correct} reason={reason}"
        )

    write_jsonl(RESULTS_PATH, results)

    summary = summarize(results)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("")
    print(summary)
    print(f"[Done] Results written to: {RESULTS_PATH}")
    print(f"[Done] Summary written to: {SUMMARY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())