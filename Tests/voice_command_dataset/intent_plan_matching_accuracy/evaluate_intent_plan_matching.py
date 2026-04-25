"""
Ubuntu
从 `plan_validity_rate/plan_validity_results.jsonl` 计算 intent–plan 匹配率：
将 `expected_steps` 与 `plan_actions`（经 action_id 映射为技能名）逐步对齐比较。

仅评估：`tested_for_plan_validity==True` 且 `plan_valid==True` 且
`category` 为 `normal_action` 或 `combo_action` 的行（与 plan validity 阶段一致）。

默认输入为上游脚本产物（与本目录平级的 plan_validity_rate 目录下）：
    ../plan_validity_rate/plan_validity_results.jsonl

输出写在 `intent_plan_matching_accuracy/` 本目录下：
    intent_plan_matching_results.jsonl
    intent_plan_matching_summary.txt

评估口径：
1. intent-plan matching accuracy:
   只检查动作类型和动作顺序是否与 expected_steps 匹配。

2. schema-duration accuracy:
   不使用自然语言中的 "two seconds / three seconds" 来判定连续动作 duration。
   根据 VLA_Agent_Core/schemas/robot_skill_schema.json：
   - Mode 0 emergency: action_id=1, duration 必须为 0
   - Mode 1 continuous: action_id in {0,2,3,4,5,6}, duration 必须为 0
   - Mode 2 blocking: action_id in {7,8,9,10,11,12}, duration 应在 [2,5] 秒
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# .../Tests/voice_command_dataset/intent_plan_matching_accuracy
DATASET_DIR = Path(__file__).resolve().parent
VOICE_DATASET_DIR = DATASET_DIR.parent

DEFAULT_INPUT = VOICE_DATASET_DIR / "plan_validity_rate" / "plan_validity_results.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "intent_plan_matching_results.jsonl"
DEFAULT_SUMMARY = DATASET_DIR / "intent_plan_matching_summary.txt"


# 与 VLA_Agent_Core/schemas/robot_skill_schema.json 中 0–12 约定一致
ACTION_ID_TO_NAME = {
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

ACTION_NAME_TO_ID = {v: k for k, v in ACTION_ID_TO_NAME.items()}

# robot_skill_schema.json:
# Mode 0: emergency stop, duration = 0
# Mode 1: continuous motion, duration = 0
# Mode 2: blocking action, duration in [2, 5]
MODE_0_EMERGENCY_IDS = {1}
MODE_1_CONTINUOUS_IDS = {0, 2, 3, 4, 5, 6}
MODE_2_BLOCKING_IDS = {7, 8, 9, 10, 11, 12}


# expected_steps 中可能使用的别名，统一到项目真实动作集合
ALIASES = {
    "move_forward_slow": "move_forward",
    "recover_stand": "stand_up",
    "keep_standing": "stand_up",
    "lie_down": "sit_down",

    "greeting": "greeting",
    "roll_over": "roll_over",
    "pose": "pose",
    "stretch": "stretch",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_action(name: str) -> str:
    name = str(name).strip().lower()
    return ALIASES.get(name, name)


def action_match(expected_action: str, generated_action: str) -> bool:
    expected_action = canonical_action(expected_action)
    generated_action = canonical_action(generated_action)

    if expected_action == generated_action:
        return True

    # 模糊旋转类指令：rotate / turn_around 可以由左转或右转实现
    if expected_action in {"rotate", "turn_around"} and generated_action in {"turn_left", "turn_right"}:
        return True

    return False


def parse_expected_steps(s: str) -> list[dict[str, Any]]:
    """
    Parse:
        move_forward:3
        stand_up|move_forward:3
        move_left:1|move_right:1
        move_forward:short|turn_left

    注意：
    这里保留 expected duration 仅用于诊断展示；
    主 intent-plan matching 不使用它。
    schema-duration accuracy 也只检查 generated duration 是否符合 schema。
    """
    out: list[dict[str, Any]] = []
    s = str(s or "").strip()
    if not s:
        return out

    for part in s.split("|"):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            name, dur = part.split(":", 1)
            name = canonical_action(name)
            dur = dur.strip()

            if dur == "short":
                duration: Any = "short"
            else:
                try:
                    duration = float(dur)
                except ValueError:
                    duration = None
        else:
            name = canonical_action(part)
            duration = None

        out.append({"action": name, "duration": duration})

    return out


def generated_steps_from_plan_actions(plan_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for a in plan_actions or []:
        try:
            action_id = int(a.get("action_id"))
        except Exception:
            out.append(
                {
                    "action": "unknown",
                    "duration": None,
                    "action_id": a.get("action_id"),
                }
            )
            continue

        name = ACTION_ID_TO_NAME.get(action_id, f"unknown_{action_id}")
        name = canonical_action(name)

        try:
            duration = float(a.get("duration", 0.0))
        except Exception:
            duration = None

        out.append(
            {
                "action": name,
                "duration": duration,
                "action_id": action_id,
            }
        )

    return out


def compare_steps_action_only(
    expected: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    主指标：
    只检查 action type 和 action order。
    """
    if not expected:
        return False, "empty expected_steps"

    if not generated:
        return False, "empty generated plan_actions"

    if len(expected) != len(generated):
        return False, f"step count mismatch: expected {len(expected)}, got {len(generated)}"

    for i, (e, g) in enumerate(zip(expected, generated)):
        expected_action = e["action"]
        generated_action = g["action"]

        if not action_match(expected_action, generated_action):
            return False, (
                f"step[{i}] action mismatch: "
                f"expected {expected_action}, got {generated_action}"
            )

    return True, "action_sequence_match"


def schema_duration_match(generated_action: dict[str, Any]) -> tuple[bool, str]:
    """
    根据 robot_skill_schema.json 检查 generated duration。

    Mode 0 和 Mode 1:
        duration 必须为 0。
    Mode 2:
        duration 应为 2–5 秒。
    """
    try:
        action_id = int(generated_action.get("action_id"))
        duration = float(generated_action.get("duration"))
    except Exception:
        return False, "invalid action_id or duration"

    if action_id in MODE_0_EMERGENCY_IDS or action_id in MODE_1_CONTINUOUS_IDS:
        if abs(duration - 0.0) <= 1e-6:
            return True, "schema_duration_match"
        return False, f"action_id={action_id} requires duration=0, got {duration}"

    if action_id in MODE_2_BLOCKING_IDS:
        if 2.0 <= duration <= 5.0:
            return True, "schema_duration_match"
        return False, f"action_id={action_id} requires duration in [2,5], got {duration}"

    return False, f"unknown action_id={action_id}"


def compare_steps_schema_duration(
    expected: list[dict[str, Any]],
    generated: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    辅助指标：
    检查 action type/order + generated duration 是否符合 robot_skill_schema.json。
    """
    ok, reason = compare_steps_action_only(expected, generated)
    if not ok:
        return False, reason

    for i, g in enumerate(generated):
        ok_dur, dur_reason = schema_duration_match(g)
        if not ok_dur:
            return False, f"step[{i}] {dur_reason}"

    return True, "schema_duration_match"


def summarize(rows: list[dict[str, Any]]) -> str:
    tested = [r for r in rows if r.get("tested_for_intent_match") is True]
    n = len(tested)

    m = sum(1 for r in tested if r.get("intent_plan_match") is True)
    schema_m = sum(1 for r in tested if r.get("schema_duration_match") is True)

    lines: list[str] = []
    lines.append("======== Intent-Plan Matching Summary ========")
    lines.append(f"Total input rows: {len(rows)}")
    lines.append(f"Intent-tested rows: {n}")
    lines.append(f"Intent-plan matches: {m}/{n}")
    lines.append(f"Intent-plan matching accuracy: {(m / n * 100) if n else 0:.2f}%")
    lines.append(f"Schema-duration matches: {schema_m}/{n}")
    lines.append(f"Schema-duration accuracy: {(schema_m / n * 100) if n else 0:.2f}%")
    lines.append("")

    lines.append("===== By category =====")
    lines.append(f"{'category':<20} n    match%   schema_dur%")
    lines.append("-" * 58)

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in tested:
        by_cat[str(r.get("category", ""))].append(r)

    for cat in sorted(by_cat.keys()):
        g = by_cat[cat]
        ng = len(g)
        mg = sum(1 for r in g if r.get("intent_plan_match") is True)
        sg = sum(1 for r in g if r.get("schema_duration_match") is True)
        lines.append(
            f"{cat:<20} {ng:<4} "
            f"{(mg / ng * 100) if ng else 0:7.2f}   "
            f"{(sg / ng * 100) if ng else 0:10.2f}"
        )

    failures = [r for r in tested if r.get("intent_plan_match") is not True]
    lines.append("")
    lines.append("===== Intent mismatch cases: action sequence only =====")
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')} / {r.get('category')}: "
                f"REF='{r.get('reference_text')}' | ASR='{r.get('asr_text')}' | "
                f"expected={r.get('parsed_expected_steps')} | "
                f"generated={r.get('generated_steps')} | "
                f"reason={r.get('intent_match_reason')}"
            )

    schema_failures = [
        r
        for r in tested
        if r.get("intent_plan_match") is True
        and r.get("schema_duration_match") is not True
    ]
    lines.append("")
    lines.append("===== Schema-duration mismatch cases among action-matched rows =====")
    if not schema_failures:
        lines.append("None")
    else:
        for r in schema_failures:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')} / {r.get('category')}: "
                f"REF='{r.get('reference_text')}' | ASR='{r.get('asr_text')}' | "
                f"expected={r.get('parsed_expected_steps')} | "
                f"generated={r.get('generated_steps')} | "
                f"reason={r.get('schema_duration_reason')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 plan_validity 输出中的 plan_actions 与 expected_steps 对齐评估 intent-plan 匹配。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="默认: ../plan_validity_rate/plan_validity_results.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="默认: 本目录下 intent_plan_matching_results.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="默认: 本目录下 intent_plan_matching_summary.txt",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 行，0 表示全量。")
    args = parser.parse_args()

    if not args.input.is_file():
        print(
            f"未找到输入: {args.input}\n"
            f"请先运行 plan_validity_rate/evaluate_plan_validity.py 或指定 --input。",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    results: list[dict[str, Any]] = []

    for r in rows:
        tested = (
            r.get("tested_for_plan_validity") is True
            and r.get("plan_valid") is True
            and str(r.get("category", "")) in {"normal_action", "combo_action"}
        )

        merged = {
            **r,
            "tested_for_intent_match": tested,
            "intent_plan_match": None,
            "intent_match_reason": "",
            "schema_duration_match": None,
            "schema_duration_reason": "",
            "parsed_expected_steps": [],
            "generated_steps": [],
        }

        if not tested:
            results.append(merged)
            continue

        expected_steps = parse_expected_steps(str(r.get("expected_steps", "")))
        generated_steps = generated_steps_from_plan_actions(r.get("plan_actions", []))

        match, reason = compare_steps_action_only(expected_steps, generated_steps)
        schema_match, schema_reason = compare_steps_schema_duration(expected_steps, generated_steps)

        merged.update(
            {
                "intent_plan_match": match,
                "intent_match_reason": reason,
                "schema_duration_match": schema_match,
                "schema_duration_reason": schema_reason,
                "parsed_expected_steps": expected_steps,
                "generated_steps": generated_steps,
            }
        )

        results.append(merged)

    write_jsonl(args.output, results)
    summary_text = summarize(results)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"[Done] Results written to: {args.output}")
    print(f"[Done] Summary written to: {args.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())