"""
Ubuntu
对 emergency_routing 结果中应走 planner 的样本，用 `PlannerClient.plan(asr_text)` 做规划并校验
ActionTask 列表是否可执行（与 orchestrator 使用的 `VLA_Pipeline` 封装一致）。

默认输入为 `emergency_routing_accuracy/evaluate_emergency_routing.py` 产出的
`emergency_routing_results.jsonl`；需本机 vLLM（见 VLA_Agent_Core/config.yaml）已就绪。
默认输出 `plan_validity_results.jsonl` / `plan_validity_summary.txt` 与脚本同目录（本文件夹）。

用法示例：
    python evaluate_plan_validity.py --limit 3
    python evaluate_plan_validity.py --input /path/to/emergency_routing_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# .../New_Architecture/Tests/voice_command_dataset/plan_validity_rate
DATASET_DIR = Path(__file__).resolve().parent
VOICE_DATASET_DIR = DATASET_DIR.parent
REPO_ROOT = VOICE_DATASET_DIR.parent.parent  # .../New_Architecture
VLA_PIPELINE_DIR = REPO_ROOT / "VLA_Pipeline"

sys.path.insert(0, str(VLA_PIPELINE_DIR))

from src.cognition.planner_client import PlannerClient  # noqa: E402


# 与 evaluate_emergency_routing.py 默认输出一致
DEFAULT_INPUT = VOICE_DATASET_DIR / "emergency_routing_accuracy" / "emergency_routing_results.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "plan_validity_results.jsonl"
DEFAULT_SUMMARY = DATASET_DIR / "plan_validity_summary.txt"

# VLABrainPlanner 使用的全局配置（与 VLA_Pipeline/config/pipeline.yaml 中 planner.config_path 一致）
DEFAULT_PLANNER_CONFIG = REPO_ROOT / "VLA_Agent_Core" / "config.yaml"


ALLOWED_ACTION_IDS = set(range(0, 100))  # 先宽松；最好之后改成你项目真实 action_id 集合


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def action_to_dict(a: Any) -> dict[str, Any]:
    if isinstance(a, dict):
        return a

    out = {}
    for name in ("action_id", "duration", "source", "mode", "vx", "vy", "omega"):
        if hasattr(a, name):
            out[name] = getattr(a, name)
    return out


def validate_actions(actions: list[Any]) -> tuple[bool, str, list[dict[str, Any]]]:
    if not isinstance(actions, list):
        return False, "actions is not a list", []

    if len(actions) == 0:
        return False, "actions is empty", []

    normalized = []

    for i, action in enumerate(actions):
        d = action_to_dict(action)

        if "action_id" not in d:
            return False, f"action[{i}] missing action_id", normalized

        if "duration" not in d:
            return False, f"action[{i}] missing duration", normalized

        try:
            action_id = int(d["action_id"])
        except Exception:
            return False, f"action[{i}] action_id is not int", normalized

        try:
            duration = float(d["duration"])
        except Exception:
            return False, f"action[{i}] duration is not numeric", normalized

        if action_id not in ALLOWED_ACTION_IDS:
            return False, f"action[{i}] action_id out of allowed range: {action_id}", normalized

        if duration < 0:
            return False, f"action[{i}] duration is negative: {duration}", normalized

        d["action_id"] = action_id
        d["duration"] = duration
        normalized.append(d)

    return True, "valid", normalized


def summarize(rows: list[dict[str, Any]]) -> str:
    tested = [r for r in rows if r.get("tested_for_plan_validity") is True]
    n = len(tested)
    valid = sum(1 for r in tested if r.get("plan_valid") is True)

    lines = []
    lines.append("======== Plan Validity Summary ========")
    lines.append(f"Total input rows: {len(rows)}")
    lines.append(f"Planner-tested rows: {n}")
    lines.append(f"Valid plans: {valid}/{n}")
    lines.append(f"Plan validity rate: {(valid / n * 100) if n else 0:.2f}%")
    lines.append("")

    # category breakdown
    cats = sorted(set(str(r.get("category", "")) for r in tested))
    lines.append("===== By category =====")
    lines.append(f"{'category':<20} n    valid%   mean_planning_s")
    lines.append("-" * 58)
    for c in cats:
        g = [r for r in tested if str(r.get("category", "")) == c]
        ng = len(g)
        vg = sum(1 for r in g if r.get("plan_valid") is True)
        mean_lat = sum(float(r.get("planning_latency", 0.0)) for r in g) / ng if ng else 0.0
        lines.append(f"{c:<20} {ng:<4} {(vg/ng*100) if ng else 0:7.2f}   {mean_lat:.3f}")

    failures = [r for r in tested if r.get("plan_valid") is not True]
    lines.append("")
    lines.append("===== Invalid plan cases =====")
    if not failures:
        lines.append("None")
    else:
        for r in failures:
            lines.append(
                f"- {r.get('sample_id')} / {r.get('voice')} / {r.get('category')}: "
                f"ASR='{r.get('asr_text')}' | reason={r.get('plan_valid_reason')}"
            )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--planner-config", type=Path, default=DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.planner_config.is_file():
        print(
            f"未找到 planner 配置: {args.planner_config}\n"
            f"请指定 --planner-config 或检查 VLA_Agent_Core 是否已克隆到仓库内。",
            file=sys.stderr,
        )
        return 2
    if not args.input.is_file():
        print(
            f"未找到输入 JSONL: {args.input}\n"
            f"请先运行 emergency_routing_accuracy/evaluate_emergency_routing.py 或传 --input。",
            file=sys.stderr,
        )
        return 2

    rows = load_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    planner = PlannerClient(config_path=str(args.planner_config))

    results = []

    for idx, row in enumerate(rows, start=1):
        category = str(row.get("category", ""))
        expected_route = str(row.get("expected_route", ""))
        predicted_route = str(row.get("predicted_route", ""))

        # 只测应该进入 planner 并生成动作计划的 normal/combo 样本
        should_test = (
            predicted_route == "planner"
            and expected_route == "planner"
            and category in {"normal_action", "combo_action"}
        )

        merged = {
            **row,
            "tested_for_plan_validity": should_test,
            "plan_valid": None,
            "plan_valid_reason": "",
            "planning_latency": 0.0,
            "plan_actions": [],
            "planner_error": "",
        }

        if not should_test:
            results.append(merged)
            continue

        text = str(row.get("asr_text", "") or "").strip()
        print(f"[{idx}/{len(rows)}] Planning: {row.get('sample_id')} / {row.get('voice')} | {text}")

        if not text:
            merged.update(
                {
                    "plan_valid": False,
                    "plan_valid_reason": "empty asr_text",
                }
            )
            results.append(merged)
            continue

        try:
            t0 = time.perf_counter()
            plan = planner.plan(text)
            t1 = time.perf_counter()

            actions = list(getattr(plan, "actions", []) or [])
            valid, reason, normalized_actions = validate_actions(actions)

            merged.update(
                {
                    "plan_valid": valid,
                    "plan_valid_reason": reason,
                    "planning_latency": round(t1 - t0, 6),
                    "plan_actions": normalized_actions,
                    "plan_sequence_name": str(getattr(plan, "sequence_name", "")),
                }
            )

            print(f"  valid={valid} | reason={reason} | latency={t1 - t0:.3f}s")

        except Exception as e:
            merged.update(
                {
                    "plan_valid": False,
                    "plan_valid_reason": "planner exception",
                    "planner_error": repr(e),
                }
            )
            print(f"  [Error] {repr(e)}")

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