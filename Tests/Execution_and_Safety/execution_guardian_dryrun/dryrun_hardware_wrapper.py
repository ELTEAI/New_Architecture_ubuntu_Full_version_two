from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/ubuntu/New_Architecture")
ROBOT_SKILL_SCHEMA_PATH = REPO_ROOT / "VLA_Agent_Core" / "schemas" / "robot_skill_schema.json"


@dataclass
class ActionTask:
    action_id: int
    duration: float = 0.0
    source: str = "test"


def load_schema_action_ids(schema_path: Path = ROBOT_SKILL_SCHEMA_PATH) -> set[int]:
    """
    Load supported action_id enum from VLA_Agent_Core/schemas/robot_skill_schema.json.

    This keeps the dry-run execution test aligned with the same skill schema
    used by the LLM planner.
    """
    if not schema_path.is_file():
        raise FileNotFoundError(f"robot_skill_schema.json not found: {schema_path}")

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        action_enum = (
            schema["function"]["parameters"]["properties"]["actions"]
            ["items"]["properties"]["action_id"]["enum"]
        )
    except KeyError as e:
        raise KeyError(f"Invalid robot_skill_schema.json structure, missing key: {e}") from e

    return {int(x) for x in action_enum}


class MockSportClient:
    """
    Mock version of Unitree high-level sport client.

    It does not communicate with a real robot.
    It only returns a structured dry-run response proving that
    the corresponding high-level command would have been called.
    """

    def Move(self, vx: float, vy: float, omega: float) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "Move",
            "args": {"vx": vx, "vy": vy, "omega": omega},
            "message": "dry-run Move accepted",
        }

    def StopMove(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "StopMove",
            "args": {},
            "message": "dry-run StopMove accepted",
        }

    def StandUp(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "StandUp",
            "args": {},
            "message": "dry-run StandUp accepted",
        }

    def StandDown(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "StandDown",
            "args": {},
            "message": "dry-run StandDown accepted",
        }

    def Stretch(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "Stretch",
            "args": {},
            "message": "dry-run Stretch accepted",
        }

    def RollOver(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "RollOver",
            "args": {},
            "message": "dry-run RollOver accepted",
        }

    def Pose(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "Pose",
            "args": {},
            "message": "dry-run Pose accepted",
        }

    def Greeting(self) -> dict[str, Any]:
        return {
            "ok": True,
            "method": "Greeting",
            "args": {},
            "message": "dry-run Greeting accepted",
        }


class DryRunHardwareWrapper:
    """
    Dry-run execution adapter.

    It preserves the ActionTask -> high-level SDK command mapping,
    but replaces physical robot actuation with mock responses.

    The action_id space is checked against VLA_Agent_Core/schemas/robot_skill_schema.json.
    """

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

    SPEED_MAP = {
        0: {"vx": 0.2, "vy": 0.0, "omega": 0.0},
        2: {"vx": -0.3, "vy": 0.0, "omega": 0.0},
        3: {"vx": 0.0, "vy": 0.3, "omega": 0.0},
        4: {"vx": 0.0, "vy": -0.3, "omega": 0.0},
        5: {"vx": 0.0, "vy": 0.0, "omega": 0.5},
        6: {"vx": 0.0, "vy": 0.0, "omega": -0.5},
    }

    def __init__(self, schema_path: Path = ROBOT_SKILL_SCHEMA_PATH) -> None:
        self.client = MockSportClient()
        self.logs: list[dict[str, Any]] = []

        self.schema_path = schema_path
        self.schema_action_ids = load_schema_action_ids(schema_path)
        self.local_action_ids = set(self.ACTION_NAMES.keys())

        if self.schema_action_ids != self.local_action_ids:
            raise ValueError(
                "Action ID mismatch between robot_skill_schema.json and DryRunHardwareWrapper. "
                f"schema={sorted(self.schema_action_ids)}, "
                f"local={sorted(self.local_action_ids)}"
            )

        mode_union = self.MODE_0_EMERGENCY | self.MODE_1_CONTINUOUS | self.MODE_2_BLOCKING
        if mode_union != self.local_action_ids:
            raise ValueError(
                "Action ID mismatch between mode sets and ACTION_NAMES. "
                f"mode_union={sorted(mode_union)}, "
                f"local={sorted(self.local_action_ids)}"
            )

    def validate_schema_duration(self, task: ActionTask) -> tuple[bool, str]:
        action_id = int(task.action_id)
        duration = float(task.duration)

        if action_id not in self.schema_action_ids:
            return False, f"action_id={action_id} is not defined in robot_skill_schema.json"

        if action_id in self.MODE_0_EMERGENCY:
            if duration == 0.0:
                return True, "valid mode0 duration"
            return False, f"Mode 0 action requires duration=0, got {duration}"

        if action_id in self.MODE_1_CONTINUOUS:
            if duration == 0.0:
                return True, "valid mode1 duration"
            return False, f"Mode 1 action requires duration=0, got {duration}"

        if action_id in self.MODE_2_BLOCKING:
            if 2.0 <= duration <= 5.0:
                return True, "valid mode2 duration"
            return False, f"Mode 2 action requires duration in [2,5], got {duration}"

        return False, f"Unknown action_id={action_id}"

    def dispatch(self, task: ActionTask) -> dict[str, Any]:
        t0 = time.perf_counter()

        action_id = int(task.action_id)
        action_name = self.ACTION_NAMES.get(action_id, "unknown")

        schema_ok, schema_reason = self.validate_schema_duration(task)

        if not schema_ok:
            t1 = time.perf_counter()
            result = {
                "ok": False,
                "method": None,
                "args": {},
                "message": schema_reason,
            }
            log = {
                "task": asdict(task),
                "action_name": action_name,
                "schema_ok": False,
                "schema_reason": schema_reason,
                "dispatch_success": False,
                "dispatch_latency": t1 - t0,
                "result": result,
            }
            self.logs.append(log)
            return log

        try:
            if action_id in self.SPEED_MAP:
                speed = self.SPEED_MAP[action_id]
                result = self.client.Move(
                    speed["vx"],
                    speed["vy"],
                    speed["omega"],
                )

            elif action_id == 1:
                result = self.client.StopMove()

            elif action_id == 7:
                result = self.client.StandDown()

            elif action_id == 8:
                result = self.client.StandUp()

            elif action_id == 9:
                result = self.client.Stretch()

            elif action_id == 10:
                result = self.client.RollOver()

            elif action_id == 11:
                result = self.client.Pose()

            elif action_id == 12:
                result = self.client.Greeting()

            else:
                result = {
                    "ok": False,
                    "method": None,
                    "args": {},
                    "message": f"No mapping for action_id={action_id}",
                }

            t1 = time.perf_counter()

            log = {
                "task": asdict(task),
                "action_name": action_name,
                "schema_ok": schema_ok,
                "schema_reason": schema_reason,
                "dispatch_success": bool(result.get("ok")),
                "dispatch_latency": t1 - t0,
                "result": result,
            }

        except Exception as e:
            t1 = time.perf_counter()
            log = {
                "task": asdict(task),
                "action_name": action_name,
                "schema_ok": schema_ok,
                "schema_reason": schema_reason,
                "dispatch_success": False,
                "dispatch_latency": t1 - t0,
                "result": {
                    "ok": False,
                    "method": None,
                    "args": {},
                    "message": repr(e),
                },
            }

        self.logs.append(log)
        return log