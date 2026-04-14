from pathlib import Path
import sys

from src.pipeline.contracts import ActionTask, PlanResult


class PlannerClient:
    def __init__(self, config_path: str):
        root = Path(__file__).resolve().parents[3]
        vla_core = root / "VLA_Agent_Core"
        if str(vla_core) not in sys.path:
            sys.path.insert(0, str(vla_core))
        from core.agent_planner import VLABrainPlanner  # local import to reduce hard dependency at module import time

        self._planner = VLABrainPlanner(config_path=config_path)

    def plan(self, text: str) -> PlanResult:
        seq_name, actions = self._planner.compile_tactical_plan(text)
        tasks = [
            ActionTask(
                action_id=int(a.get("action_id", 1)),
                duration=float(a.get("duration", 0.0)),
                source="planner",
            )
            for a in actions
        ]
        return PlanResult(sequence_name=seq_name or "unknown", actions=tasks, raw={"text": text})

