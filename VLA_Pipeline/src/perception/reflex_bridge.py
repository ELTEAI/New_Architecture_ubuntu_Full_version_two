import time
from dataclasses import dataclass

from src.pipeline.contracts import ActionTask, PerceptionEvent


@dataclass
class ReflexConfig:
    conf_threshold: float = 0.85
    cooldown_sec: float = 1.0


class ReflexBridge:
    """pred_id -> ActionTask，含置信度阈值 + 冷却 + 去重。"""

    def __init__(self, conf_threshold: float = 0.85, cooldown_sec: float = 1.0):
        self.cfg = ReflexConfig(conf_threshold=conf_threshold, cooldown_sec=cooldown_sec)
        self._last_action_id: int | None = None
        self._last_emit_ts: float = 0.0

    @staticmethod
    def _duration_for_action(action_id: int) -> float:
        return 3.0 if action_id in {7, 8, 9, 10, 11, 12} else 0.0

    def to_task(self, event: PerceptionEvent) -> ActionTask | None:
        now = time.time()
        if event.confidence < self.cfg.conf_threshold:
            return None
        if self._last_action_id == event.pred_id and now - self._last_emit_ts < self.cfg.cooldown_sec:
            return None
        self._last_action_id = event.pred_id
        self._last_emit_ts = now
        return ActionTask(
            action_id=int(event.pred_id),
            duration=self._duration_for_action(int(event.pred_id)),
            source="reflex",
            confidence=float(event.confidence),
        )

