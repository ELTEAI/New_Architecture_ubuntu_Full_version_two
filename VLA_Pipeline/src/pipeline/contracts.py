from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class ActionTask:
    action_id: int
    duration: float = 0.0
    source: str = "unknown"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {"action_id": int(self.action_id), "duration": float(self.duration)}


@dataclass
class PlanResult:
    sequence_name: str
    actions: List[ActionTask] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


@dataclass
class PerceptionEvent:
    pred_id: int
    confidence: float
    ts: float = field(default_factory=time.time)


@dataclass
class SpeechEvent:
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class SystemEvent:
    level: str
    message: str
    source: str = "system"
    ts: float = field(default_factory=time.time)
    payload: Optional[Dict[str, Any]] = None


@dataclass
class ErrorEvent:
    message: str
    source: str = "system"
    exc_type: str = "Exception"
    ts: float = field(default_factory=time.time)
    payload: Optional[Dict[str, Any]] = None

