from pathlib import Path
import sys
from typing import Iterable

from src.pipeline.contracts import ActionTask


_ROOT = Path(__file__).resolve().parents[3]
_VLA_CORE = _ROOT / "VLA_Agent_Core"
if str(_VLA_CORE) not in sys.path:
    sys.path.insert(0, str(_VLA_CORE))

from execution.task_queue import TaskQueue  # noqa: E402


class TaskQueueAdapter:
    def __init__(self, max_size: int = 20):
        self.inner = TaskQueue(max_size=max_size)

    def push_tasks(self, tasks: Iterable[ActionTask]) -> None:
        self.inner.push_sequence([t.to_dict() for t in tasks])

    def clear(self) -> None:
        self.inner.clear_queue()

    def wait_all_done(self) -> None:
        self.inner.wait_until_all_done()

    def pending_count(self) -> int:
        return self.inner.pending_count()

