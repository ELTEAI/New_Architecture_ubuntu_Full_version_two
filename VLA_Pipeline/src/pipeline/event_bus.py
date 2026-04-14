import queue
from typing import Any


class EventBus:
    """最简线程安全事件总线（基于 Queue）。"""

    def __init__(self, maxsize: int = 256):
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)

    def publish(self, event: Any) -> bool:
        if self._q.full():
            return False
        self._q.put(event)
        return True

    def consume(self, timeout: float | None = None) -> Any:
        return self._q.get(timeout=timeout)

