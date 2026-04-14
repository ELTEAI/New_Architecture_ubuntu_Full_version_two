from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[3]
_VLA_CORE = _ROOT / "VLA_Agent_Core"
if str(_VLA_CORE) not in sys.path:
    sys.path.insert(0, str(_VLA_CORE))

from execution.fsm_guardian import FSMGuardian  # noqa: E402


class FSMAdapter:
    def __init__(self, task_queue):
        self.guardian = FSMGuardian(task_queue)

    def start(self) -> None:
        self.guardian.start()

    def runtime_state(self) -> dict:
        return self.guardian.get_runtime_state()

