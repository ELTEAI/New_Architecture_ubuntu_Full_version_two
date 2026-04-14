import time

from src.perception.reflex_bridge import ReflexBridge
from src.pipeline.contracts import PerceptionEvent


def test_reflex_bridge_threshold_and_cooldown():
    rb = ReflexBridge(conf_threshold=0.8, cooldown_sec=0.3)

    assert rb.to_task(PerceptionEvent(pred_id=0, confidence=0.5)) is None

    t1 = rb.to_task(PerceptionEvent(pred_id=0, confidence=0.95))
    assert t1 is not None
    assert t1.action_id == 0

    t2 = rb.to_task(PerceptionEvent(pred_id=0, confidence=0.95))
    assert t2 is None

    time.sleep(0.31)
    t3 = rb.to_task(PerceptionEvent(pred_id=0, confidence=0.95))
    assert t3 is not None

