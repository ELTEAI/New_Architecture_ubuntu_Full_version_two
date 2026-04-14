from src.pipeline.contracts import ActionTask


def test_action_task_to_dict():
    t = ActionTask(action_id=2, duration=0.0, source="unit")
    d = t.to_dict()
    assert d["action_id"] == 2
    assert d["duration"] == 0.0

