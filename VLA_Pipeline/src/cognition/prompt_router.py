def is_emergency_text(text: str) -> bool:
    t = text.strip().lower()
    return t in {"停下", "停止", "急停", "stop", "halt", "emergency"}


def route_text(text: str) -> str:
    if is_emergency_text(text):
        return "emergency"
    if not text.strip():
        return "empty"
    return "plan"

