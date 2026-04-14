class Metrics:
    def __init__(self):
        self.counters: dict[str, int] = {}

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def snapshot(self) -> dict[str, int]:
        return dict(self.counters)

