import threading
import time


class Heartbeat(threading.Thread):
    def __init__(self, interval_sec: float = 2.0):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            print(f"💓 [Health] pipeline alive @ {time.strftime('%H:%M:%S')}")
            time.sleep(self.interval_sec)

