from datetime import datetime


def time_log() -> str:
    a = datetime.now()
    return f"-" * 60 + f"  {a.year:>4}/{a.month:>2}/{a.day:>2} | {a.hour:>2}:{a.minute:>2}:{a.second:>2}\n"

class Timer:
    def __init__(self):
        self._now = time.process_time_ns()

    def update(self) -> float:
        current = time.process_time_ns()
        duration = current - self._now
        self._now = current
        return duration / 1e6  # ms