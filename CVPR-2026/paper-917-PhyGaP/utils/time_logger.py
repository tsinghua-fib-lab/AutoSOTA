import os
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TIME_LOG_PATH = os.path.join(_REPO_ROOT, "time.log")


def append_time_log(message: str) -> None:
    """Append a timestamped message to the repository-level time log."""
    timestamp = datetime.now().isoformat()
    line = f"[{timestamp}] {message.strip()}\n"
    with open(_TIME_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(line)
