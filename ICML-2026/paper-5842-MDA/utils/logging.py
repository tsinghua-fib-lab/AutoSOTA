# utils/logging.py

import datetime


def log(msg: str):
    """
    Simple timestamped logger.
    """
    t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}")
