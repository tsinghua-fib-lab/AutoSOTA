# -*- coding: utf-8 -*-
"""Training logging utilities."""

import sys


class TeeLogger:
    """
    Write stdout simultaneously to the terminal and a log file.

    Usage:
        tee = TeeLogger("run.log")
        sys.stdout = tee
        print("This goes to terminal and run.log")
        sys.stdout = tee.terminal  # restore
        tee.close()
    """

    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()
