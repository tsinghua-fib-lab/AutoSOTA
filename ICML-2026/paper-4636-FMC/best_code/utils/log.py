from mlxp.logger import Logger
from typing import List

import time


class Timer:
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self.total += self.interval
        print(f"Elapsed time: {self.interval:.2f} seconds, (total: {self.total:.2f} seconds)")


def log_loss_list(logger: Logger, loss_list: List[float], name: str, log_name) -> None:
    """Log a loss list with mlxp logger.

    Args:
        logger: mlxp logger
        loss_list: List of losse item size of the number of epochs
        name: Name of the log
    """
    for i, loss in enumerate(loss_list):
        logger.log_metrics({name: loss, "epoch": i}, log_name=log_name)
