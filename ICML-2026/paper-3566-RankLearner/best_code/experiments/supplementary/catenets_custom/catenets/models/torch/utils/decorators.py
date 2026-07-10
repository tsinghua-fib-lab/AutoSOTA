import time
from typing import Any, Callable

import torch

import catenets.logger as log


def check_input_train(func: Callable) -> Callable:
    """Decorator used for checking training params.

    Args:
        func: the function to be benchmarked.

    Returns:
        Callable: the decorator

    """

    def wrapper(self: Any, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> Any:

        w = torch.Tensor(w)

        if not ((w == 0) | (w == 1)).all():
            raise ValueError("W should be binary")

        return func(self, X, y, w)

    return wrapper


def benchmark(func: Callable) -> Callable:
    """Decorator used for function duration benchmarking. It is active only with DEBUG loglevel.

    Args:
        func: the function to be benchmarked.

    Returns:
        Callable: the decorator

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()

        log.debug(f"{func.__qualname__} took {round(end - start, 4)} seconds")
        return res

    return wrapper
