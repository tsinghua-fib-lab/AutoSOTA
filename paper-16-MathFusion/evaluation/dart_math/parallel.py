import asyncio
import functools
import inspect
import multiprocessing as mp
import queue
from typing import Any, Awaitable, Callable

from tqdm import tqdm


# %% ../nbs/05_parallel.ipynb 0
def async_wrap(func: Callable) -> Awaitable:
    """Wrap a synchronous function `func` into an asynchronous function."""

    @functools.wraps(func)
    async def coroutine_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return coroutine_wrapper


async def seq_consume_preset_queue_w_each_timeout(
    consumer: Callable,
    idxed_kwargs_queue: queue.SimpleQueue | type(mp.SimpleQueue()) | list,
    timeout: int = 5,
    pbar: tqdm = None,
) -> list[tuple[int, Any]]:
    """Sequentially run computation-intensive `consumer` along a preset (no more input) indexed task `idxed_kwargs_queue` with each task having `timeout`.
    `queue.SimpleQueue` is not thread-safe, don't run multiple consumers in the same process.
    However, `multiprocessing.SimpleQueue` is process-safe based on pipe, you can run multiple consumers in the same number of processes.
    NOTE: co-routine would get stuck with some consumer like creating and running new interactive IPython shells even with timeout set.

    Parameters
    ----------
    consumer : Awaitable
        An `Awaitable` coroutine function.
    idxed_kwargs_queue : queue.SimpleQueue | type(mp.SimpleQueue())
        Indexed kwargs queue, comprising elements like `(idx, kwargs)`.
        For the weird `type` hint, refer to https://github.com/python/cpython/issues/99509
    timeout : int, default: 5
        Timeout for each task.
    pbar : tqdm, default: None
        Progress bar to update. `None` means no progress bar.

    Returns
    -------
    list[tuple[int, Any]]
        Indexed return values, comprising elements like `(idx, retval)`.
    """
    if not inspect.isawaitable(consumer):
        consumer = async_wrap(consumer)

    if isinstance(idxed_kwargs_queue, (list, enumerate, tuple)):
        idx_kwargs_list = idxed_kwargs_queue
        idxed_kwargs_queue = queue.SimpleQueue()
        for idx_kwargs in idx_kwargs_list:
            idxed_kwargs_queue.put(idx_kwargs)
        idxed_kwargs_queue.put(None)

    idxed_retvals = []
    while True:
        idxed_kwargs = idxed_kwargs_queue.get()
        if idxed_kwargs is None:
            break
        idx, kwargs = idxed_kwargs
        task = asyncio.create_task(consumer(**kwargs))
        try:
            retval = await asyncio.wait_for(task, timeout)
        except Exception as e:  # e.g. `asyncio.TimeoutError`
            task.cancel()
            retval = e
        idxed_retvals.append((idx, retval))
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    return idxed_retvals
