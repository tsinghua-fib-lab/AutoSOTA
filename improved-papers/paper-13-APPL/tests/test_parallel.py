import time
from typing import List

from appl import CallFuture, call, ppl, records
from appl.types import ExecutorType


def test_call_future():
    def run(t):
        time.sleep(t)
        return 1

    t0 = time.time()
    n = 3
    calls: List[CallFuture] = []
    for i in range(n):
        calls.append(call(run, t=0.1))
    used = time.time() - t0
    assert used < 0.05

    c = sum(call() for call in calls)
    used = time.time() - t0
    assert c == n
    assert used < 0.2

    c = sum(call.val for call in calls)
    used = time.time() - t0
    assert c == n
    assert used < 0.2


def sleep(t):
    time.sleep(t)
    return 1


def test_call_future_process():
    c = call(sleep, t=0.1, executor_type=ExecutorType.NEW_PROCESS)
    assert c.val == 1


def test_format_future():
    def run(t):
        time.sleep(t)
        return 1

    n = 3

    @ppl
    def func():
        t0 = time.time()
        for i in range(n):
            f"{call(run, t=0.1)}"
            print(time.time() - t0)
        return records()

    t0 = time.time()
    assert str(func()) == "1\n1\n1"
    used = time.time() - t0
    assert used < 0.2
