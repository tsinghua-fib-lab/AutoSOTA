import rpyc
from typing import Sequence, Any
from threading import Thread, Lock
from collections import deque


class TaskQueue(rpyc.Service):
    """
    A simple queue of tasks that can be accessed over pure-python rpc (RPyC).
    **WARNING** this class causes a minor memory leak after deletion due to cyclic references. We didn't care.
    :param tasks: a list of tasks to be dispatched
    :param start: if specified, runs this in a background thread (use .shutdown() to terminate)
    :param kwargs: any additional argumepts are passed into a ThreadedServer, e.g. (hostname, port, ipv6)
    """

    def __init__(self, tasks: Sequence[Any], *, start: bool, **kwargs):
        super().__init__()
        self.tasks, self.unordered_results = deque(tasks), deque()
        self.lock = Lock()
        self._server = rpyc.ThreadedServer(self, **kwargs)
        self.endpoint = f"{self._server.host}:{self._server.port}"
        if start:
            Thread(target=self._server.start, daemon=True).start()  # fire-and-forget thread w/o reference

    def exposed_get_task(self):
        with self.lock:
            if len(self.tasks) == 0:
                raise EOFError("No tasks left")  # note: if server is shut down, it also raises EOFError
            return self.tasks.popleft()

    def shutdown(self):
        assert self._server is not None, "server wasn't started "
        self._server.close()  # this will also

    @staticmethod
    def iterate_tasks_from_queue(endpoint: str):
        """Connect to a queue and receive tasks until queue becomes empty"""
        conn = rpyc.connect(endpoint[:endpoint.rindex(":")], port=int(endpoint[endpoint.rindex(":") + 1:]))
        while True:
            try:
                yield conn.root.get_task()
            except EOFError:
                break
