"""
Multiprocessing helper class
"""

import os
import multiprocessing as mp


class Pool:
    """Multiprocessing pool for running a function across multiple workers.
    Analogous to multiprocessing.Pool except that it also passes the process id to the target function as the first argument.
    """

    def __init__(self, processes=1):
        "Create pool, processes is number of processes"

        self.processes = processes

    def _worker_loop(self, func, ip, inputQueue):
        "Worker loop"

        while True:
            args = inputQueue.get(block=True, timeout=None)# get task
            if args == -1: break# poison apple

            # just in case there is a memory leak
            p = mp.Process(target=func, args=[ip]+args, daemon=False)
            p.start()
            p.join()
            if p.exitcode != 0:
                print(f"ERROR: process {os.getpid()} terminated unexpectedly")
                break

    def starmap(self, func, iterable):
        """Analogous to multiprocessing.Pool.starmap, except that the process id is also passed as the first arugment to func,
        i.e. computes func(ip, *iterable)
        """

        # put all inputs on input queue
        inputQueue = mp.Queue()
        for args in iterable: inputQueue.put(list(args))
        for _ in range(self.processes): inputQueue.put(-1)# poison apples

        # start processes running
        ps = [mp.Process(target=self._worker_loop, args=(func,ip,inputQueue), daemon=False) for ip in range(self.processes)]
        for p in ps: p.start()
        for p in ps: p.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _f(ip, x, y):
    #print(ip, x*y)
    int("asd")

if __name__ == "__main__":

    import numpy as np

    with Pool(processes=4) as pool:

        pool.starmap(_f, zip(np.arange(10), np.arange(10)))