import os, time, threading
import psutil
import torch

import os, time, threading
import psutil

class PeakRSS:
    def __init__(self, interval=0.01, include_children=True):
        self.interval = interval
        self.include_children = include_children
        self.proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self.peak = 0
        self._thread = None

    def _rss_tree(self):
        total = 0
        # self
        try:
            total += self.proc.memory_info().rss
        except psutil.Error:
            pass
        if self.include_children:
            for ch in self.proc.children(recursive=True):
                try:
                    total += ch.memory_info().rss
                except psutil.Error:
                    pass
        return total

    def _poll(self):
        while not self._stop.is_set():
            rss = self._rss_tree()
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self._rss_tree()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def bytes_to_gb(x): 
    return x / (1024**3)
