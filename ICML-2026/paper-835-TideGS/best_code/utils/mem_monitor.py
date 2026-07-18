"""
Lightweight CPU RSS / available-memory monitor for OOM debugging.

Usage in train_ssdoffload.py:
    from utils.mem_monitor import MemMonitor
    mem_mon = MemMonitor(log_dir=output_dir, warn_avail_gb=15.0)

    # inside the training loop, after each TideGS batch:
    mem_mon.tick(iteration)
"""

import csv
import os
import time
import psutil


class MemMonitor:
    """Write one CSV row per tick with RSS / available / total."""

    def __init__(self, log_dir: str, warn_avail_gb: float = 15.0,
                 flush_every: int = 10, track_children: bool = True):
        self._proc = psutil.Process()
        self._warn_avail = warn_avail_gb
        self._flush_every = flush_every
        self._trackchildren = track_children
        self._cnt = 0

        path = os.path.join(log_dir, "mem_monitor.csv")
        self._fp = open(path, "w", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow([
            "iter", "wall_s",
            "rss_gb", "uss_gb", "tree_rss_gb",
            "avail_gb", "total_gb", "swap_used_gb",
            "num_children"
        ])
        self._t0 = time.time()
    
    def _sum_tree_rss_bytes(self) -> (int, int):
        """Return (tree_rss_bytes, num_children)."""
        mi = self._proc.memory_info()
        rss = mi.rss
        nchild = 0
        if not self._trackchildren:
            return rss, nchild
        
        try:
            children = self._proc.children(recursive=True)
        except psutil.Error:
            children = []
        
        for ch in children:
            try:
                rss += ch.memory_info().rss
                nchild += 1
            except psutil.Error:
                continue
        return rss, nchild

    # ------------------------------------------------------------------ #
    def tick(self, iteration: int) -> None:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        # RSS
        mi = self._proc.memory_info()
        rss_gb = mi.rss / (1 << 30)

        # USS
        try:
            uss = self._proc.memory_full_info().uss
            uss_gb = uss / (1 << 30)
        except (psutil.AccessDenied, AttributeError):
            uss_gb = float("nan")

        # Tree RSS (process + children)
        tree_rss, nchild = self._sum_tree_rss_bytes()
        tree_rss_gb = tree_rss / (1 << 30)

        avail_gb = vm.available / (1 << 30)
        total_gb = vm.total / (1 << 30)
        swap_used_gb = sm.used / (1 << 30)

        self._writer.writerow([
            iteration,
            f"{time.time() - self._t0:.1f}",
            f"{rss_gb:.2f}",
            f"{uss_gb:.2f}" if uss_gb == uss_gb else "N/A",
            f"{tree_rss_gb:.2f}",
            f"{avail_gb:.2f}",
            f"{total_gb:.2f}",
            f"{swap_used_gb:.2f}",
            nchild
        ])

        self._cnt += 1
        if self._cnt % self._flush_every == 0 or avail_gb < self._warn_avail:
            self._fp.flush()

        if avail_gb < self._warn_avail:
            print(f"\n[MemMonitor] WARNING iter={iteration}: "
                  f"avail={avail_gb:.1f} GB < {self._warn_avail} GB  "
                  f"RSS={rss_gb:.1f} GB, Tree RSS={tree_rss_gb:.1f} GB, Swap used={swap_used_gb:.1f} GB "
                  f"children={nchild}\n", 
                  flush=True)


    def close(self):
        try:
            self._fp.flush()
        finally:
            self._fp.close()
