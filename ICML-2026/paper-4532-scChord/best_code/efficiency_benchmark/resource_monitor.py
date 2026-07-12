# -*- coding: utf-8 -*-
"""
Resource monitoring utilities for wall-clock time,
peak GPU memory, and peak CPU memory.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _is_cuda_device(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


class ResourceMonitor:
    """Stage-level resource monitor."""

    def __init__(self, device: torch.device, poll_interval_s: float = 0.05):
        self.device = device
        self.poll_interval_s = poll_interval_s
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cpu_peak_rss_bytes = 0
        self._cpu_start_rss_bytes = 0
        self._cpu_end_rss_bytes = 0
        self._start_time = 0.0

    def _sample_cpu_peak_loop(self):
        while self._running:
            try:
                rss = self._proc.memory_info().rss if self._proc is not None else 0
                if rss > self._cpu_peak_rss_bytes:
                    self._cpu_peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.poll_interval_s)

    def start(self):
        self._start_time = time.perf_counter()

        if self._proc is not None:
            try:
                self._cpu_start_rss_bytes = self._proc.memory_info().rss
                self._cpu_peak_rss_bytes = self._cpu_start_rss_bytes
            except Exception:
                self._cpu_start_rss_bytes = 0
                self._cpu_peak_rss_bytes = 0

        if _is_cuda_device(self.device):
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        self._running = True
        self._thread = threading.Thread(target=self._sample_cpu_peak_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=2.0)

        if self._proc is not None:
            try:
                self._cpu_end_rss_bytes = self._proc.memory_info().rss
                self._cpu_peak_rss_bytes = max(self._cpu_peak_rss_bytes, self._cpu_end_rss_bytes)
            except Exception:
                pass

        wall_clock_s = time.perf_counter() - self._start_time

        gpu_peak_mem_mb: Optional[float] = None
        if _is_cuda_device(self.device):
            torch.cuda.synchronize(self.device)
            gpu_peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

        cpu_peak_mem_mb = self._cpu_peak_rss_bytes / (1024 ** 2)
        cpu_start_mem_mb = self._cpu_start_rss_bytes / (1024 ** 2)
        cpu_end_mem_mb = self._cpu_end_rss_bytes / (1024 ** 2)

        return {
            "wall_clock_s": float(wall_clock_s),
            "gpu_peak_mem_mb": None if gpu_peak_mem_mb is None else float(gpu_peak_mem_mb),
            "cpu_peak_mem_mb": float(cpu_peak_mem_mb),
            "cpu_start_mem_mb": float(cpu_start_mem_mb),
            "cpu_end_mem_mb": float(cpu_end_mem_mb),
            "device": str(self.device),
        }


def save_metrics_json(metrics: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
