import gc
import threading
import psutil
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback


def byte2gb(x):
    return float(x / 2**30)


class MemoryTrace:
    def __init__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = byte2gb(torch.cuda.memory_allocated())
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)


class MemoryCallback(Callback):
    """
        Trace the memory usage.
    """
    def __init__(self):
        self.memory_summary = {
            'train': {},
            'val': {},
            'test': {}
        }
    
    def update_memory_summary(self, key, memtrace):
        self.memory_summary[key] = {
            "mem_peak": max(memtrace.peak, self.memory_summary[key].get("mem_peak", 0)),
            "max_reserved": max(memtrace.max_reserved, self.memory_summary[key].get("max_reserved", 0)),
            "peak_active_gb": max(memtrace.peak_active_gb, self.memory_summary[key].get("peak_active_gb", 0)),
            "cuda_malloc_retires": max(memtrace.cuda_malloc_retires, self.memory_summary[key].get("cuda_malloc_retires", 0)),
            "cpu_total_peaked": max(memtrace.cpu_peaked + memtrace.cpu_begin, self.memory_summary[key].get("cpu_total_peaked", 0)),
        }
    
    def on_train_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the train epoch begins"""
        self.train_memtrace = MemoryTrace()
    
    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the train epoch ends"""
        self.train_memtrace.__exit__()
        self.update_memory_summary('train', self.train_memtrace)

    def on_validation_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the validation epoch begins"""
        self.val_memtrace = MemoryTrace()
    
    def on_validation_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the validation epoch ends"""
        self.val_memtrace.__exit__()
        self.update_memory_summary('val', self.val_memtrace)
    
    def on_test_epoch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the test epoch begins"""
        self.test_memtrace = MemoryTrace()
    
    def on_test_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the test epoch ends"""
        self.test_memtrace.__exit__()
        self.update_memory_summary('test', self.test_memtrace)
