"""
Metric tracking and progress logging utilities.

Trainers use these small stateful helpers to aggregate scalar values, estimate
iteration timing, and format progress output during training and validation.
"""
import logging
import time
from typing import List, Dict, Any, Optional


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display training progress."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class TimeMeter:
    """Measures elapsed time between events."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.elapsed = 0
    
    def update(self):
        """Update and return time (in seconds) since last update."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        return elapsed
    
    def total_elapsed(self):
        """Return total time (in seconds) since initialization or last reset."""
        return time.time() - self.start_time


def create_progress_meters(
    num_losses: int = 1, 
    loss_names: Optional[List[str]] = None
) -> Dict[str, AverageMeter]:
    """
    Create a standard set of meters for tracking training progress.
    
    Args:
        num_losses: Number of loss components to track
        loss_names: Names of loss components (defaults to "loss_1", "loss_2", etc.)
        
    Returns:
        Dictionary of meters by name
    """
    if loss_names is None:
        loss_names = [f"loss_{i+1}" for i in range(num_losses)]
    
    meters = {
        "batch_time": AverageMeter("Time", ":.4f"),
        "data_time": AverageMeter("Data", ":.4f"),
        "loss": AverageMeter("Loss", ":.4f"),
    }
    
    # Add individual loss meters
    for name in loss_names:
        meters[name] = AverageMeter(name, ":.4f")
    
    return meters
