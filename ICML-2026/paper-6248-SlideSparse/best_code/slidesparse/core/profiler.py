# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Profiling Module

Enabled via SLIDESPARSE_PROFILE=1 env var.

Usage:
======
    with ProfileTimer("operation_name"):
        # Code to time
        ...

    # Call after each apply
    profile_step()

Notes:
======
- Enabling profiling breaks CUDA Graph, must use eager mode
- Async timing with batch sync to reduce GPU pipeline stalls
- Auto-disabled during torch.compile tracing (returns nullcontext)
"""

import os
from collections import defaultdict
from contextlib import nullcontext

import torch


# ============================================================================
# Configuration Constants
# ============================================================================

_PROFILE_ENABLED = os.environ.get("SLIDESPARSE_PROFILE", "0") == "1"
_PROFILE_PRINT_INTERVAL = 1000  # Print stats every N calls
_PENDING_FLUSH_THRESHOLD = 1000  # Batch sync after N events


# ============================================================================
# Global State
# ============================================================================

_profile_data = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
_profile_call_count = 0

# Async timing: store pending CUDA events, lazy sync
_pending_events: list = []  # [(name, start_event, end_event), ...]


# ============================================================================
# ProfileTimer Implementation
# ============================================================================

class _ProfileTimerImpl:
    """
    Async CUDA timer using events (no GPU pipeline blocking).
    Events are batched and synchronized lazily for accurate timing.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled and _PROFILE_ENABLED
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            _pending_events.append((self.name, self.start_event, self.end_event))
            if len(_pending_events) >= _PENDING_FLUSH_THRESHOLD:
                _flush_pending_events()


def ProfileTimer(name: str, enabled: bool = True):
    """ProfileTimer factory - returns nullcontext during torch.compile tracing."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return _ProfileTimerImpl(name, enabled)


# ============================================================================
# Event Processing Functions
# ============================================================================

def _flush_pending_events():
    """Synchronize and collect all pending events."""
    global _pending_events
    if not _pending_events:
        return
    
    torch.cuda.synchronize()
    
    for name, start_event, end_event in _pending_events:
        elapsed_ms = start_event.elapsed_time(end_event)
        _profile_data[name]["total_ms"] += elapsed_ms
        _profile_data[name]["count"] += 1
    
    _pending_events = []


def profile_step():
    """Check if stats should be printed (called after each apply)."""
    global _profile_call_count
    if not _PROFILE_ENABLED:
        return
    
    _profile_call_count += 1
    if _profile_call_count % _PROFILE_PRINT_INTERVAL == 0:
        print_profile_stats()


# ============================================================================
# Statistics Output Functions
# ============================================================================

def print_profile_stats():
    """Print profiling statistics."""
    _flush_pending_events()
    
    if not _profile_data:
        return
    
    print(f"\n{'=' * 80}")
    print(f"SlideSparse Profile Stats (after {_profile_call_count} calls)")
    print(f"{'=' * 80}")
    print(f"{'Operation':<40} {'Count':>10} {'Total(ms)':>12} {'Avg(us)':>10}")
    print(f"{'-' * 80}")
    
    # Sort by total_ms descending
    sorted_items = sorted(_profile_data.items(), key=lambda x: -x[1]["total_ms"])
    
    for name, data in sorted_items:
        count = data["count"]
        total_ms = data["total_ms"]
        avg_ms = total_ms / count if count > 0 else 0
        print(f"{name:<40} {count:>10} {total_ms:>12.3f} {avg_ms * 1000:>10.1f}")
    
    print(f"{'=' * 80}\n")


def reset_profile_stats():
    """Reset profiling stats (discards all unflushed pending events)"""
    global _profile_call_count, _pending_events
    _profile_data.clear()
    _profile_call_count = 0
    _pending_events = []


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "ProfileTimer",
    "profile_step",
    "print_profile_stats",
    "reset_profile_stats",
    "_flush_pending_events",
]
