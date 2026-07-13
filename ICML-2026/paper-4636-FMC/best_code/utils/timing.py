"""Hierarchical timing infrastructure with NFE tracking.

This module provides utilities for measuring wall clock time and NFE
(Number of Function Evaluations) for different components of methods and baselines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time
import json
from pathlib import Path
import numpy as np


@dataclass
class OperationTiming:
    """Timing data for a single operation."""
    name: str
    wall_time: float = 0.0
    nfe: int = 0
    count: int = 0  # Number of times this operation was called

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "wall_time": self.wall_time,
            "nfe": self.nfe,
            "count": self.count
        }


@dataclass
class ComponentTiming:
    """Timing data for a component with multiple operations."""
    operations: Dict[str, OperationTiming] = field(default_factory=dict)

    def add_operation(self, name: str, wall_time: float, nfe: int = 0):
        """Add timing data for an operation."""
        if name not in self.operations:
            self.operations[name] = OperationTiming(name=name)
        self.operations[name].wall_time += wall_time
        self.operations[name].nfe += nfe
        self.operations[name].count += 1

    def get_stats(self) -> Dict:
        """Compute statistics (mean, total) for operations."""
        stats = {}
        for name, op in self.operations.items():
            if op.count > 0:
                stats[name] = {
                    "time_total": float(op.wall_time),
                    "time_mean": float(op.wall_time / op.count),
                    "time_std": 0.0,  # Will be computed from aggregated data
                    "nfe_total": int(op.nfe),
                    "nfe_mean": float(op.nfe / op.count) if op.count > 0 else 0.0,
                    "count": int(op.count)
                }
        return stats

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {name: op.to_dict() for name, op in self.operations.items()}


class HierarchicalTimer:
    """Hierarchical timer with NFE tracking.

    This timer supports nested timing contexts organized by component and operation.
    For example, a method might have components like "training", "sampling", etc.,
    and each component might have operations like "flow_integration", "npe_sample", etc.

    Example usage:
        timer = HierarchicalTimer()

        # Manual timing
        timer.start("training", "epoch")
        # ... do training ...
        timer.stop(nfe=100)

        # Context manager (recommended)
        with timer.time_operation("sampling", "flow_integration") as ctx:
            # ... do flow integration ...
            ctx.add_nfe(20)  # Add NFE count
    """

    def __init__(self):
        self.components: Dict[str, ComponentTiming] = {}
        self.current_component: Optional[str] = None
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None

    def start(self, component: str, operation: str):
        """Start timing an operation within a component.

        Args:
            component: Component name (e.g., "training", "sampling")
            operation: Operation name (e.g., "epoch", "flow_integration")
        """
        self.current_component = component
        self.current_operation = operation
        self.start_time = time.perf_counter()

        if component not in self.components:
            self.components[component] = ComponentTiming()

    def stop(self, nfe: int = 0):
        """Stop timing the current operation.

        Args:
            nfe: Number of function evaluations for this operation
        """
        if self.start_time is None:
            return

        elapsed = time.perf_counter() - self.start_time

        if self.current_component and self.current_operation:
            self.components[self.current_component].add_operation(
                self.current_operation,
                wall_time=elapsed,
                nfe=nfe
            )

        self.start_time = None
        self.current_component = None
        self.current_operation = None

    def time_operation(self, component: str, operation: str):
        """Context manager for timing operations.

        Args:
            component: Component name
            operation: Operation name

        Returns:
            TimingContext that can accumulate NFE via add_nfe()

        Example:
            with timer.time_operation("sampling", "flow") as ctx:
                result, nfe = flow.sample(...)
                ctx.add_nfe(nfe)
        """
        return TimingContext(self, component, operation)

    def get_summary(self) -> Dict:
        """Get summary statistics for all components.

        Returns:
            Dictionary with structure:
            {
                component: {
                    operation: {
                        time_total: float,
                        time_mean: float,
                        time_std: float,
                        nfe_total: int,
                        nfe_mean: float,
                        count: int
                    }
                }
            }
        """
        return {
            component: timing.get_stats()
            for component, timing in self.components.items()
        }

    def save(self, path: Path):
        """Save timing data to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

    def to_dict(self) -> Dict:
        """Convert to raw dictionary (all data, not summary)."""
        return {
            component: timing.to_dict()
            for component, timing in self.components.items()
        }

    def get_total_time(self) -> float:
        """Get total wall clock time across all components and operations.

        Returns:
            Total wall time in seconds
        """
        total = 0.0
        for timing in self.components.values():
            for op in timing.operations.values():
                total += op.wall_time
        return total

    def merge(self, other: 'HierarchicalTimer'):
        """Merge another timer's data into this one.

        Useful for aggregating timing data across multiple runs.

        Args:
            other: Another HierarchicalTimer instance
        """
        for component, timing in other.components.items():
            if component not in self.components:
                self.components[component] = ComponentTiming()

            for op_name, op in timing.operations.items():
                self.components[component].add_operation(
                    op_name,
                    wall_time=op.wall_time,
                    nfe=op.nfe
                )


class TimingContext:
    """Context manager for timing operations.

    Automatically starts timing on enter and stops on exit.
    Supports NFE accumulation via add_nfe().
    """

    def __init__(self, timer: HierarchicalTimer, component: str, operation: str):
        self.timer = timer
        self.component = component
        self.operation = operation
        self.nfe = 0

    def __enter__(self):
        self.timer.start(self.component, self.operation)
        return self

    def __exit__(self, *args):
        self.timer.stop(nfe=self.nfe)

    def add_nfe(self, nfe: int):
        """Add NFE count to this operation.

        Args:
            nfe: Number of function evaluations to add
        """
        self.nfe += nfe


def compute_timing_statistics(timings: List[Dict]) -> Dict:
    """Compute aggregated statistics from multiple timing runs.

    Args:
        timings: List of timing dictionaries from HierarchicalTimer.get_summary()

    Returns:
        Dictionary with mean and std for each component/operation
    """
    if not timings:
        return {}

    # Collect all component/operation combinations
    all_components = {}

    for timing in timings:
        for component, operations in timing.items():
            if component not in all_components:
                all_components[component] = {}

            for operation, stats in operations.items():
                if operation not in all_components[component]:
                    all_components[component][operation] = {
                        'time_total': [],
                        'time_mean': [],
                        'nfe_total': [],
                        'nfe_mean': [],
                        'count': []
                    }

                all_components[component][operation]['time_total'].append(stats['time_total'])
                all_components[component][operation]['time_mean'].append(stats['time_mean'])
                all_components[component][operation]['nfe_total'].append(stats['nfe_total'])
                all_components[component][operation]['nfe_mean'].append(stats['nfe_mean'])
                all_components[component][operation]['count'].append(stats['count'])

    # Compute statistics
    aggregated = {}
    for component, operations in all_components.items():
        aggregated[component] = {}
        for operation, values in operations.items():
            aggregated[component][operation] = {
                'time_total_mean': float(np.mean(values['time_total'])),
                'time_total_std': float(np.std(values['time_total'])),
                'time_mean_mean': float(np.mean(values['time_mean'])),
                'time_mean_std': float(np.std(values['time_mean'])),
                'nfe_total_mean': float(np.mean(values['nfe_total'])),
                'nfe_total_std': float(np.std(values['nfe_total'])),
                'nfe_mean_mean': float(np.mean(values['nfe_mean'])),
                'nfe_mean_std': float(np.std(values['nfe_mean'])),
                'count_mean': float(np.mean(values['count'])),
            }

    return aggregated
