import glob
import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(v) for v in obj]
    return obj


def save_results_json(results, directory, prefix, timestamp=None):
    """Save results to timestamped JSON file."""
    os.makedirs(directory, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{directory}/{prefix}_{timestamp}.json"
    json_serializable = convert_numpy_to_list(results)

    with open(filename, "w") as f:
        json.dump(json_serializable, f, indent=4)

    return filename


def find_latest_file(patterns):
    """Find most recent file matching glob patterns."""
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))

    return max(all_files, key=os.path.getmtime) if all_files else None


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def aggregate_results_across_seeds(results, metrics=None):
    """Aggregate results across seeds (compute mean and std)."""
    if metrics is None and results:
        metrics = [k for k in results[0].keys() if isinstance(results[0][k], list)]

    aggregated = {}
    for metric in metrics:
        metric_data = [r[metric] for r in results if metric in r and r[metric]]

        if metric_data and len(set(len(d) for d in metric_data)) == 1:
            aggregated[f"{metric}_mean"] = list(map(mean, zip(*metric_data)))
            if len(metric_data) > 1:
                aggregated[f"{metric}_std"] = list(map(stdev, zip(*metric_data)))

    return aggregated


class ExperimentTimer:
    """Context manager for timing experiments."""

    def __init__(self, name="Experiment"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        import time

        self.start_time = time.time()
        print(f"\n[{self.name}] Starting...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"[{self.name}] Completed in {elapsed:.2f}s ({elapsed / 60:.2f}min)\n")
        return False

    @property
    def elapsed(self):
        if self.start_time is None:
            return 0.0
        import time

        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
