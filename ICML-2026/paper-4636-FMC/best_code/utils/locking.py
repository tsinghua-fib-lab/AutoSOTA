"""File locking utilities for safe parallel execution.

This module provides utilities for concurrent training and evaluation jobs,
ensuring that multiple workers don't interfere with each other when accessing
shared resources.

Usage:
    from utils.locking import TrainingLock, atomic_write

    # Lock for training a specific model
    with TrainingLock(locks_dir, task, seed, ncal, variant_name):
        if not model_exists():
            train_model()

    # Atomic file writes
    atomic_write(data, target_path)
"""

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from filelock import FileLock, Timeout
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False


class LockTimeout(Exception):
    """Raised when a lock cannot be acquired within the timeout."""
    pass


class ResourceLock:
    """Generic resource lock for parallel execution.

    Context manager that acquires a file lock and optionally checks
    if a result already exists after acquiring the lock.

    Example:
        with ResourceLock(locks_dir, "my_resource") as lock:
            if lock.already_exists:
                return load_result()
            result = compute()
            save(result)
            return result
    """

    def __init__(
        self,
        locks_dir: Path,
        lock_name: str,
        timeout: float = 300.0,
        check_exists: Optional[Callable[[], bool]] = None,
    ):
        self.locks_dir = Path(locks_dir)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

        self.lock_name = lock_name + ".lock"
        self.lock_path = self.locks_dir / self.lock_name
        self.timeout = timeout
        self.check_exists = check_exists
        self.already_exists = False
        self._lock = None

    def __enter__(self):
        if not HAS_FILELOCK:
            raise ImportError(
                "filelock package is required for parallel execution. "
                "Install it with: pip install filelock"
            )

        self._lock = FileLock(str(self.lock_path), timeout=self.timeout)

        try:
            self._lock.acquire()
        except Timeout:
            raise LockTimeout(
                f"Could not acquire lock for {self.lock_name} within {self.timeout}s"
            )

        if self.check_exists is not None:
            self.already_exists = self.check_exists()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock is not None:
            self._lock.release()
        return False


# Backward-compatible factory functions

def TrainingLock(
    locks_dir: Path,
    task: str,
    seed: int,
    ncal: int,
    variant_name: Optional[str] = None,
    method: Optional[str] = None,
    timeout: float = 300.0,
    check_exists: Optional[Callable[[], bool]] = None,
) -> ResourceLock:
    """Create a lock for training operations."""
    parts = [task, f"seed_{seed}", f"ncal_{ncal}"]
    if variant_name:
        parts.insert(0, variant_name)
    if method:
        parts.append(method)
    return ResourceLock(locks_dir, "_".join(parts), timeout, check_exists)


def SimulationModelLock(
    locks_dir: Path,
    task: str,
    model_type: str,
    timeout: float = 600.0,
    check_exists: Optional[Callable[[], bool]] = None,
) -> ResourceLock:
    """Create a lock for simulation model training."""
    return ResourceLock(locks_dir, f"simulation_{task}_{model_type}", timeout, check_exists)


def DataGenerationLock(
    locks_dir: Path,
    task: str,
    timeout: float = 300.0,
    check_exists: Optional[Callable[[], bool]] = None,
) -> ResourceLock:
    """Create a lock for data generation."""
    return ResourceLock(locks_dir, f"data_{task}", timeout, check_exists)


def atomic_write(
    data: Any,
    target_path: Path,
    write_fn: Optional[Callable[[Any, Path], None]] = None,
) -> Path:
    """Write data atomically by writing to temp file then renaming.

    This ensures that the target file is never in a partial state,
    which is important for parallel execution.

    Args:
        data: Data to write
        target_path: Final path for the file
        write_fn: Custom write function. If None, uses torch.save for
                  .pt/.pth files, json.dump for .json, pickle for .pkl

    Returns:
        Path to written file
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    fd, temp_path = tempfile.mkstemp(
        dir=target_path.parent,
        suffix=target_path.suffix,
    )
    os.close(fd)
    temp_path = Path(temp_path)

    try:
        if write_fn is not None:
            write_fn(data, temp_path)
        elif target_path.suffix in (".pt", ".pth"):
            import torch
            torch.save(data, temp_path)
        elif target_path.suffix == ".json":
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
        elif target_path.suffix == ".pkl":
            import pickle
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)
        else:
            # Default to string write
            with open(temp_path, "w") as f:
                f.write(str(data))

        # Atomic rename
        shutil.move(str(temp_path), str(target_path))

    except Exception:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise

    return target_path


@contextmanager
def optional_lock(
    use_lock: bool,
    lock_class: type,
    *args,
    **kwargs,
):
    """Context manager that optionally applies a lock.

    Useful when you want to conditionally enable locking based on config.

    Args:
        use_lock: Whether to actually acquire the lock
        lock_class: Lock class to use (TrainingLock, SimulationModelLock, etc.)
        *args: Arguments to pass to lock class
        **kwargs: Keyword arguments to pass to lock class

    Example:
        with optional_lock(parallel_mode, TrainingLock, locks_dir, task, seed, ncal):
            train_model()
    """
    if use_lock:
        with lock_class(*args, **kwargs) as lock:
            yield lock
    else:
        # Create a dummy object with already_exists=False
        class DummyLock:
            already_exists = False
        yield DummyLock()


def wait_for_file(
    file_path: Path,
    timeout: float = 300.0,
    poll_interval: float = 1.0,
) -> bool:
    """Wait for a file to exist.

    Useful when one worker is waiting for another to finish generating data.

    Args:
        file_path: Path to wait for
        timeout: Maximum wait time in seconds
        poll_interval: Time between checks in seconds

    Returns:
        True if file exists within timeout, False otherwise
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if file_path.exists():
            return True
        time.sleep(poll_interval)
    return False


def clean_stale_locks(locks_dir: Path, max_age_hours: float = 24.0) -> int:
    """Clean up stale lock files.

    Lock files older than max_age_hours are considered stale and removed.
    This helps recover from crashed jobs that didn't release their locks.

    Args:
        locks_dir: Directory containing lock files
        max_age_hours: Maximum age for lock files in hours

    Returns:
        Number of lock files removed
    """
    locks_dir = Path(locks_dir)
    if not locks_dir.exists():
        return 0

    max_age_seconds = max_age_hours * 3600
    current_time = time.time()
    removed = 0

    for lock_file in locks_dir.glob("*.lock"):
        try:
            file_age = current_time - lock_file.stat().st_mtime
            if file_age > max_age_seconds:
                lock_file.unlink()
                removed += 1
        except (OSError, FileNotFoundError):
            # File may have been removed by another process
            pass

    return removed
