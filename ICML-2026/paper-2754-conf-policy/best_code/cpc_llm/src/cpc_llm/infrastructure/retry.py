"""Shared retry utilities for CUDA initialization with exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_module_logger = logging.getLogger(__name__)


def is_cuda_error(exc: Exception) -> bool:
    """Check whether an exception is a CUDA-related error.

    Detects CUDA busy/unavailable errors, AcceleratorError, and other
    GPU-related failures that may be transient in multi-process environments.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception appears to be CUDA-related.
    """
    error_str = str(exc)
    return (
        "CUDA" in error_str
        or "busy" in error_str.lower()
        or "unavailable" in error_str.lower()
        or "AcceleratorError" in type(exc).__name__
    )


def cuda_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    stagger: bool = True,
    on_retry: Callable[[], None] | None = None,
    logger: logging.Logger | None = None,
    operation: str = "operation",
) -> T:
    """Call *fn* with exponential-backoff retries on CUDA errors.

    Useful for wrapping GPU-dependent initialization that may fail transiently
    when multiple processes race to acquire CUDA devices.

    Args:
        fn: Zero-argument callable to execute.
        max_retries: Maximum number of attempts before giving up.
        base_delay: Base delay in seconds (doubled each retry).
        stagger: If True, sleep a random 0.1–0.5 s before the first attempt
            to reduce thundering-herd effects across processes.
        on_retry: Optional callback invoked after the backoff sleep but
            before the next attempt (e.g. ``wait_for_gpu_availability``).
        logger: Logger for warning/error messages. Falls back to the
            module-level logger if not provided.
        operation: Human-readable label included in log messages.

    Returns:
        The return value of *fn* on success.

    Raises:
        ValueError: If *max_retries* < 1.
        Exception: Re-raises the last CUDA error after *max_retries*
            attempts, or immediately for non-CUDA errors.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    _logger = logger or _module_logger

    if stagger:
        time.sleep(random.uniform(0.1, 0.5))

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not is_cuda_error(e):
                raise

            if attempt < max_retries - 1:
                wait_time = base_delay * (2**attempt)
                _logger.warning(
                    f"{operation} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                if on_retry is not None:
                    on_retry()
            else:
                _logger.error(f"{operation} failed after {max_retries} attempts: {e}")
                raise

    # Unreachable, but satisfies type checkers.
    raise AssertionError("unreachable")  # pragma: no cover
