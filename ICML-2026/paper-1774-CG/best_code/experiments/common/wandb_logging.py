"""Unified Weights & Biases logging for every experiment.

Every experiment logs into a single project (default ``calibrated-guidance``),
grouped by experiment line, with a consistent ``config`` schema (always carrying
an ``experiment`` key) and consistent metric namespacing. This makes runs from
the SBI benchmark, black-hole imaging and super-resolution directly comparable.

The driver owns the run via :class:`WandbRun`. Crucially, the run is *always*
initialised when ``wandb`` is importable — in **disabled** mode when logging is
turned off — so that any ``wandb.log(...)`` calls embedded in the copied
experiment code become safe no-ops without us editing that code.

Configuration via environment variables:

* ``CBG_WANDB_PROJECT``  — project name (default ``calibrated-guidance``).
* ``CBG_WANDB_ENTITY``   — W&B entity / team (default: your personal entity).
* ``CBG_WANDB``          — set to ``off``/``0``/``false`` to disable (runs in
                           wandb "disabled" mode: no network, no files).
* ``WANDB_MODE``         — standard W&B mode (``online``/``offline``/``disabled``);
                           takes precedence when set.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

DEFAULT_PROJECT = "calibrated-guidance"


def _disabled() -> bool:
    return os.environ.get("CBG_WANDB", "").lower() in {"off", "0", "false", "no"}


def _resolve_mode() -> str:
    env_mode = os.environ.get("WANDB_MODE")
    if env_mode:
        return env_mode
    return "disabled" if _disabled() else "online"


class WandbRun:
    """Context-manager wrapper around a single ``wandb`` run.

    Always initialises a run when ``wandb`` is importable (disabled mode when
    logging is off), so embedded ``wandb.log`` calls never crash. Falls back to a
    complete no-op only if ``wandb`` is not installed at all.
    """

    def __init__(
        self,
        experiment: str,
        config: Mapping[str, Any],
        run_name: Optional[str] = None,
    ):
        self.experiment = experiment
        self.config = dict(config)
        self.run_name = run_name
        self.run = None

    def __enter__(self) -> "WandbRun":
        try:
            import wandb
        except ImportError:
            self.run = None
            return self

        self.run = wandb.init(
            project=os.environ.get("CBG_WANDB_PROJECT", DEFAULT_PROJECT),
            entity=os.environ.get("CBG_WANDB_ENTITY") or None,
            group=self.experiment,
            name=self.run_name,
            config={"experiment": self.experiment, **self.config},
            mode=_resolve_mode(),
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )
        return self

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.run is not None:
            self.run.finish(exit_code=0 if exc_type is None else 1)
            self.run = None


def init_run(
    experiment: str,
    config: Mapping[str, Any],
    run_name: Optional[str] = None,
) -> WandbRun:
    """Create (and enter) a :class:`WandbRun`. Returns the entered run."""
    return WandbRun(experiment, config, run_name).__enter__()


def log(metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Log to the active global run, if any (no-op otherwise)."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(dict(metrics), step=step)


def finish() -> None:
    """Finish the active global run, if any."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
