"""
Evaluation orchestration driven by the ``eval`` section of the YAML config.
"""
from __future__ import annotations

from typing import Any

from utils.evaluation import EvaluationManager


def run_evaluation(
    training_results,
    cfg: dict[str, Any],
    ssp_space,
    *,
    test_dir: str,
    device: str,
    batch_size: int,
) -> None:
    ev = cfg.get("eval", {})
    tr = cfg.get("trainer", {})
    noise_type = ev.get("noise_type", tr.get("noise_type", "uniform_hypersphere"))
    target_type = ev.get("target_type", tr.get("target_type", "coordinate"))
    mgr = EvaluationManager(
        training_results,
        test_dir=test_dir,
        device=device,
        signal_strengths=ev.get("signal_strengths"),
        eval_steps=ev.get("num_steps"),
        repeats=ev.get("repeats", 5),
        noise_type=str(noise_type),
        target_type=str(target_type),
    )
    mgr.run_all(ssp_space, batch_size=batch_size)
