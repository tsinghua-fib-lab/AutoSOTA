from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def prepare_log_dir(args, task: str, dataset_label: str, model_label: str, root: str | Path = "logs") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    log_dir = Path(root) / task / dataset_label / model_label / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def save_metrics(log_dir: Path, src_dir: str | Path, metadata: Dict[str, Any], args: Any, metrics: Dict[str, Any]) -> Path:
    if isinstance(args, dict):
        args_payload = args
    elif hasattr(args, "__dict__"):
        args_payload = vars(args)
    else:
        args_payload = args
    out = {
        "src_dir": str(src_dir),
        "metadata": metadata,
        "args": args_payload,
        "metrics": metrics,
    }
    fpath = Path(log_dir) / "metrics.json"
    with open(fpath, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=True, indent=2, default=_json_default)
    return fpath


def _json_default(obj):
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    try:
        import torch  # type: ignore
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    return str(obj)


def setup_file_logger(log_dir: Path, name: str = "runner") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(getattr(h, "_runner_handler", False) for h in logger.handlers):
        fh = logging.FileHandler(Path(log_dir) / "runner.log")
        fh.setLevel(logging.INFO)
        fh._runner_handler = True
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def log_and_print_metrics(
    stats: Dict[str, Any],
    alpha: float | None = None,
    target_cov: float | None = None,
    logger: logging.Logger | None = None,
    print_fn=None,
) -> None:
    if print_fn is None:
        print("Metrics:", stats)
    else:
        print_fn(stats, alpha=alpha, target_cov=target_cov)

    if logger is not None:
        ok_cov = stats.get("observed_coverage", None)
        piw = stats.get("pi_width", None)
        wink = stats.get("winkler", None)
        logger.info(
            "alpha=%s target_cov=%s observed_cov=%s pi_width=%s winkler=%s",
            alpha,
            target_cov,
            ok_cov,
            piw,
            wink,
        )
