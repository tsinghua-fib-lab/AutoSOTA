from pathlib import Path
from typing import Any, Dict, Optional, Callable
import yaml
import pickle
import os
from datetime import datetime
from accelerate import Accelerator
from config.train_config import TrainingConfig
from os_utilities import smart_pickle


def merge_results_artifacts(
    metrics_dir: Path,
    new_metrics: Dict[str, Any],
    *,
    backfill_if_missing: Optional[Dict[str, Any]] = None,
    yaml_filename: str = "results.yaml",
    pkl_filename: Optional[str] = "results.pkl",
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Merge existing YAML/PKL fold metrics with new entries and optional backfill.
    """
    merged: Dict[str, Any] = {}
    yaml_path = metrics_dir / yaml_filename
    pkl_path = metrics_dir / pkl_filename if pkl_filename else None

    try:
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing_yaml = yaml.safe_load(f) or {}
            if isinstance(existing_yaml, dict):
                merged.update(existing_yaml)
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"[WARNING] merge_results_artifacts: failed reading {yaml_path}: {exc}")

    if pkl_path is not None and pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                existing_pickle = pickle.load(f) or {}
            if isinstance(existing_pickle, dict):
                for key, value in existing_pickle.items():
                    if key not in merged:
                        merged[key] = value
        except Exception as exc:
            if log_fn is not None:
                log_fn(f"[WARNING] merge_results_artifacts: failed reading {pkl_path}: {exc}")

    merged.update({k: v for k, v in new_metrics.items() if v is not None})

    if backfill_if_missing:
        for key, value in backfill_if_missing.items():
            if key not in merged and value is not None:
                merged[key] = value

    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f)
    except Exception as exc:
        if log_fn is not None:
            log_fn(f"[WARNING] merge_results_artifacts: failed writing {yaml_path}: {exc}")

    if pkl_path is not None:
        try:
            smart_pickle(str(pkl_path), merged, overwrite=True)
        except Exception as exc:
            if log_fn is not None:
                log_fn(f"[WARNING] merge_results_artifacts: failed writing {pkl_path}: {exc}")

    return merged


def propagate_training_summary_to_context(
    context: Any,
    *,
    best_epoch: Any,
    mean_train_time: Any,
    mean_infer_time: Any,
) -> None:
    """
    Attach training summary stats to a mutable context dict for downstream hooks.
    """
    if not isinstance(context, dict):
        return
    payload = {
        "best_epoch": best_epoch,
        "mean_train_time": mean_train_time,
        "mean_infer_time": mean_infer_time,
    }
    for key, value in payload.items():
        if value is None:
            continue
        context[key] = value


def main_print(*args, timestamp=False, **kwargs):
    # Helper function for main process printing
    # Expects 'acc' or 'accelerator' to be in the caller's scope
    # (or pass as a kwarg if needed)
    acc = kwargs.pop('acc', None) or kwargs.pop('accelerator', None)
    config = kwargs.pop('config', None)
    if acc is not None:
        if acc.is_main_process:
            if timestamp:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{current_time}]", *args, **kwargs)
            else:
                print(*args, **kwargs)
            # Only main process writes to log file
            if config is not None and hasattr(config, 'train_logs_save_dir') \
            and (config.train_logs_save_dir is not None):
                log_filepath = os.path.join(
                    config.train_logs_save_dir,
                    getattr(config, 'train_logs_filename', 'logs.txt')
                )
                with open(log_filepath, 'a') as f:
                    if timestamp:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"[{current_time}] {' '.join(map(str, args))}\n")
                    else:
                        f.write(f"{' '.join(map(str, args))}\n")
    else:
        # Fallback: just print
        if timestamp:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}]", *args, **kwargs)
        else:
            print(*args, **kwargs)


def log_to_train_print(
    out: str,
    acc: Accelerator,
    config: TrainingConfig,
) -> None:
    """
    Append a line to the training print log file (train_print.txt) on the main process.
    """
    try:
        if acc.is_main_process and (config.train_logs_save_dir is not None):
            log_filepath = os.path.join(
                config.train_logs_save_dir,
                config.train_print_filename
            )
            with open(log_filepath, 'a') as f:
                f.write(out + '\n')
    except Exception as e:
        # Fallback warning; avoid raising to not disrupt training flow
        main_print(f"[WARNING] Could not write to train_print log: {e}", acc=acc, config=config)