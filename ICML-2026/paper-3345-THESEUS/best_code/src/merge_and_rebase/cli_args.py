from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from merge_and_rebase.utils.helpers import parse_csv


def merge_non_none(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def parse_json_object_arg(raw: str | None, *, arg_name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON object. Got: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object.")
    return parsed


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. CLI overrides config values.",
    )


def add_suite_arg(
    parser: argparse.ArgumentParser,
    *,
    choices: Sequence[str] | None = None,
    default: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {"type": str, "default": default}
    if choices is not None:
        kwargs["choices"] = list(choices)
    parser.add_argument("--suite", **kwargs)


def add_tasks_arg(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
    default: str | None = None,
) -> None:
    parser.add_argument("--tasks", type=str, default=default, help=help_text)


def add_device_dtype_args(
    parser: argparse.ArgumentParser,
    *,
    device_default: str | None,
    dtype_default: str | None,
) -> None:
    parser.add_argument("--device", type=str, default=device_default)
    parser.add_argument("--dtype", type=str, default=dtype_default, choices=[None, "fp16", "bf16", "fp32"])


def add_merge_io_args(
    parser: argparse.ArgumentParser,
    *,
    method_choices: Sequence[str],
    subspace_choices: Sequence[str] | None,
    tuned_help: str,
    weights_help: str,
    strict_mode: str,
) -> None:
    parser.add_argument("--base-ckpt", type=str, default=None)
    parser.add_argument("--tuned-ckpts", type=str, nargs="+", default=None, help=tuned_help)
    parser.add_argument("--weights", type=float, nargs="*", default=None, help=weights_help)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help=f"Merge method. Available: {', '.join(method_choices)}",
    )
    parser.add_argument("--method-params", type=str, default=None, help="JSON object for merge-method kwargs.")

    if strict_mode == "store_true":
        parser.add_argument("--strict-load", action="store_true", help="Fail on missing/unexpected keys")
    elif strict_mode == "bool_optional":
        parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=None)
    else:
        raise ValueError(f"Unsupported strict_mode='{strict_mode}'.")

    if subspace_choices is not None:
        parser.add_argument(
            "--peft-subspace",
            type=str,
            choices=list(subspace_choices),
            default=None,
            help=f"Subspace for merging PEFT checkpoints. Available: {', '.join(subspace_choices)}.",
        )

    parser.add_argument("--save-merged", type=str, default=None, help="Save merged state_dict to this path.")


def add_alpha_args(
    parser: argparse.ArgumentParser,
    *,
    alpha_default: float | None,
    alpha_min_default: float | None,
    alpha_max_default: float | None,
    alpha_step_default: float | None,
    alpha_search_default: bool | None,
    alpha_search_help: str | None = None,
) -> None:
    parser.add_argument(
        "--alpha-search",
        action=argparse.BooleanOptionalAction,
        default=alpha_search_default,
        help=alpha_search_help,
    )
    parser.add_argument(
        "--alpha-early-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop alpha search early when the average normalized accuracy starts decreasing.",
    )
    parser.add_argument("--alpha-min", type=float, default=alpha_min_default)
    parser.add_argument("--alpha-max", type=float, default=alpha_max_default)
    parser.add_argument("--alpha-step", type=float, default=alpha_step_default)
    parser.add_argument("--alpha", type=float, default=alpha_default)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--local-log-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--log-every-n-steps", type=int, default=None)


def add_postmerge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--postmerge-method", type=str, default=None)
    parser.add_argument("--postmerge-alpha-mode", type=str, choices=["task", "layer"], default=None)
    parser.add_argument("--postmerge-loss", type=str, choices=["ce", "entropy"], default=None)
    parser.add_argument("--postmerge-steps", type=int, default=None)
    parser.add_argument("--postmerge-lr", type=float, default=None)
    parser.add_argument("--postmerge-max-batches-per-task", type=int, default=None)


def build_postmerge_overrides(args: argparse.Namespace) -> dict[str, Any]:
    raw = {
        "method": getattr(args, "postmerge_method", None),
        "alpha_mode": getattr(args, "postmerge_alpha_mode", None),
        "loss": getattr(args, "postmerge_loss", None),
        "steps": getattr(args, "postmerge_steps", None),
        "lr": getattr(args, "postmerge_lr", None),
        "max_batches_per_task": getattr(args, "postmerge_max_batches_per_task", None),
    }
    out = {k: v for k, v in raw.items() if v is not None}
    return {"postmerge": out} if out else {}


def build_logging_overrides(args: argparse.Namespace) -> dict[str, Any]:
    tags_raw = getattr(args, "wandb_tags", None)
    tags = parse_csv(tags_raw) if isinstance(tags_raw, str) and tags_raw.strip() else None
    return {
        "use_wandb": getattr(args, "use_wandb", None),
        "project": getattr(args, "wandb_project", None),
        "entity": getattr(args, "wandb_entity", None),
        "tags": tags,
        "mode": getattr(args, "wandb_mode", None),
        "local_log_dir": getattr(args, "local_log_dir", None),
        "run_name": getattr(args, "run_name", None),
        "log_every_n_steps": getattr(args, "log_every_n_steps", None),
    }


def build_common_eval_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "suite": getattr(args, "suite", None),
        "tasks": getattr(args, "tasks", None),
        "device": getattr(args, "device", None),
        "dtype": getattr(args, "dtype", None),
    }


def build_common_merge_overrides(
    *,
    args: argparse.Namespace,
    method_params: dict[str, Any] | None,
    strict_as_bool: bool,
) -> dict[str, Any]:
    strict_value = bool(getattr(args, "strict_load", False)) if strict_as_bool else getattr(args, "strict_load", None)
    return {
        "base_ckpt": getattr(args, "base_ckpt", None),
        "tuned_ckpts": getattr(args, "tuned_ckpts", None),
        "weights": getattr(args, "weights", None),
        "method": getattr(args, "method", None),
        "method_params": method_params,
        "strict_load": strict_value,
        "peft_subspace": getattr(args, "peft_subspace", None),
        "alpha_search": getattr(args, "alpha_search", None),
        "alpha_early_stop": getattr(args, "alpha_early_stop", None),
        "alpha_min": getattr(args, "alpha_min", None),
        "alpha_max": getattr(args, "alpha_max", None),
        "alpha_step": getattr(args, "alpha_step", None),
        "alpha": getattr(args, "alpha", None),
        "save_merged": getattr(args, "save_merged", None),
    }
