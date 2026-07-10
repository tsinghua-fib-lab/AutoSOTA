#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_SUITES = ("vision8", "vision14", "vision20")
DEFAULT_MERGE_METHODS = (
    "weighted_average",
    "task_arithmetic",
    "ties_merge",
    "dare_merge",
    "tsv_merge",
    "isoc_merge",
    "isocts_merge",
    "cart_merge",
    "pcb",
)
DEFAULT_REBASE_METHODS = ("identity", "orthogonal_shift", "gradfix", "theseus")

MODEL_ALIASES = {
    "b": "vitb",
    "base": "vitb",
    "vit-b": "vitb",
    "vit_b": "vitb",
    "vitb": "vitb",
    "vit-b-32": "vitb",
    "vit-b/32": "vitb",
    "l": "vitl",
    "large": "vitl",
    "vit-l": "vitl",
    "vit_l": "vitl",
    "vitl": "vitl",
    "vit-l-14": "vitl",
    "vit-l/14": "vitl",
}


def log(message: str) -> None:
    print(f"[vision-experiments] {message}", flush=True)


def split_csv_args(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    if not values:
        return list(default)
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def normalize_model(raw: str) -> str:
    key = raw.strip().lower()
    try:
        return MODEL_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(set(MODEL_ALIASES)))
        raise argparse.ArgumentTypeError(f"Unknown model '{raw}'. Try one of: {choices}") from exc


def model_label(model: str) -> str:
    return "vitl" if model == "vitl" else "vitb"


def default_results_root(model: str) -> Path:
    if env_root := os.environ.get("RESULTS_ROOT"):
        return Path(env_root)
    if model == "vitl":
        return Path("results/paper/vision_L14")
    return Path("results/paper/vision")


def default_device(model: str) -> str:
    if env_device := os.environ.get("DEVICE"):
        return env_device
    if model == "vitl":
        return "cuda:1"
    return "cuda"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def merge_config_for_suite(suite: str, model: str) -> Path:
    suffix = "_vitl" if model == "vitl" else ""
    path = Path("configs") / f"{suite}_task_arithmetic{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing merge config for {model}/{suite}: {path}")
    return path


def default_rebase_template(model: str) -> Path:
    # Preserve the behavior of the two previous shell launchers.
    if model == "vitl":
        return Path("configs/vision8_gradfix_rebase.json")
    return Path("configs/vision8_gradfix.json")


def default_theseus_template() -> Path:
    return Path("configs/vision8_theseus_all_alpha_sweep.json")


def merge_method_params(method: str) -> str | None:
    if method == "ties_merge":
        return json.dumps(
            {
                "topk": 1.0,
                "merging_type": "mean",
                "low_memory": True,
                "cache_prepared": False,
            },
            separators=(",", ":"),
        )
    if method == "dare_merge":
        return json.dumps({"low_memory": True, "cache_prepared": False}, separators=(",", ":"))
    return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_rebase_config(
    *,
    suite: str,
    method: str,
    model: str,
    out_path: Path,
    rebase_template: Path,
    theseus_template: Path,
    tasks: str | None,
) -> None:
    if method == "theseus":
        payload = read_json(theseus_template)
        if tasks:
            payload["tasks"] = tasks
        write_json(out_path, payload)
        return

    template = read_json(rebase_template)
    suite_cfg = read_json(merge_config_for_suite(suite, model))

    template["suite"] = suite
    template["tasks"] = tasks or "all"
    template["method"] = method
    template["tuned_ckpts"] = suite_cfg["tuned_ckpts"]
    template["weights"] = suite_cfg.get("weights")

    if method != "gradfix":
        template.pop("mask_mode", None)
        template.pop("vote", None)
        template["method_params"] = {}

    write_json(out_path, template)


def is_run_complete(out_dir: Path) -> bool:
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        payload = read_json(summary_path)
    except Exception:
        return False
    status = str((payload.get("run_logging") or {}).get("status") or "").strip().lower()
    return status == "success"


def extend_if_value(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def append_extra_args(cmd: list[str], raw: str | None) -> None:
    if raw:
        cmd.extend(shlex.split(raw))


def merged_checkpoint_path(out_dir: Path, checkpoint_name: str) -> Path:
    return out_dir / checkpoint_name


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    log(shlex.join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not existing_pythonpath else f"src{os.pathsep}{existing_pythonpath}"
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def alpha_flags(args: argparse.Namespace, *, theseus: bool = False) -> list[str]:
    if not args.alpha_search:
        flags = ["--no-alpha-search"]
        if args.alpha is not None:
            flags.extend(["--alpha", str(args.alpha)])
        return flags

    if theseus:
        return [
            "--alpha-search",
            "--alpha-min",
            str(args.theseus_alpha_min),
            "--alpha-max",
            str(args.theseus_alpha_max),
            "--alpha-step",
            str(args.theseus_alpha_step),
            "--alpha-selection",
            args.theseus_alpha_selection,
            "--alpha-patience",
            str(args.theseus_alpha_patience),
        ]
    return [
        "--alpha-search",
        "--alpha-min",
        str(args.alpha_min),
        "--alpha-max",
        str(args.alpha_max),
        "--alpha-step",
        str(args.alpha_step),
    ]


def run_merge_case(args: argparse.Namespace, *, suite: str, method: str, results_root: Path) -> None:
    out_dir = results_root / "merge" / suite / method
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and is_run_complete(out_dir):
        log(f"Skipping merge/{suite}/{method}; summary.json already marked success.")
        return
    merge_config = Path(args.merge_config) if getattr(args, "merge_config", None) is not None else merge_config_for_suite(suite, args.model)

    cmd = [
        args.python_bin,
        "-m",
        "merge_and_rebase.eval.vision_merge",
        "--config",
        str(merge_config),
        "--method",
        method,
        "--device",
        args.device,
        "--single-acc-cache",
        str(args.cache_path),
        "--local-log-dir",
        str(out_dir),
        "--run-name",
        args.run_name,
        *alpha_flags(args),
    ]
    if args.tasks:
        cmd.extend(["--tasks", args.tasks])
    extend_if_value(cmd, "--dtype", args.dtype)
    extend_if_value(cmd, "--batch-size", args.batch_size)
    extend_if_value(cmd, "--num-workers", args.num_workers)
    extend_if_value(cmd, "--seed", args.seed)
    if args.wandb_mode:
        cmd.extend(["--wandb-mode", args.wandb_mode])
    if args.save_merged_checkpoints:
        cmd.extend(["--save-merged", str(merged_checkpoint_path(out_dir, args.save_merged_name))])

    method_params = merge_method_params(method)
    if method_params:
        cmd.extend(["--method-params", method_params])
    append_extra_args(cmd, args.merge_extra)
    run_cmd(cmd, dry_run=args.dry_run)


def run_rebase_case(
    args: argparse.Namespace,
    *,
    suite: str,
    method: str,
    results_root: Path,
    rebase_template: Path,
    theseus_template: Path,
) -> None:
    if method == "theseus" and suite != "vision8":
        log(f"Skipping rebase/{suite}/{method}; repo only has a full-suite Theseus config for vision8.")
        return

    out_dir = results_root / "rebase" / suite / method
    run_cfg = out_dir / "run_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and is_run_complete(out_dir):
        log(f"Skipping rebase/{suite}/{method}; summary.json already marked success.")
        return

    write_rebase_config(
        suite=suite,
        method=method,
        model=args.model,
        out_path=run_cfg,
        rebase_template=rebase_template,
        theseus_template=theseus_template,
        tasks=args.tasks,
    )

    cmd = [
        args.python_bin,
        "-m",
        "merge_and_rebase.eval.vision_rebase",
        "--config",
        str(run_cfg),
        "--device",
        args.device,
        "--local-log-dir",
        str(out_dir),
        "--run-name",
        args.run_name,
        *alpha_flags(args, theseus=(method == "theseus")),
    ]
    if args.tasks:
        cmd.extend(["--tasks", args.tasks])
    extend_if_value(cmd, "--dtype", args.dtype)
    extend_if_value(cmd, "--batch-size", args.batch_size)
    extend_if_value(cmd, "--num-workers", args.num_workers)
    extend_if_value(cmd, "--seed", args.seed)
    if args.wandb_mode:
        cmd.extend(["--wandb-mode", args.wandb_mode])
    append_extra_args(cmd, args.rebase_extra)
    run_cmd(cmd, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the vision merge/rebase experiment sweeps from one CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=normalize_model, default="vitb", help="Backbone family: vitb/b or vitl/l.")
    parser.add_argument(
        "--family",
        "--run",
        dest="family",
        choices=("all", "merge", "rebase"),
        default="all",
        help="Experiment family to launch.",
    )
    parser.add_argument("--device", default=None, help="Torch device passed to the eval entrypoints.")
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used for child runs.",
    )
    parser.add_argument("--results-root", type=Path, default=None, help="Root directory for generated logs/results.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path(os.environ.get("CACHE_PATH", "src/.cache/single_task_acc.json")),
    )
    parser.add_argument("--suites", nargs="+", default=None, help="Suites as space- or comma-separated values.")
    parser.add_argument("--merge-methods", nargs="+", default=None, help="Merge methods as space- or comma-separated values.")
    parser.add_argument("--rebase-methods", nargs="+", default=None, help="Rebase methods as space- or comma-separated values.")
    parser.add_argument("--tasks", default=None, help="Optional comma-separated task override, or 'all'.")
    parser.add_argument("--run-name", default="summary")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=env_bool("DRY_RUN"),
        help="Print commands without executing eval entrypoints.",
    )
    parser.add_argument("--force", action="store_true", help="Run even when summary.json is already marked success.")

    parser.add_argument("--alpha-search", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha when --no-alpha-search is used.")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--theseus-alpha-min", type=float, default=0.8)
    parser.add_argument("--theseus-alpha-max", type=float, default=10.0)
    parser.add_argument("--theseus-alpha-step", type=float, default=0.2)
    parser.add_argument("--theseus-alpha-selection", choices=("shared", "per_task"), default="per_task")
    parser.add_argument("--theseus-alpha-patience", type=int, default=5)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument(
        "--save-merged-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=env_bool("SAVE_MERGED_CHECKPOINTS", env_bool("SAVE_MERGED", False)),
        help="For merge runs, save each selected best-alpha checkpoint inside that run's output directory.",
    )
    parser.add_argument(
        "--save-merged-name",
        default=os.environ.get("SAVE_MERGED_NAME", "merged.pt"),
        help="Filename, or relative path, used below each merge run directory when saving merged checkpoints.",
    )

    parser.add_argument(
        "--rebase-template",
        type=Path,
        default=None,
        help="Template JSON for generated non-Theseus rebase configs.",
    )
    parser.add_argument(
        "--theseus-template",
        type=Path,
        default=default_theseus_template(),
        help="Template JSON for generated Theseus rebase configs.",
    )
    parser.add_argument("--merge-extra", default=None, help="Extra quoted args appended to vision_merge.")
    parser.add_argument("--rebase-extra", default=None, help="Extra quoted args appended to vision_rebase.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.model = model_label(args.model)
    args.device = args.device or default_device(args.model)
    suites = split_csv_args(args.suites, DEFAULT_SUITES)
    merge_methods = split_csv_args(args.merge_methods, DEFAULT_MERGE_METHODS)
    rebase_methods = split_csv_args(args.rebase_methods, DEFAULT_REBASE_METHODS)
    results_root = args.results_root or default_results_root(args.model)
    rebase_template = args.rebase_template or default_rebase_template(args.model)
    theseus_template = args.theseus_template

    if not rebase_template.exists():
        raise FileNotFoundError(f"Missing rebase template: {rebase_template}")
    if not theseus_template.exists():
        raise FileNotFoundError(f"Missing Theseus template: {theseus_template}")

    (ROOT_DIR / results_root).mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / args.cache_path).parent.mkdir(parents=True, exist_ok=True)

    suite_label = ",".join(suites)
    log(f"model={args.model} family={args.family} device={args.device} suites={suite_label} results={results_root}")

    if args.family in {"all", "merge"}:
        for suite in suites:
            for method in merge_methods:
                run_merge_case(args, suite=suite, method=method, results_root=results_root)

    if args.family in {"all", "rebase"}:
        for suite in suites:
            for method in rebase_methods:
                run_rebase_case(
                    args,
                    suite=suite,
                    method=method,
                    results_root=results_root,
                    rebase_template=rebase_template,
                    theseus_template=theseus_template,
                )


if __name__ == "__main__":
    main()
