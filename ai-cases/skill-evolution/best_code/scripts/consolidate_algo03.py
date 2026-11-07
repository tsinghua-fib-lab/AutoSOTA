#!/usr/bin/env python3
"""Run ALGO-03 consolidation on an existing SkillOpt/ALGO-02 run.

This script reuses completed training trajectories from ``--source_out_root``
and writes the consolidated ALGO-03 artifacts to ``env.out_root`` from the
config. It does not mutate the source run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_project_dotenv() -> None:
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_project_dotenv()

from skillopt.config import flatten_config, load_config
from skillopt.model import (
    configure_openai_compatible,
    get_token_summary,
    get_returned_models,
    reset_token_tracker,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from skillopt.model.common import default_model_for_backend, normalize_backend_name
from skillopt.optimizer.algo03 import run_algo03_consolidation
from skillopt.utils import compute_score
from skillopt.utils.console import force_utf8_stdout_stderr
from skillopt.utils.provenance import build_provenance, file_sha256

from scripts.eval_only import get_adapter


force_utf8_stdout_stderr()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _configure_models(cfg: dict) -> None:
    backend = normalize_backend_name(
        cfg.get("model_backend") or cfg.get("target_backend") or "openai_chat"
    )
    set_optimizer_backend(cfg.get("optimizer_backend", "openai_chat"))
    set_target_backend(cfg.get("target_backend", "openai_chat"))
    set_optimizer_deployment(cfg.get("optimizer_model", default_model_for_backend(backend)))
    set_target_deployment(cfg.get("target_model", default_model_for_backend(backend)))
    configure_openai_compatible(
        base_url=cfg.get("openai_compatible_base_url")
        or os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
        or None,
        api_key=cfg.get("openai_compatible_api_key")
        or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
        or None,
        model=cfg.get("openai_compatible_model")
        or os.environ.get("OPENAI_COMPATIBLE_MODEL")
        or None,
        max_tokens=cfg.get("openai_compatible_max_tokens")
        or os.environ.get("OPENAI_COMPATIBLE_MAX_TOKENS")
        or None,
        optimizer_base_url=cfg.get("optimizer_openai_compatible_base_url")
        or os.environ.get("OPTIMIZER_OPENAI_COMPATIBLE_BASE_URL")
        or None,
        optimizer_api_key=cfg.get("optimizer_openai_compatible_api_key")
        or os.environ.get("OPTIMIZER_OPENAI_COMPATIBLE_API_KEY")
        or None,
        optimizer_model=cfg.get("optimizer_openai_compatible_model") or None,
        target_base_url=cfg.get("target_openai_compatible_base_url")
        or os.environ.get("TARGET_OPENAI_COMPATIBLE_BASE_URL")
        or None,
        target_api_key=cfg.get("target_openai_compatible_api_key")
        or os.environ.get("TARGET_OPENAI_COMPATIBLE_API_KEY")
        or None,
        target_model=cfg.get("target_openai_compatible_model") or None,
    )
    set_reasoning_effort(cfg.get("reasoning_effort", "") or None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALGO-03 consolidation for an existing run")
    parser.add_argument("--config", required=True, help="Config for env/model/output settings")
    parser.add_argument("--source_out_root", required=True, help="Existing ALGO-02 run directory")
    parser.add_argument("--case_source_root", default="", help="Development rollouts under the skill being consolidated")
    parser.add_argument("--out_root", default="", help="Isolated output directory for this comparison arm")
    parser.add_argument("--skill", default="", help="Skill file to consolidate; defaults to best step")
    parser.add_argument("--baseline_hard", type=float, default=None)
    parser.add_argument("--baseline_soft", type=float, default=None)
    parser.add_argument("--validation_split", default="valid_seen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = flatten_config(load_config(args.config))
    if args.out_root:
        cfg["out_root"] = args.out_root

    source_root = Path(args.source_out_root).resolve()
    source_summary = _read_json(source_root / "summary.json")
    source_cfg = source_summary.get("config") if isinstance(source_summary.get("config"), dict) else {}

    for key, value in source_cfg.items():
        cfg.setdefault(key, value)

    out_root = Path(cfg["out_root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    best_step = source_summary.get("best_step")
    skill_path = Path(args.skill).resolve() if args.skill else source_root / "skills" / f"skill_v{int(best_step):04d}.md"
    skill = skill_path.read_text(encoding="utf-8")

    baseline_hard = (
        args.baseline_hard
        if args.baseline_hard is not None
        else float(source_summary.get("best_selection_hard"))
    )
    baseline_soft = (
        args.baseline_soft
        if args.baseline_soft is not None
        else source_summary.get("final_selection_soft")
    )
    if baseline_soft is not None:
        baseline_soft = float(baseline_soft)

    cfg["algo03_case_source_root"] = str(Path(args.case_source_root).resolve()) if args.case_source_root else str(source_root)
    cfg.setdefault("algo03_enabled", True)
    cfg.setdefault("algo03_max_failure_cases", 32)
    cfg.setdefault("algo03_max_success_cases", 32)
    cfg.setdefault("algo03_max_prune_candidates", 4)
    cfg.setdefault("algo03_max_demos", 5)
    cfg.setdefault("algo03_validation_gate", True)
    cfg.setdefault("algo03_rewrite_reasoning_effort", cfg.get("rewrite_reasoning_effort") or "high")
    cfg.setdefault("algo03_max_completion_tokens", cfg.get("rewrite_max_completion_tokens") or 64000)

    _write_json(out_root / "config.json", cfg)
    shutil.copyfile(skill_path, out_root / "source_best_skill.md")

    reset_token_tracker()
    _configure_models(cfg)

    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    seed = int(cfg.get("seed", 42) or 42)
    validation_split = args.validation_split

    def validate(skill_text: str, label: str) -> tuple[float, float]:
        env_num = int(cfg.get("sel_env_num", 0) or 0)
        items = adapter.build_eval_env(env_num, validation_split, seed)
        eval_dir = out_root / "algo03" / "validation" / label
        print(f"  [ALGO-03 validate] {label}: split={validation_split} items={len(items)}")
        results = adapter.rollout(items, skill_text, str(eval_dir))
        return compute_score(results)

    print("  [ALGO-03] source:", source_root)
    print("  [ALGO-03] skill:", skill_path)
    print("  [ALGO-03] out_root:", out_root)
    print(f"  [ALGO-03] baseline_hard={baseline_hard:.4f}")

    result = run_algo03_consolidation(
        out_root=str(out_root),
        skill=skill,
        validate=validate,
        baseline_hard=baseline_hard,
        baseline_soft=baseline_soft,
        cfg=cfg,
    )

    final_skill_path = out_root / "best_skill.md"
    final_skill_path.write_text(result.skill, encoding="utf-8")
    _write_json(
        out_root / "summary.json",
        {
            "source_out_root": str(source_root),
            "source_skill": str(skill_path),
            "source_skill_sha256": file_sha256(skill_path),
            "baseline_hard": baseline_hard,
            "baseline_soft": baseline_soft,
            "algo03": result.report,
            "validation_hard": result.validation_hard,
            "validation_soft": result.validation_soft,
            "accepted": result.accepted,
            "token_summary": get_token_summary(),
            "returned_models": get_returned_models(),
            "provenance": build_provenance(
                _PROJECT_ROOT,
                extra_files=[os.path.relpath(args.config, _PROJECT_ROOT)]
                if os.path.isabs(args.config)
                else [args.config],
            ),
        },
    )

    print(
        "  [ALGO-03 done] "
        f"accepted={result.accepted} validation_hard={result.validation_hard} "
        f"final_skill={final_skill_path}"
    )


if __name__ == "__main__":
    main()
