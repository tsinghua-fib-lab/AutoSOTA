#!/usr/bin/env python3
"""Run iterative ALGO-03 consolidation rounds.

This runner keeps the same ALGO-03 implementation used by
``scripts/consolidate_algo03.py`` but applies it repeatedly. Each round uses a
rotated pool of ALGO-02 trajectory cases, validates the generated candidate on
the configured validation split, and only updates the current exported skill
when the validation gate accepts it.
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

from scripts.consolidate_algo03 import _configure_models, _load_project_dotenv
from scripts.eval_only import get_adapter
from skillopt.config import flatten_config, load_config
from skillopt.model import get_token_summary, reset_token_tracker
from skillopt.optimizer.algo03 import run_algo03_consolidation
from skillopt.utils import compute_score
from skillopt.utils.console import force_utf8_stdout_stderr
from skillopt.utils.provenance import build_provenance, file_sha256


force_utf8_stdout_stderr()
_load_project_dotenv()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source_out_root", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--baseline_hard", type=float, default=None)
    parser.add_argument("--baseline_soft", type=float, default=None)
    parser.add_argument("--validation_split", default="valid_seen")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--case_stride", type=int, default=16)
    parser.add_argument("--case_pool_multiplier", type=int, default=4)
    parser.add_argument("--cfg-options", nargs="+", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")

    cfg = flatten_config(load_config(args.config, overrides=args.cfg_options))
    source_root = Path(args.source_out_root).resolve()
    source_summary = _read_json(source_root / "summary.json")
    source_cfg = source_summary.get("config") if isinstance(source_summary.get("config"), dict) else {}
    for key, value in source_cfg.items():
        cfg.setdefault(key, value)

    base_out_root = Path(cfg["out_root"]).resolve()
    base_out_root.mkdir(parents=True, exist_ok=True)

    current_skill_path = Path(args.skill).resolve()
    current_skill = current_skill_path.read_text(encoding="utf-8")
    current_hard = (
        args.baseline_hard
        if args.baseline_hard is not None
        else float(source_summary.get("best_selection_hard"))
    )
    current_soft = (
        args.baseline_soft
        if args.baseline_soft is not None
        else source_summary.get("final_selection_soft")
    )
    if current_soft is not None:
        current_soft = float(current_soft)

    cfg["algo03_case_source_root"] = str(source_root)
    cfg["algo03_case_pool_multiplier"] = int(args.case_pool_multiplier)
    cfg.setdefault("algo03_enabled", True)
    cfg.setdefault("algo03_validation_gate", True)

    reset_token_tracker()
    _configure_models(cfg)
    adapter = get_adapter(cfg)
    adapter.setup(cfg)
    seed = int(cfg.get("seed", 42) or 42)
    validation_split = args.validation_split
    eval_n = int(cfg.get("sel_env_num", 0) or 0)
    validation_items = adapter.build_eval_env(eval_n, validation_split, seed)

    round_records: list[dict] = []
    round_memory: list[dict] = []
    no_improve = 0
    best_skill = current_skill
    best_hard = current_hard
    best_soft = current_soft

    for round_idx in range(1, args.rounds + 1):
        round_root = base_out_root / f"round_{round_idx:04d}"
        round_root.mkdir(parents=True, exist_ok=True)
        round_cfg = dict(cfg)
        round_cfg["out_root"] = str(round_root)
        round_cfg["algo03_case_offset"] = (round_idx - 1) * int(args.case_stride)
        round_cfg["algo03_round_memory"] = list(round_memory)

        def validate(skill_text: str, label: str) -> tuple[float, float]:
            eval_dir = round_root / "algo03" / "validation" / label
            print(
                f"  [ALGO-03 iterative validate] round={round_idx} "
                f"label={label}: split={validation_split} items={len(validation_items)}"
            )
            results = adapter.rollout(validation_items, skill_text, str(eval_dir))
            return compute_score(results)

        print(
            f"  [ALGO-03 iterative] round={round_idx} "
            f"baseline_hard={current_hard:.4f} case_offset={round_cfg['algo03_case_offset']}"
        )
        result = run_algo03_consolidation(
            out_root=str(round_root),
            skill=current_skill,
            validate=validate,
            baseline_hard=current_hard,
            baseline_soft=current_soft,
            cfg=round_cfg,
        )

        candidate_hard = result.validation_hard
        candidate_soft = result.validation_soft
        accepted = bool(result.accepted)
        improved = accepted and candidate_hard is not None and candidate_hard > current_hard
        if accepted:
            current_skill = result.skill
            current_hard = float(candidate_hard)
            current_soft = candidate_soft
        if improved:
            best_skill = result.skill
            best_hard = float(candidate_hard)
            best_soft = candidate_soft
            no_improve = 0
        else:
            no_improve += 1

        record = {
            "round": round_idx,
            "round_root": str(round_root),
            "case_offset": round_cfg["algo03_case_offset"],
            "accepted": accepted,
            "improved": improved,
            "candidate_hard": candidate_hard,
            "candidate_soft": candidate_soft,
            "current_hard_after_round": current_hard,
            "current_soft_after_round": current_soft,
            "best_hard_after_round": best_hard,
            "best_soft_after_round": best_soft,
            "no_improve": no_improve,
        }
        lesson = (
            "accepted; subsequent rounds should preserve this skill and search "
            "for additional independent gains"
            if accepted
            else (
                "rejected by validation gate; do not repeat the same rewrite "
                "hypothesis or over-compression pattern"
            )
        )
        round_memory.append(
            {
                "round": round_idx,
                "accepted": accepted,
                "baseline_hard": record["current_hard_after_round"]
                if accepted else current_hard,
                "candidate_hard": candidate_hard,
                "candidate_soft": candidate_soft,
                "lesson": lesson,
            }
        )
        round_records.append(record)
        _write_json(base_out_root / "iterative_summary.json", {
            "status": "running",
            "source_out_root": str(source_root),
            "initial_skill": str(current_skill_path),
            "initial_skill_sha256": file_sha256(current_skill_path),
            "validation_split": validation_split,
            "rounds_requested": args.rounds,
            "patience": args.patience,
            "records": round_records,
            "best_hard": best_hard,
            "best_soft": best_soft,
            "token_summary": get_token_summary(),
            "provenance": build_provenance(
                _PROJECT_ROOT,
                extra_files=[os.path.relpath(args.config, _PROJECT_ROOT)]
                if os.path.isabs(args.config)
                else [args.config],
            ),
        })
        if no_improve >= args.patience:
            print(
                f"  [ALGO-03 iterative stop] no improvement for {no_improve} rounds"
            )
            break

    best_path = base_out_root / "best_skill.md"
    best_path.write_text(best_skill.rstrip() + "\n", encoding="utf-8")
    shutil.copyfile(current_skill_path, base_out_root / "initial_source_skill.md")
    _write_json(base_out_root / "iterative_summary.json", {
        "status": "complete",
        "source_out_root": str(source_root),
        "initial_skill": str(current_skill_path),
        "initial_skill_sha256": file_sha256(current_skill_path),
        "validation_split": validation_split,
        "rounds_requested": args.rounds,
        "patience": args.patience,
        "records": round_records,
        "best_hard": best_hard,
        "best_soft": best_soft,
        "best_skill": str(best_path),
        "token_summary": get_token_summary(),
        "provenance": build_provenance(
            _PROJECT_ROOT,
            extra_files=[os.path.relpath(args.config, _PROJECT_ROOT)]
            if os.path.isabs(args.config)
            else [args.config],
        ),
    })
    print(
        "  [ALGO-03 iterative done] "
        f"best_hard={best_hard} best_skill={best_path}"
    )


if __name__ == "__main__":
    main()
