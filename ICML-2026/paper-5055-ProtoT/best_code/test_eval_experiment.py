"""
Test-eval helper for EasyTuna experiments.

For each study in an experiment directory:
1) read study/results.json and find best trial number,
2) test-eval all seed runs inside that best trial (or the trial itself if no seed dirs),
3) write average test perplexity to a txt file in the study directory.

Usage example:
python test_eval_experiment.py --experiment_dir logs/scalability_models_ctx
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


TEST_PPL_RE = re.compile(r"test_ppl=([0-9]+(?:\.[0-9]+)?)")


def load_best_trial(study_dir: Path) -> tuple[int | None, float | None]:
    results_path = study_dir / "results.json"
    if not results_path.is_file():
        return None, None

    try:
        results = json.loads(results_path.read_text())
    except json.JSONDecodeError:
        return None, None

    best_trial = results.get("best_trial", {})
    trial_num = best_trial.get("number")
    trial_val = best_trial.get("value")
    if not isinstance(trial_num, int):
        return None, None
    if not isinstance(trial_val, (int, float)):
        trial_val = None
    return trial_num, trial_val


def extract_test_ppl(output: str) -> float | None:
    matches = TEST_PPL_RE.findall(output)
    if not matches:
        return None
    return float(matches[-1])


def run_test_eval(run_clm_path: Path, load_dir: Path) -> float | None:
    cmd = [
        sys.executable,
        str(run_clm_path),
        f"--LOAD_DIR={load_dir}",
        "--EPOCHS1=0",
        "--TEST_EVALUATE",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    joined_output = f"{result.stdout}\n{result.stderr}"
    test_ppl = extract_test_ppl(joined_output)

    if result.returncode != 0:
        print(f"[ERROR] test-eval failed for {load_dir} (return code {result.returncode})")
        return None
    if test_ppl is None:
        print(f"[WARN] test_ppl not found in output for {load_dir}")
        return None
    return test_ppl


def write_study_summary(
    study_dir: Path,
    out_name: str,
    best_trial_num: int,
    best_trial_dev_ppl: float | None,
    per_run: list[tuple[str, float]],
) -> None:
    out_path = study_dir / out_name
    avg_test_ppl = sum(v for _, v in per_run) / len(per_run)

    lines = [
        f"study: {study_dir.name}",
        f"best_trial: trial{best_trial_num:03d}",
        f"best_trial_dev_ppl: {best_trial_dev_ppl if best_trial_dev_ppl is not None else 'NA'}",
    ]
    for run_name, ppl in per_run:
        lines.append(f"{run_name}_test_ppl: {ppl:.6f}")
    lines.append(f"avg_test_ppl: {avg_test_ppl:.6f}")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"[OK] wrote {out_path} (avg_test_ppl={avg_test_ppl:.6f})")


def evaluate_study(study_dir: Path, run_clm_path: Path, out_name: str) -> None:
    best_trial_num, best_trial_dev_ppl = load_best_trial(study_dir)
    if best_trial_num is None:
        print(f"[SKIP] {study_dir}: missing/invalid results.json best_trial.number")
        return

    trial_dir = study_dir / "trial_runs" / f"trial{best_trial_num:03d}"
    if not trial_dir.is_dir():
        print(f"[SKIP] {study_dir}: missing trial directory {trial_dir}")
        return

    seed_dirs = sorted([p for p in trial_dir.glob("seed_*") if p.is_dir()])
    run_dirs = seed_dirs if seed_dirs else [trial_dir]

    per_run: list[tuple[str, float]] = []
    for run_dir in run_dirs:
        label = run_dir.name
        ppl = run_test_eval(run_clm_path, run_dir)
        if ppl is not None:
            per_run.append((label, ppl))

    if not per_run:
        print(f"[SKIP] {study_dir}: no successful test-eval runs")
        return

    write_study_summary(
        study_dir=study_dir,
        out_name=out_name,
        best_trial_num=best_trial_num,
        best_trial_dev_ppl=best_trial_dev_ppl,
        per_run=per_run,
    )


def list_studies(experiment_dir: Path) -> list[Path]:
    return sorted(
        [
            p
            for p in experiment_dir.iterdir()
            if p.is_dir() and (p / "results.json").is_file() and (p / "trial_runs").is_dir()
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-eval best trial of each study in an experiment directory."
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=True,
        help="Experiment directory containing study subdirectories (e.g. logs/scalability_models_ctx).",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="avg_test_ppl.txt",
        help="Output txt filename to write inside each study directory.",
    )
    args = parser.parse_args()

    run_clm_path = Path(__file__).resolve().parent / "run_clm.py"
    if not run_clm_path.is_file():
        raise FileNotFoundError(f"run_clm.py not found at {run_clm_path}")

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {experiment_dir}")

    studies = list_studies(experiment_dir)
    if not studies:
        print(f"[INFO] no studies found in {experiment_dir}")
        return

    print(f"[INFO] found {len(studies)} studies in {experiment_dir}")
    for study_dir in studies:
        evaluate_study(study_dir, run_clm_path=run_clm_path, out_name=args.out_name)


if __name__ == "__main__":
    main()
