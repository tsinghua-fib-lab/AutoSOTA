

#!/usr/bin/env python3
"""Optuna-driven hyperparameter search for main.py.

This replaces the previous random search. It launches `main.py` as a
subprocess for each Optuna trial, parses the resulting
`results/<name>_<trial>.txt`, and minimizes the final **Model**
discrepancy from the last row of the discrepancy table.

NOTE on parallelism:
- To run trials in parallel on HTCondor, submit multiple jobs that point
  to the same `--study_name` and `--storage` (e.g., a shared SQLite file
  on a network filesystem). Optuna will coordinate trials across workers.

Fixed (non-searched) parameters (must be passed via CLI if you need to override defaults):
    dim, seq_total_length, train_seq, epochs, pretrain_epochs, pre_eps, ft_eps, disc_loss

Searched parameters:
    hidden, layers, K, lr, fine_tune_lr, warmup_epochs,
    min_finetune_lr, final_lr_ratio

You may also pass through any other flags supported by main.py (e.g.,
`--gamma ...`) using the native `--gamma` flag (preferred, optional), or for other flags, via `--extra_args`.
"""

import argparse
import os
import subprocess
import sys
import re
from typing import Optional

# Results directory anchored to repository root (one level above this file)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


# Optuna is required
try:
    import optuna
except Exception as e:
    raise RuntimeError(
        "Optuna is required. Install with `pip install optuna` on the cluster environment."
    ) from e


# ---------------------------- helpers ---------------------------------------

def _parse_objective_value(txt_path: str) -> Optional[float]:
    """Objective = min over:
       - rolling means of window size 10 taken at 10 evenly spaced endpoints
       - the final (last) Model discrepancy
    from the table:
       Length  Sobol  Halton  Model  ScrambledSobolMean
    """
    if not os.path.exists(txt_path):
        return None
    model_vals = []
    in_table = False
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not in_table:
                if line.startswith("Length") and "Model" in line:
                    in_table = True
                continue
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 5 and parts[0].isdigit():
                try:
                    model_vals.append(float(parts[3]))
                except ValueError:
                    pass
    if not model_vals:
        return None
    import numpy as _np
    N = len(model_vals)
    anchors = _np.linspace(10, N, num=min(10, max(1, N)), dtype=int)
    means = []
    for a in anchors:
        a = max(1, min(N, int(a)))
        start = max(0, a - 10)
        window = model_vals[start:a]
        if window:
            means.append(float(_np.mean(window)))
    means.append(float(model_vals[-1]))  # include the last point explicitly
    return float(min(means)) if means else float(model_vals[-1])


def _fmt(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


# ---------------------------- objective -------------------------------------

def make_objective(args):
    main_py = os.path.join(os.path.dirname(__file__), "main.py")

    def objective(trial: "optuna.trial.Trial") -> float:
        # Sample ONLY the allowed parameters
        hidden           = trial.suggest_categorical("hidden", [256, 512, 768, 1024])
        layers           = trial.suggest_int("layers", 4, 8)
        K                = trial.suggest_categorical("K", [8, 16, 20, 32, 64])
        lr               = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        fine_tune_lr     = trial.suggest_float("fine_tune_lr", 1e-4, 1e-2, log=True)
        warmup_epochs    = trial.suggest_int("warmup_epochs", 10, 100)
        min_finetune_lr  = trial.suggest_float("min_finetune_lr", 1e-7, 1e-5, log=True)
        final_lr_ratio   = trial.suggest_float("final_lr_ratio", 0.03, 0.3)
        group = args.group if args.group else f"DIM{args.dim}_L{args.seq_total_length}_seq{args.train_seq}_disc{args.disc_loss}"
        run_name = f"{args.name}_{trial.number:04d}"
        # Remember the group for this trial
        trial.set_user_attr("group", group)

        cmd = [
            sys.executable, main_py,
            "--name", f"{group}/{run_name}",
            # fixed (non-searched) params
            "--dim", _fmt(args.dim),
            "--seq_total_length", _fmt(args.seq_total_length),
            "--epochs", _fmt(args.epochs),
            "--train_seq", args.train_seq,
            "--pretrain_epochs", _fmt(args.pretrain_epochs),
            "--pre_eps", _fmt(args.pre_eps),
            "--ft_eps", _fmt(args.ft_eps),
            "--disc_loss", args.disc_loss,
            # searched params
            "--hidden", _fmt(hidden),
            "--layers", _fmt(layers),
            "--K", _fmt(K),
            "--lr", _fmt(lr),
            "--fine_tune_lr", _fmt(fine_tune_lr),
            "--warmup_epochs", _fmt(warmup_epochs),
            "--min_finetune_lr", _fmt(min_finetune_lr),
            "--final_lr_ratio", _fmt(final_lr_ratio),
        ]

        # Forward gamma vector if provided (preferred over --extra_args quoting)
        if getattr(args, "gamma", None) is not None:
            cmd += ["--gamma"] + [str(v) for v in args.gamma]

        # passthrough extra args for main.py (e.g., --verb True)
        if args.extra_args:
            cmd.extend(args.extra_args.split())

        # Allow pruning by killing long/poor trials if desired (future work)
        print("[Optuna] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            # Return a large penalty on failure so the trial is considered bad
            print(f"[WARN] Trial {trial.number} failed with code {e.returncode}")
            return float("inf")

        txt_path = os.path.join(RESULTS_ROOT, group, run_name, f"{run_name}.txt")
        value = _parse_objective_value(txt_path)
        # Persist where this trial actually wrote its outputs
        trial.set_user_attr("run_name", run_name)
        trial.set_user_attr("run_dir", os.path.join(RESULTS_ROOT, group, run_name))
        trial.set_user_attr("result_txt", txt_path)
        if value is None:
            # If parsing failed, penalize
            print(f"[WARN] Could not parse objective from {txt_path}")
            return float("inf")
        print(f"[Optuna] Trial {trial.number} objective = {value}")
        return float(value)

    return objective


# ---------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Optuna search wrapper for main.py")
    ap.add_argument("--name", required=True, help="Base name; runs saved to results/<name>_<trial>.txt/png")
    ap.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run in this process")
    ap.add_argument("--study_name", default="mpmc_optuna", help="Optuna study name (shared across workers for parallelism)")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL (e.g., sqlite:////home/user/optuna_mpmc.db). If None, use in-memory (no cross-process parallelism).")
    # Fixed parameters (NOT searched) — defaults mirror main.py but can be overridden
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--seq_total_length", type=int, default=1000)
    ap.add_argument("--train_seq", choices=["sobol", "halton"], default="sobol")
    ap.add_argument("--pretrain_epochs", type=int, default=30000)
    ap.add_argument("--pre_eps", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--ft_eps", type=float, default=0.0)
    ap.add_argument("--disc_loss", choices=["star","sym","per","ext","asd","ctr",
                                             "star_w","sym_w","per_w","ext_w","ctr_w"],
                    default="star")
    ap.add_argument("--gamma", nargs="+", type=float, default=None,
                    help="Optional weight vector forwarded to main.py when using *_w losses; length should equal --dim.")
    ap.add_argument("--extra_args", default="",
                    help="Additional args to pass through to main.py (avoid quotes issues); prefer native flags like --gamma.")
    ap.add_argument("--group", default=None,
                help="Hierarchical group for results under results/<group>/ "
                     "(defaults to DIM{dim}_seq{train_seq}_disc{disc_loss}).")
    args = ap.parse_args()

    # Create/Load study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction="minimize",
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize")

    # Optimize
    objective = make_objective(args)
    study.optimize(objective, n_trials=args.n_trials)

    print("=== Optuna finished ===")
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    group = args.group if args.group else f"DIM{args.dim}_L{args.seq_total_length}_seq{args.train_seq}_disc{args.disc_loss}"
    # Persist best setup and the corresponding run’s txt
    best_dir = os.path.join(RESULTS_ROOT, "best", args.study_name, group)
    os.makedirs(best_dir, exist_ok=True)

    # Compute best trial/run info before writing best_params.txt, so we can record identifiers and use them for copying artifacts.
    # Copy the best run’s artifacts next to best_params.txt
    # NOTE: best trial may have run under a *different worker name* than this process.
    # We stored the true run_name/run_dir in trial.user_attrs.
    best_trial = study.best_trial
    best_run_name = best_trial.user_attrs.get("run_name") or f"{args.name}_{best_trial.number:04d}"
    src_dir = best_trial.user_attrs.get("run_dir") or os.path.join(RESULTS_ROOT, group, best_run_name)

    # Save best params
    best_txt = os.path.join(best_dir, "best_params.txt")
    with open(best_txt, "w") as f:
        f.write(f"study_name: {args.study_name}\n")
        f.write(f"base_name: {args.name}\n")
        f.write(f"best_value: {study.best_value}\n")
        f.write(f"best_trial_number: {study.best_trial.number}\n")
        f.write(f"best_run_name: {best_run_name}\n")
        f.write("best_params:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        # also include fixed args that matter for reproducibility
        f.write("fixed_args:\n")
        f.write(f"  epochs: {args.epochs}\n")
        f.write(f"  dim: {args.dim}\n")
        f.write(f"  seq_total_length: {args.seq_total_length}\n")
        f.write(f"  train_seq: {args.train_seq}\n")
        f.write(f"  pretrain_epochs: {args.pretrain_epochs}\n")
        f.write(f"  pre_eps: {args.pre_eps}\n")
        f.write(f"  ft_eps: {args.ft_eps}\n")
        f.write(f"  disc_loss: {args.disc_loss}\n")
        if getattr(args, "gamma", None) is not None:
            f.write("  gamma: [" + ", ".join(str(v) for v in args.gamma) + "]\n")
        if args.extra_args:
            f.write(f"  extra_args: {args.extra_args}\n")
        f.write("\nResults are organized hierarchically by a group path. By default, group = DIM{dim}_L{seq_total_length}_seq{train_seq}_disc{disc_loss}, but you can override with --group.\n")

    files_to_copy = [
        (os.path.join(src_dir, f"{best_run_name}.txt"),         os.path.join(best_dir, f"{best_run_name}.txt")),
        (os.path.join(src_dir, f"{best_run_name}.png"),         os.path.join(best_dir, f"{best_run_name}.png")),
        (os.path.join(src_dir, f"{best_run_name}_pretrain.png"), os.path.join(best_dir, f"{best_run_name}_pretrain.png")),
    ]
    copied_any = False
    for src, dst in files_to_copy:
        try:
            with open(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())
            copied_any = True
        except FileNotFoundError:
            print(f"[WARN] Could not find file to copy: {src}")
    if copied_any:
        print(f"Best params saved to {best_txt} and best run artifacts copied into {best_dir}")
    else:
        print(f"[WARN] No best run artifacts found under {src_dir}")
if __name__ == "__main__":
    main()

