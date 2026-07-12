# -*- coding: utf-8 -*-
"""
scBridge-Flow resource benchmarking script.
Measures wall-clock time, peak GPU memory, and peak CPU memory
across multiple dataset sizes on a fixed dataset.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _run_cmd(cmd: List[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_rows(size: int, run_id: int, records: List[Dict], stage: str, metrics: Dict, meta: Dict):
    records.append({
        "dataset": meta["dataset_name"],
        "size": size,
        "run_id": run_id,
        "stage": stage,
        "wall_clock_s": metrics.get("wall_clock_s"),
        "gpu_peak_mem_mb": metrics.get("gpu_peak_mem_mb"),
        "cpu_peak_mem_mb": metrics.get("cpu_peak_mem_mb"),
        "device": metrics.get("device"),
        "batch_size": meta["batch_size"],
        "stage1_epochs": meta["stage1_epochs"],
        "stage2_epochs": meta["stage2_epochs"],
        "ode_method": meta["ode_method"],
        "status": "ok",
    })


def _plot_metric(agg_df: pd.DataFrame, metric: str, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)
    stages = ["stage1", "stage2", "inference"]

    for ax, stage in zip(axes, stages):
        sdf = agg_df[agg_df["stage"] == stage].sort_values("size")
        if sdf.empty:
            ax.set_title(stage)
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x = sdf["size"].to_numpy()
        y = sdf[f"{metric}_mean"].to_numpy()
        yerr = sdf[f"{metric}_std"].fillna(0).to_numpy()

        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xticks([1000, 10000, 50000, 100000])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(stage)
        ax.set_xlabel("dataset size")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"scBridge-Flow Benchmark: {metric}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    root = Path(__file__).resolve().parent
    stage1_script = root / "train_stage1_vae.py"
    stage2_script = root / "train_stage2_cfm.py"
    infer_script = root / "infer.py"

    bench_root = Path(args.output_dir)
    bench_root.mkdir(parents=True, exist_ok=True)

    dataset_name = Path(args.data_path).stem
    records: List[Dict] = []

    for size in args.sizes:
        for run_id in range(1, args.repeats + 1):
            run_seed = args.base_seed + run_id
            run_root = bench_root / dataset_name / f"size_{size}" / f"run_{run_id}"
            stage1_out = run_root / "stage1"
            stage2_out = run_root / "stage2"
            infer_out = run_root / "inference"

            stage1_metric = stage1_out / "resource_metrics_stage1.json"
            stage2_train_metric = stage2_out / "resource_metrics_stage2_train.json"
            infer_metric = infer_out / "resource_metrics_infer.json"

            try:
                stage1_cmd = [
                    sys.executable,
                    str(stage1_script),
                    "--data_path", args.data_path,
                    "--output_dir", str(stage1_out),
                    "--device", args.device,
                    "--epochs", str(args.stage1_epochs),
                    "--n_top_genes", str(args.n_top_genes),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.stage1_lr),
                    "--dz", str(args.dz),
                    "--beta_kl", str(args.beta_kl),
                    "--dist_type", args.dist_type,
                    "--seed", str(run_seed),
                    "--subset_size", str(size),
                    "--resource_metrics_path", str(stage1_metric),
                ]
                if args.use_raw_for_nb and args.dist_type != "Gaussian":
                    stage1_cmd.append("--use_raw_for_nb")
                _run_cmd(stage1_cmd)

                stage2_cmd = [
                    sys.executable,
                    str(stage2_script),
                    "--data_path", args.data_path,
                    "--vae_path", str(stage1_out / "vae_best.pt"),
                    "--output_dir", str(stage2_out),
                    "--device", args.device,
                    "--epochs", str(args.stage2_epochs),
                    "--n_top_genes", str(args.n_top_genes),
                    "--batch_size", str(args.batch_size),
                    "--lr", str(args.stage2_lr),
                    "--dc", str(args.dc),
                    "--p_uncond", str(args.p_uncond),
                    "--lambda_cons", str(args.lambda_cons),
                    "--n_steps", str(args.n_steps),
                    "--cfg_scale", str(args.cfg_scale),
                    "--ode_method", args.ode_method,
                    "--ode_rtol", str(args.ode_rtol),
                    "--ode_atol", str(args.ode_atol),
                    "--seed", str(run_seed),
                    "--subset_size", str(size),
                    "--skip_final_eval",
                    "--resource_metrics_train_path", str(stage2_train_metric),
                ]
                _run_cmd(stage2_cmd)

                infer_cmd = [
                    sys.executable,
                    str(infer_script),
                    "--vae_path", str(stage1_out / "vae_best.pt"),
                    "--flow_path", str(stage2_out / "flow_best.pt"),
                    "--data_info_path", str(stage1_out / "data_info.pt"),
                    "--data_path", args.data_path,
                    "--batch_size", str(args.batch_size),
                    "--device", args.device,
                    "--n_steps", str(args.n_steps),
                    "--cfg_scale", str(args.cfg_scale),
                    "--ode_method", args.ode_method,
                    "--ode_rtol", str(args.ode_rtol),
                    "--ode_atol", str(args.ode_atol),
                    "--subset_size", str(size),
                    "--subset_seed", str(run_seed),
                    "--output_path", str(infer_out / "predictions.csv"),
                    "--resource_metrics_path", str(infer_metric),
                ]
                _run_cmd(infer_cmd)

                common_meta = {
                    "dataset_name": dataset_name,
                    "batch_size": args.batch_size,
                    "stage1_epochs": args.stage1_epochs,
                    "stage2_epochs": args.stage2_epochs,
                    "ode_method": args.ode_method,
                }
                _build_rows(size, run_id, records, "stage1", _read_json(stage1_metric), common_meta)
                _build_rows(size, run_id, records, "stage2", _read_json(stage2_train_metric), common_meta)
                _build_rows(size, run_id, records, "inference", _read_json(infer_metric), common_meta)

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] size={size}, run={run_id}, cmd failed: {e}")
                for stage in ["stage1", "stage2", "inference"]:
                    records.append({
                        "dataset": dataset_name,
                        "size": size,
                        "run_id": run_id,
                        "stage": stage,
                        "wall_clock_s": np.nan,
                        "gpu_peak_mem_mb": np.nan,
                        "cpu_peak_mem_mb": np.nan,
                        "device": args.device,
                        "batch_size": args.batch_size,
                        "stage1_epochs": args.stage1_epochs,
                        "stage2_epochs": args.stage2_epochs,
                        "ode_method": args.ode_method,
                        "status": "failed",
                    })

    raw_df = pd.DataFrame(records)
    raw_csv = bench_root / "summary_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"Saved raw summary to {raw_csv}")

    ok_df = raw_df[raw_df["status"] == "ok"].copy()
    agg_df = ok_df.groupby(["dataset", "size", "stage"], as_index=False).agg(
        wall_clock_s_mean=("wall_clock_s", "mean"),
        wall_clock_s_std=("wall_clock_s", "std"),
        gpu_peak_mem_mb_mean=("gpu_peak_mem_mb", "mean"),
        gpu_peak_mem_mb_std=("gpu_peak_mem_mb", "std"),
        cpu_peak_mem_mb_mean=("cpu_peak_mem_mb", "mean"),
        cpu_peak_mem_mb_std=("cpu_peak_mem_mb", "std"),
        n_runs=("run_id", "nunique"),
    )

    agg_csv = bench_root / "summary_agg.csv"
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated summary to {agg_csv}")

    fig_dir = bench_root / "figures"
    _plot_metric(agg_df, "wall_clock_s", fig_dir / "wall_clock_vs_size.png")
    _plot_metric(agg_df, "gpu_peak_mem_mb", fig_dir / "gpu_peak_mem_vs_size.png")
    _plot_metric(agg_df, "cpu_peak_mem_mb", fig_dir / "cpu_peak_mem_vs_size.png")
    print(f"Saved benchmark figures to {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scBridge-Flow resources across dataset sizes")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset (GSE164378_3P.h5ad)")
    parser.add_argument("--output_dir", type=str, default="./outputs/benchmark", help="Benchmark output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000, 100000], help="Dataset sizes")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per size")
    parser.add_argument("--base_seed", type=int, default=1234, help="Base seed")

    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--stage1_epochs", type=int, default=600)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--dz", type=int, default=32)
    parser.add_argument("--beta_kl", type=float, default=0.8)
    parser.add_argument("--dist_type", type=str, default="Gaussian", choices=["Gaussian", "NB", "ZINB"])
    parser.add_argument("--use_raw_for_nb", action="store_true")

    parser.add_argument("--stage2_epochs", type=int, default=200)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--dc", type=int, default=512)
    parser.add_argument("--p_uncond", type=float, default=0.2)
    parser.add_argument("--lambda_cons", type=float, default=0.1)

    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--ode_method", type=str, default="dopri5", choices=["dopri5", "dopri8", "rk4", "euler", "midpoint", "heun3", "adaptive_heun"])
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)

    args = parser.parse_args()
    main(args)
