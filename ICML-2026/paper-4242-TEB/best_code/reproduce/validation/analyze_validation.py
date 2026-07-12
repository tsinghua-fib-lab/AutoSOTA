#!/usr/bin/env python3
"""Analyse CFR online-controller validation results.

Reads the *_stats.json files written by run_validation.sh, plus the
benchmark JSONs, and reports per-workload:
  (i)   estimation accuracy of p̂_0 vs ground-truth p_0 = 1/E[O];
        estimation accuracy of μ̂_L vs ground-truth E[L];
  (ii)  realised TP vs the fluid-optimal TP at the controller's
        (k̂, N̂) — the closed-form throughput model
        TP_fluid = N·θ_0 / (α_d τ_R + β_d N θ_0/p_0 + α_p + β_p N θ_0 μ_L);
  (iii) OOM event rate vs the prescribed ε.

Outputs:
  validation_summary.csv         (one row per workload + scheduler)
  validation_table.tex           (LaTeX table for the paper)
  per-workload trace plots in plots/<scenario>/...
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


GROUND_TRUTH = {
    "decode_heavy":  {"input_len": 128,  "output_len": 1024},
    "balanced":      {"input_len": 512,  "output_len": 512},
    "prefill_heavy": {"input_len": 1024, "output_len": 128},
}


def fluid_tp(theta: float, p_0: float, n: int, mu_L: float,
              alpha_p: float, beta_p: float,
              alpha_d: float, beta_d: float) -> float:
    """Closed-form fluid throughput TP(θ, N) (paper Eq. 9)."""
    if theta <= 0 or theta >= 1 or p_0 <= 0 or n <= 0:
        return 0.0
    tau = math.log(1 - theta) / math.log(1 - p_0)
    decode_time = alpha_d * tau + beta_d * n * theta / p_0
    prefill_time = alpha_p + beta_p * n * theta * mu_L
    served_tokens = n * theta * (mu_L + 1.0 / p_0)
    return served_tokens / max(decode_time + prefill_time, 1e-9)


def load_run(workload_dir: Path, sched: str) -> dict | None:
    bench = workload_dir / f"bench_{sched}.json"
    stats = workload_dir / f"{sched}_stats.json"
    if not bench.exists():
        return None
    with open(bench) as f:
        b = json.load(f)
    s = {}
    if stats.exists():
        with open(stats) as f:
            s = json.load(f)
    return {"bench": b, "stats": s, "bench_path": bench, "stats_path": stats}


def discover_schedulers(workload_dir: Path) -> list[str]:
    """Return scheduler names in a stable order: v1, eb, fixed-k sweep."""
    names = {p.stem.removeprefix("bench_") for p in workload_dir.glob("bench_*.json")}
    ordered: list[str] = []
    for primary in ("v1", "eb"):
        if primary in names:
            ordered.append(primary)
    fixed = sorted(
        (n for n in names if n.startswith("eb_fixed_k_")),
        key=lambda n: int(n.removeprefix("eb_fixed_k_")),
    )
    ordered.extend(fixed)
    return ordered


def summarise(workload_dir: Path, scen: str, cfg: dict) -> list[dict]:
    rows = []
    for sched in discover_schedulers(workload_dir):
        run = load_run(workload_dir, sched)
        if run is None:
            continue
        b = run["bench"]
        s = run["stats"]
        hist = s.get("update_history") or s.get("cfr_update_history") or []
        last = hist[-1] if hist else {}
        pd_cfg = s.get("pd_config", {})
        is_fixed_k = sched.startswith("eb_fixed_k_")

        # Ground truth
        gt_o = GROUND_TRUTH[scen]["output_len"]
        gt_l = GROUND_TRUTH[scen]["input_len"]
        gt_p_0 = 1.0 / gt_o

        # Online estimates (NaN for fixed-k — controller is disabled)
        out_lens = b.get("output_lens") or []
        in_lens = b.get("input_lens") or []
        realised_o = float(np.mean(out_lens)) if out_lens else 0.0
        realised_l = float(np.mean(in_lens)) if in_lens else 0.0
        if is_fixed_k:
            p_hat = float("nan")
            mu_l_hat = float("nan")
            theta_0 = float("nan")
            k_hat = int(sched.removeprefix("eb_fixed_k_"))
        else:
            p_hat = last.get("p_0_estimate", 0.0)
            mu_l_hat = last.get("mu_L_estimate", 0.0)
            theta_0 = last.get("theta_0", 0.0)
            k_hat = int(last.get("k_hat_int", pd_cfg.get("final_k_star", 0)))

        # Throughput attainment (only meaningful for the adaptive controller)
        n_hat = int(last.get("N_hat", pd_cfg.get("max_num_seqs", 0)))
        cal = cfg.get("calibration_params", {})
        if is_fixed_k or not isinstance(theta_0, (int, float)) or math.isnan(theta_0):
            tp_fluid = float("nan")
            attainment = float("nan")
        else:
            tp_fluid = fluid_tp(
                theta_0, gt_p_0, n_hat, gt_l,
                float(cal.get("alpha_p", 0.0)),
                float(cal.get("beta_p", 0.0)),
                float(cal.get("alpha_d", 0.0)),
                float(cal.get("beta_d", 0.0)),
            )
            attainment = tp_real_attain(b, tp_fluid)
        tp_real = b.get("total_token_throughput", 0.0)
        mean_tpot_ms = b.get("mean_tpot_ms", float("nan"))
        # vLLM bench reports percentiles via separate top-level fields when
        # --percentile-metrics tpot is set (default); newer versions also
        # expose percentiles_tpot_ms as a list[(percentile, ms)].
        p99_tpot_ms = b.get("p99_tpot_ms")
        if p99_tpot_ms is None:
            for pct, val in b.get("percentiles_tpot_ms") or []:
                if abs(pct - 99) < 1e-6:
                    p99_tpot_ms = val
                    break
        if p99_tpot_ms is None:
            p99_tpot_ms = float("nan")

        # OOM rate
        completed = b.get("completed", 0)
        oom = pd_cfg.get("total_oom_events", 0)
        oom_rate = oom / max(completed, 1)
        eps = float(cfg.get("oom_tolerance", 0.01))

        rows.append({
            "scenario": scen,
            "scheduler": sched,
            "ground_truth_E_L": gt_l,
            "ground_truth_E_O": gt_o,
            "ground_truth_p_0": gt_p_0,
            "realised_E_L": realised_l,
            "realised_E_O": realised_o,
            "p_hat_final": p_hat,
            "mu_L_hat_final": mu_l_hat,
            "p_hat_relerr_pct": (abs(p_hat - gt_p_0) / gt_p_0 * 100)
                if (gt_p_0 > 0 and not math.isnan(p_hat)) else float("nan"),
            "mu_L_hat_relerr_pct": (abs(mu_l_hat - gt_l) / gt_l * 100)
                if (gt_l > 0 and not math.isnan(mu_l_hat)) else float("nan"),
            "theta_0_final": theta_0,
            "k_hat_final": k_hat,
            "N_hat_final": n_hat,
            "tp_real": tp_real,
            "mean_tpot_ms": mean_tpot_ms,
            "p99_tpot_ms": p99_tpot_ms,
            "tp_fluid": tp_fluid,
            "attainment_pct": attainment * 100 if not math.isnan(attainment) else float("nan"),
            "oom_events": oom,
            "completed": completed,
            "oom_rate": oom_rate,
            "oom_rate_pct": oom_rate * 100,
            "eps_target": eps,
            "n_updates": len(hist),
        })
    return rows


def tp_real_attain(b: dict, tp_fluid: float) -> float:
    tp_real = b.get("total_token_throughput", 0.0)
    return tp_real / tp_fluid if tp_fluid > 0 else 0.0


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(empty)\n")
        return
    fields = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            f.write(",".join(repr(r[k]) if isinstance(r[k], str) else f"{r[k]}"
                              for k in fields) + "\n")


def write_latex(rows: list[dict], path: Path, eps: float) -> None:
    eb_rows = [r for r in rows if r["scheduler"] == "eb"]
    eb_rows.sort(key=lambda r: ("decode_heavy", "balanced", "prefill_heavy").index(r["scenario"]))
    with open(path, "w") as f:
        f.write("% Auto-generated by analyze_validation.py\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Workload & $\\hat p_0/p_0$ (\\%) & $\\hat\\mu_L/\\mu_L$ (\\%) "
                "& $\\theta_0$ & $\\hat k$ & $\\hat N$ "
                "& TP attain.\\ (\\%) & OOM rate (\\%) \\\\\n")
        f.write("\\midrule\n")
        for r in eb_rows:
            f.write(f"{r['scenario']} & "
                    f"{(1 - r['p_hat_relerr_pct']/100)*100:.1f} & "
                    f"{(1 - r['mu_L_hat_relerr_pct']/100)*100:.1f} & "
                    f"{r['theta_0_final']:.4f} & "
                    f"{r['k_hat_final']} & "
                    f"{r['N_hat_final']} & "
                    f"{r['attainment_pct']:.1f} & "
                    f"{r['oom_rate_pct']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(f"% Prescribed OOM tolerance ε = {eps}\n")


def maybe_plot_traces(workload_dir: Path, scen: str, cfg: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    run = load_run(workload_dir, "eb")
    if run is None:
        return
    hist = (
        run["stats"].get("update_history")
        or run["stats"].get("cfr_update_history")
        or []
    )
    if not hist:
        return
    times = [h["timestamp"] for h in hist]
    p_hat = [h["p_0_estimate"] for h in hist]
    mu_l_hat = [h["mu_L_estimate"] for h in hist]
    k_hat = [h["k_hat_int"] for h in hist]
    n_hat = [h["N_hat"] for h in hist]
    gt = GROUND_TRUTH[scen]
    plot_dir = workload_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(times, p_hat, label=r"$\hat p_0$")
    axes[0, 0].axhline(1 / gt["output_len"], color="r", linestyle="--",
                        label=r"truth $1/\mu_O$")
    axes[0, 0].set_title("p_0 estimate"); axes[0, 0].legend()
    axes[0, 1].plot(times, mu_l_hat, label=r"$\hat\mu_L$")
    axes[0, 1].axhline(gt["input_len"], color="r", linestyle="--",
                        label=r"truth $\mu_L$")
    axes[0, 1].set_title("mu_L estimate"); axes[0, 1].legend()
    axes[1, 0].plot(times, k_hat, color="C2")
    axes[1, 0].set_title(r"$\hat k$ over time")
    axes[1, 1].plot(times, n_hat, color="C3")
    axes[1, 1].set_title(r"$\hat N$ over time")
    for ax in axes.flat:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{scen}: online estimator trace")
    fig.tight_layout()
    out = plot_dir / "estimator_trace.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("experiment_dir", type=Path)
    args = p.parse_args()
    root = args.experiment_dir
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    cfg = json.loads((root / "experiment_config.json").read_text())
    eps = float(cfg.get("oom_tolerance", 0.01))

    rows: list[dict] = []
    for scen in ("decode_heavy", "balanced", "prefill_heavy"):
        sub = sorted(root.glob(f"{scen}_in*_out*"))
        if not sub:
            continue
        wd = sub[0]
        rows.extend(summarise(wd, scen, cfg))
        maybe_plot_traces(wd, scen, cfg)

    write_csv(rows, root / "validation_summary.csv")
    write_latex(rows, root / "validation_table.tex", eps)
    print(f"\nWrote {root / 'validation_summary.csv'}")
    print(f"Wrote {root / 'validation_table.tex'}")
    print()
    if rows:
        print(f"  {'workload':14s} {'sched':8s} "
              f"{'p̂_0':>10s} {'gt':>10s} "
              f"{'μ̂_L':>7s} {'gt':>5s} "
              f"{'TP':>9s} {'TP_flu':>9s} {'attain':>7s} "
              f"{'OOM%':>7s} {'ε':>6s}")
        for r in rows:
            print(f"  {r['scenario']:14s} {r['scheduler']:8s} "
                  f"{r['p_hat_final']:>10.5f} "
                  f"{r['ground_truth_p_0']:>10.5f} "
                  f"{r['mu_L_hat_final']:>7.0f} "
                  f"{r['ground_truth_E_L']:>5d} "
                  f"{r['tp_real']:>9.0f} "
                  f"{r['tp_fluid']:>9.0f} "
                  f"{r['attainment_pct']:>6.1f}% "
                  f"{r['oom_rate_pct']:>6.2f}% "
                  f"{r['eps_target']*100:>5.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
