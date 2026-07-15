#!/usr/bin/env python3
"""
Make a compact summary for ProbeSwitch:

1) Threshold transfer (zero-tuning) across tasks/budgets:
   - input: evidence/probe_threshold_transfer/transfer_summary.csv
2) VOI / probe-overhead-vs-gain curve (logreg sweep):
   - input: evidence/logreg_voi_overhead_gain_curve/curve_summary.csv

Outputs (default):
  evidence/probeswitch_transfer_overhead_summary/
    - transfer_overhead_main.png / .pdf  (main multi-panel figure)
    - transfer_summary_compact.csv
    - overhead_curve_compact.csv
    - summary.md  (1 file containing 2 tables)
    - README.md
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _project import BASE_DIR, repo_relpath
from plot_style import apply_style, COLORS, add_grid, save_figure, get_figsize


def _format_float(x: float | int | None, *, digits: int = 3) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return "-"
    if not np.isfinite(v):
        return "-"
    return f"{v:.{digits}f}"


def _format_p(x: float | int | None) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except Exception:
        return "-"
    if not np.isfinite(v):
        return "-"
    if v < 1e-4:
        return f"{v:.1e}"
    return f"{v:.4f}"


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_readme(
    out_dir: Path,
    *,
    fig_png: Path,
    table_md: Path,
    transfer_method: str,
    safe_transfer_method: str,
) -> None:
    extra = ""
    if safe_transfer_method:
        extra = f"\n   - Also compares a conservative `safe` rule (`{safe_transfer_method}`) to reduce boundary failures."
    readme = f"""# ProbeSwitch: Transfer + Overhead Summary

This evidence package aggregates two complementary results into **one main plot** and **one table**:

1) **Zero-tuning threshold transfer**: a COCO-learned misranking threshold is frozen and applied to other budgets/tasks.
   - Highlighted rule: `{transfer_method}`.{extra}
2) **VOI / overhead-vs-gain**: under fixed budgets, probing can be pure overhead in near-deterministic regimes; warmstart fixes it.

## Artifacts

- Main figure: `{repo_relpath(str(fig_png))}`
- Summary table (two blocks): `{repo_relpath(str(table_md))}`

## Inputs

- Threshold transfer summary:
  - `evidence/probe_threshold_transfer/transfer_summary.csv`
- Overhead-vs-gain curve (logreg sweep):
  - `evidence/logreg_voi_overhead_gain_curve/curve_summary.csv`

## Reproduce

```bash
python3 tools/make_probeswitch_transfer_overhead_summary.py \\
  --out-dir evidence/probeswitch_transfer_overhead_summary
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transfer-csv",
        default="evidence/probe_threshold_transfer/transfer_summary.csv",
        help="Input transfer summary CSV.",
    )
    parser.add_argument(
        "--overhead-csv",
        default="evidence/logreg_voi_overhead_gain_curve/curve_summary.csv",
        help="Input overhead-vs-gain curve CSV.",
    )
    parser.add_argument(
        "--out-dir",
        default="evidence/probeswitch_transfer_overhead_summary",
        help="Output directory for figure + table.",
    )
    parser.add_argument(
        "--transfer-method",
        default="bbob_B500",
        help="Which transfer rule to highlight (default: COCO-learned threshold from bbob_B500).",
    )
    parser.add_argument(
        "--safe-transfer-method",
        default="fixed0p22",
        help=(
            "Optional conservative rule to compare against the highlighted transfer method "
            "(default: fixed t=0.22). Set to empty to disable."
        ),
    )
    args = parser.parse_args()

    os.chdir(BASE_DIR)

    transfer_csv = Path(str(args.transfer_csv))
    overhead_csv = Path(str(args.overhead_csv))
    out_dir = Path(str(args.out_dir))
    _ensure_out_dir(out_dir)

    if not transfer_csv.exists():
        raise SystemExit(f"Missing transfer CSV: {transfer_csv}")
    if not overhead_csv.exists():
        raise SystemExit(f"Missing overhead CSV: {overhead_csv}")

    transfer_df = pd.read_csv(transfer_csv)
    overhead_df = pd.read_csv(overhead_csv)

    def _canonicalize_overhead(df: pd.DataFrame, *, overhead_csv_path: Path) -> pd.DataFrame:
        """
        Accept either:
          - wide format: has columns like `median_post_true_CMA`, `p_CMA_vs_Switch`, ...
          - long format: columns {budget_mult, batch_size, algorithm, n, median_post_true}
            produced by `tools/reproduce_all.py` (full suite).

        Returns a wide, stable table with deterministic column names expected by this script.
        """

        if {"median_post_true_CMA", "median_post_true_Switch", "median_post_true_Warmstart"}.issubset(df.columns):
            return df.copy()

        required_long = {"budget_mult", "batch_size", "algorithm", "median_post_true"}
        if not required_long.issubset(df.columns):
            raise SystemExit(
                "Unsupported overhead CSV format. Expected either wide columns like "
                "`median_post_true_CMA` or long columns "
                f"{sorted(required_long)}. Got: {sorted(df.columns)}"
            )

        pivot = (
            df.pivot_table(
                index=["budget_mult", "batch_size"],
                columns="algorithm",
                values="median_post_true",
                aggfunc="first",
            )
            .reset_index()
        )

        algo_cols = [c for c in pivot.columns if c not in {"budget_mult", "batch_size"}]

        def _pick(pred) -> str:
            for c in algo_cols:
                if pred(str(c)):
                    return str(c)
            return ""

        cma_col = _pick(lambda s: s.strip().lower().startswith("cma"))
        warm_col = _pick(lambda s: "warmstart" in s.lower())
        switch_col = _pick(lambda s: ("probeswitch" in s.lower()) and ("warmstart" not in s.lower()))

        if not cma_col or not switch_col or not warm_col:
            raise SystemExit(
                "Cannot infer CMA/Switch/Warmstart columns from overhead CSV. "
                f"Found algorithm columns: {algo_cols}"
            )

        out = pd.DataFrame(
            {
                "budget_mult": pivot["budget_mult"].astype(int),
                "batch_size": pivot["batch_size"].astype(int),
                "median_post_true_CMA": pivot[cma_col].astype(float),
                "median_post_true_Switch": pivot[switch_col].astype(float),
                "median_post_true_Warmstart": pivot[warm_col].astype(float),
            }
        )

        raw_dir = overhead_csv_path.parent / "raw"
        p_cma_switch: list[float] = []
        p_cma_warm: list[float] = []
        p_switch_warm: list[float] = []

        for _, r in out.iterrows():
            b = int(r["budget_mult"])
            bs = int(r["batch_size"])
            sign_path = raw_dir / f"B{b}_bs{bs}_pairwise_sign_test_post_true.csv"
            if not sign_path.exists():
                p_cma_switch.append(float("nan"))
                p_cma_warm.append(float("nan"))
                p_switch_warm.append(float("nan"))
                continue
            sign_df = pd.read_csv(sign_path)

            def _p(a: str, b: str) -> float:
                m = ((sign_df["algo_a"] == a) & (sign_df["algo_b"] == b)) | ((sign_df["algo_a"] == b) & (sign_df["algo_b"] == a))
                if not bool(m.any()):
                    return float("nan")
                return float(sign_df.loc[m, "p_two_sided"].iloc[0])

            p_cma_switch.append(_p(cma_col, switch_col))
            p_cma_warm.append(_p(cma_col, warm_col))
            p_switch_warm.append(_p(switch_col, warm_col))

        out["p_CMA_vs_Switch"] = p_cma_switch
        out["p_CMA_vs_Warmstart"] = p_cma_warm
        out["p_Switch_vs_Warmstart"] = p_switch_warm
        return out.sort_values(["batch_size", "budget_mult"]).reset_index(drop=True)

    overhead_df = _canonicalize_overhead(overhead_df, overhead_csv_path=overhead_csv)

    transfer_method = str(args.transfer_method)
    safe_transfer_method = str(args.safe_transfer_method).strip()
    if safe_transfer_method == "":
        safe_transfer_method = ""

    transfer_label_map = {
        "bbob_B500": "transfer (COCO-learned, B=500D)",
        "bbob_B200": "transfer (COCO-learned, B=200D)",
        "fixed0p12": "transfer (fixed t=0.12)",
        "fixed0p18": "transfer (fixed t=0.18)",
        "fixed0p22": "safe transfer (fixed t=0.22)",
        "fixed0p25": "safe transfer (fixed t=0.25)",
    }
    transfer_label = transfer_label_map.get(transfer_method, f"transfer ({transfer_method})")
    safe_label = transfer_label_map.get(safe_transfer_method, f"safe ({safe_transfer_method})") if safe_transfer_method else ""
    show_methods = ["always_cma", transfer_method]
    if safe_transfer_method:
        show_methods.append(safe_transfer_method)
    show_methods.append("target_tuned")
    missing_methods = [m for m in show_methods if m not in set(transfer_df["method"].unique())]
    if missing_methods:
        raise SystemExit(f"Missing methods in transfer CSV: {missing_methods}")

    # Target naming/order for the main plot.
    target_label = {
        "bbob_B200_d40": "COCO D40 B200",
        "bbob_B500_d40": "COCO D40 B500",
        "bbob_B200_d10": "COCO D10 B200",
        "bbob_B200_d20": "COCO D20 B200",
        "logreg_synth": "LogReg (synth)",
        "logreg_breast_cancer": "LogReg (BC)",
        "logreg_digits0": "LogReg (digits0)",
        "mlp_digits0_heavytail_vs_noise_switch": "MLP (digits0, HT)",
        "hpo_noisy_logreg_digits0_sigma1p0": "HPO (digits0)",
        "rl_cartpole_cma_vs_berw": "RL (CartPole)",
        "lqr_heavytail_control": "LQR (HT)",
    }
    target_order = [
        "bbob_B200_d40",
        "bbob_B500_d40",
        "logreg_synth",
        "logreg_breast_cancer",
        "logreg_digits0",
        "mlp_digits0_heavytail_vs_noise_switch",
        "lqr_heavytail_control",
        "hpo_noisy_logreg_digits0_sigma1p0",
        "rl_cartpole_cma_vs_berw",
    ]
    # Targets to exclude from the summary.
    exclude_targets = {"bbob_B200_d20", "bbob_B200_d10"}
    targets = [t for t in target_order if t in set(transfer_df["target"].unique())]
    missing_targets = set(transfer_df["target"].unique()) - set(targets) - exclude_targets
    if missing_targets:
        # Append any unexpected targets deterministically.
        targets.extend(sorted(missing_targets))

    # Pivot transfer stats.
    pivot_regret = transfer_df.pivot_table(index="target", columns="method", values="regret_mean", aggfunc="first")
    pivot_acc = transfer_df.pivot_table(index="target", columns="method", values="accuracy", aggfunc="first")
    pivot_thr = transfer_df.pivot_table(index="target", columns="method", values="threshold", aggfunc="first")

    # Boundary / failure markers: transfer worse than always-CMA in regret_mean.
    delta = pivot_regret[transfer_method] - pivot_regret["always_cma"]
    boundary_targets = {t for t in targets if np.isfinite(delta.get(t, np.nan)) and float(delta[t]) > 0.0}

    safe_delta = None
    safe_boundary_targets: set[str] = set()
    if safe_transfer_method:
        safe_delta = pivot_regret[safe_transfer_method] - pivot_regret["always_cma"]
        safe_boundary_targets = {t for t in targets if np.isfinite(safe_delta.get(t, np.nan)) and float(safe_delta[t]) > 0.0}

    # Transfer threshold (expected constant across targets).
    transfer_thresholds = sorted(
        {
            float(v)
            for v in transfer_df.loc[transfer_df["method"] == transfer_method, "threshold"].dropna().tolist()
            if np.isfinite(float(v))
        }
    )
    transfer_threshold_str = ", ".join(f"{t:.2f}" for t in transfer_thresholds) if transfer_thresholds else "n/a"

    safe_threshold_str = ""
    if safe_transfer_method:
        safe_thresholds = sorted(
            {
                float(v)
                for v in transfer_df.loc[transfer_df["method"] == safe_transfer_method, "threshold"].dropna().tolist()
                if np.isfinite(float(v))
            }
        )
        safe_threshold_str = ", ".join(f"{t:.2f}" for t in safe_thresholds) if safe_thresholds else "n/a"

    # Build compact CSV for transfer.
    transfer_rows = []
    for t in targets:
        row = {
            "target": t,
            "target_label": target_label.get(t, t),
            "status": "boundary" if t in boundary_targets else "ok",
            "always_cma_regret_mean": float(pivot_regret.loc[t, "always_cma"]),
            f"{transfer_method}_threshold": float(pivot_thr.loc[t, transfer_method]) if pd.notna(pivot_thr.loc[t, transfer_method]) else np.nan,
            f"{transfer_method}_accuracy": float(pivot_acc.loc[t, transfer_method]),
            f"{transfer_method}_regret_mean": float(pivot_regret.loc[t, transfer_method]),
            "safe_method": safe_transfer_method if safe_transfer_method else "",
            "safe_status": ("boundary" if t in safe_boundary_targets else "ok") if safe_transfer_method else "",
            "safe_threshold": float(pivot_thr.loc[t, safe_transfer_method]) if (safe_transfer_method and safe_transfer_method in pivot_thr.columns and pd.notna(pivot_thr.loc[t, safe_transfer_method])) else np.nan,
            "safe_accuracy": float(pivot_acc.loc[t, safe_transfer_method]) if (safe_transfer_method and safe_transfer_method in pivot_acc.columns and pd.notna(pivot_acc.loc[t, safe_transfer_method])) else np.nan,
            "safe_regret_mean": float(pivot_regret.loc[t, safe_transfer_method]) if (safe_transfer_method and safe_transfer_method in pivot_regret.columns and pd.notna(pivot_regret.loc[t, safe_transfer_method])) else np.nan,
            "target_tuned_threshold": float(pivot_thr.loc[t, "target_tuned"]) if ("target_tuned" in pivot_thr.columns and pd.notna(pivot_thr.loc[t, "target_tuned"])) else np.nan,
            "target_tuned_accuracy": float(pivot_acc.loc[t, "target_tuned"]) if ("target_tuned" in pivot_acc.columns and pd.notna(pivot_acc.loc[t, "target_tuned"])) else np.nan,
            "target_tuned_regret_mean": float(pivot_regret.loc[t, "target_tuned"]) if ("target_tuned" in pivot_regret.columns and pd.notna(pivot_regret.loc[t, "target_tuned"])) else np.nan,
            "delta_transfer_minus_always": float(delta.loc[t]) if pd.notna(delta.loc[t]) else np.nan,
            "delta_safe_minus_always": float(safe_delta.loc[t]) if (safe_delta is not None and pd.notna(safe_delta.loc[t])) else np.nan,
        }
        transfer_rows.append(row)
    transfer_compact = pd.DataFrame(transfer_rows)
    transfer_compact_csv = out_dir / "transfer_summary_compact.csv"
    transfer_compact.to_csv(transfer_compact_csv, index=False)

    # Compact CSV for overhead curve (as-is, but normalized column names).
    overhead_compact = overhead_df.copy()
    overhead_compact_csv = out_dir / "overhead_curve_compact.csv"
    overhead_compact.to_csv(overhead_compact_csv, index=False)

    # ---------------------- Figure: Improvement Lollipop ----------------------
    apply_style()

    fig, ax = plt.subplots(figsize=get_figsize("single", aspect=1.2))

    # Build data for lollipop chart.
    # delta = always_cma_regret - transfer_regret (positive = transfer improves)
    lollipop_rows = []
    for t in targets:
        always_reg = float(pivot_regret.loc[t, "always_cma"])
        transfer_reg = float(pivot_regret.loc[t, transfer_method])
        d = always_reg - transfer_reg
        status = "boundary" if t in boundary_targets else "ok"
        lollipop_rows.append({
            "target": t,
            "target_label": target_label.get(t, t),
            "status": status,
            "delta": d,
        })
    lollipop_df = pd.DataFrame(lollipop_rows)

    # Split by status and sort within each group.
    df_ok = lollipop_df[lollipop_df["status"] == "ok"].sort_values("delta", ascending=True).reset_index(drop=True)
    df_boundary = lollipop_df[lollipop_df["status"] == "boundary"].sort_values("delta", ascending=True).reset_index(drop=True)

    # Build ordered task list: ok group (top), separator, boundary group (bottom).
    tasks: list[str] = list(df_ok["target_label"]) + [""] + list(df_boundary["target_label"])
    deltas: list[float] = list(df_ok["delta"]) + [np.nan] + list(df_boundary["delta"])

    y_positions = list(range(len(tasks)))
    for i, (task_name, d) in enumerate(zip(tasks, deltas)):
        if task_name == "" or np.isnan(d):
            continue
        color = COLORS["blue"] if d >= 0 else COLORS["red"]
        ax.hlines(i, 0, d, color=color, linewidth=1.5)
        ax.plot(d, i, "o", color=color, markersize=5)

    # Zero reference line.
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    # Separator between ok and boundary groups.
    sep_idx = len(df_ok)  # index of the empty row
    ax.axhline(sep_idx, color="gray", linewidth=0.5, linestyle="--")

    # Y-axis labels.
    ax.set_yticks(y_positions)
    ax.set_yticklabels(tasks, fontsize=7)
    ax.set_xlabel(r"$\Delta$ regret (always-CMA $-$ transfer)", fontsize=8)

    # Region annotations.
    xlims = ax.get_xlim()
    if len(df_ok) > 0:
        ax.text(xlims[1] * 0.7, len(df_ok) / 2, "transfer\neffective",
                ha="center", va="center", fontsize=7, color=COLORS["green"], alpha=0.7)
    if len(df_boundary) > 0:
        ax.text(xlims[0] * 0.7, sep_idx + len(df_boundary) / 2 + 1, "transfer\nfails",
                ha="center", va="center", fontsize=7, color=COLORS["red"], alpha=0.7)

    ax.invert_yaxis()
    add_grid(ax, which="major", axis="x", alpha=0.2)

    fig_png = out_dir / "transfer_overhead_main.png"
    fig_pdf = out_dir / "transfer_overhead_main.pdf"
    saved = save_figure(fig, str(fig_png))
    # save_figure handles png; also save pdf explicitly
    save_figure(fig, str(fig_pdf))
    plt.close(fig)

    # ---------------------- Table (Markdown) ----------------------
    md_lines: list[str] = []
    md_lines.append("# ProbeSwitch: Transfer + Overhead Summary\n")
    md_lines.append("This file contains two compact tables.\n")

    md_lines.append("## A) Threshold transfer (zero tuning)\n")
    md_lines.append(
        f"Transfer rule shown below: `{transfer_method}` ({transfer_label}, t={transfer_threshold_str}).\n"
    )
    if safe_transfer_method:
        md_lines.append(
            f"Safe rule: `{safe_transfer_method}` ({safe_label}, t={safe_threshold_str}). "
            "Status `boundary` marks regret worse than always-CMA.\n"
        )

    if safe_transfer_method:
        md_lines.append(
            "| target | transfer status | safe status | always-CMA regret | transfer t | transfer acc | transfer regret | safe t | safe acc | safe regret | tuned t | tuned acc | tuned regret |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        md_lines.append(
            "| target | status | always-CMA regret | transfer t | transfer acc | transfer regret | tuned t | tuned acc | tuned regret | Δ(transfer-always) |"
        )
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in targets:
        a_reg = pivot_regret.loc[t, "always_cma"]
        tr_t = pivot_thr.loc[t, transfer_method]
        tr_acc = pivot_acc.loc[t, transfer_method]
        tr_reg = pivot_regret.loc[t, transfer_method]
        tu_t = pivot_thr.loc[t, "target_tuned"] if ("target_tuned" in pivot_thr.columns) else np.nan
        tu_acc = pivot_acc.loc[t, "target_tuned"] if ("target_tuned" in pivot_acc.columns) else np.nan
        tu_reg = pivot_regret.loc[t, "target_tuned"] if ("target_tuned" in pivot_regret.columns) else np.nan

        if safe_transfer_method:
            s_t = pivot_thr.loc[t, safe_transfer_method]
            s_acc = pivot_acc.loc[t, safe_transfer_method]
            s_reg = pivot_regret.loc[t, safe_transfer_method]
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        target_label.get(t, t),
                        "boundary" if t in boundary_targets else "ok",
                        "boundary" if t in safe_boundary_targets else "ok",
                        _format_float(a_reg, digits=3),
                        _format_float(tr_t, digits=2),
                        _format_float(tr_acc, digits=3),
                        _format_float(tr_reg, digits=3),
                        _format_float(s_t, digits=2),
                        _format_float(s_acc, digits=3),
                        _format_float(s_reg, digits=3),
                        _format_float(tu_t, digits=2),
                        _format_float(tu_acc, digits=3),
                        _format_float(tu_reg, digits=3),
                    ]
                )
                + " |"
            )
        else:
            d = delta.loc[t]
            md_lines.append(
                "| "
                + " | ".join(
                    [
                        target_label.get(t, t),
                        "boundary" if t in boundary_targets else "ok",
                        _format_float(a_reg, digits=3),
                        _format_float(tr_t, digits=2),
                        _format_float(tr_acc, digits=3),
                        _format_float(tr_reg, digits=3),
                        _format_float(tu_t, digits=2),
                        _format_float(tu_acc, digits=3),
                        _format_float(tu_reg, digits=3),
                        _format_float(d, digits=3),
                    ]
                )
                + " |"
            )

    md_lines.append("\n## B) VOI / overhead-vs-gain curve (logreg sweep)\n")
    md_lines.append("Lower is better. `bs=8` is stochastic/high-misranking; `bs=256` is deterministic.\n")
    md_lines.append(
        "| batch | B/d | median(CMA) | median(Switch) | median(Warmstart) | p(CMA vs Switch) | p(CMA vs Warmstart) | p(Switch vs Warmstart) |"
    )
    md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    overhead_df_sorted = overhead_df.sort_values(["batch_size", "budget_mult"])
    for _, r in overhead_df_sorted.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(int(r["batch_size"])),
                    str(int(r["budget_mult"])),
                    _format_float(r["median_post_true_CMA"], digits=3),
                    _format_float(r["median_post_true_Switch"], digits=3),
                    _format_float(r["median_post_true_Warmstart"], digits=3),
                    _format_p(r.get("p_CMA_vs_Switch")),
                    _format_p(r.get("p_CMA_vs_Warmstart")),
                    _format_p(r.get("p_Switch_vs_Warmstart")),
                ]
            )
            + " |"
        )

    table_md = out_dir / "summary.md"
    table_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    _write_readme(
        out_dir,
        fig_png=fig_png,
        table_md=table_md,
        transfer_method=transfer_method,
        safe_transfer_method=safe_transfer_method,
    )

    print("Wrote:", repo_relpath(str(fig_png)))
    print("Wrote:", repo_relpath(str(fig_pdf)))
    print("Wrote:", repo_relpath(str(transfer_compact_csv)))
    print("Wrote:", repo_relpath(str(overhead_compact_csv)))
    print("Wrote:", repo_relpath(str(table_md)))
    print("Wrote:", repo_relpath(str(out_dir / "README.md")))


if __name__ == "__main__":
    main()
