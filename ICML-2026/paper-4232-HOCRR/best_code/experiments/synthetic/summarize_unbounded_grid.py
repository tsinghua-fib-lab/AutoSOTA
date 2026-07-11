#!/usr/bin/env python3
"""
Summarize unbounded synthetic grid runs (Section 7.1 style).

For each synthetic function and output tolerance ε_y, pick grid parameters using
the same spirit as the paper draft:

  1. Prefer **100% soundness** (certified radius ≤ true radius at every test point).
  2. Subject to that, **maximize mean certified radius** (if no fully sound config exists,
     fall back to highest soundness, then highest mean certified radius).

Reads JSONs produced by test_unbounded_certifiers_synthetic.py (nested
``results[function_name].summary``).

Outputs:
  - CSV with one row per (function, ε_y, method) at the selected grid point
  - Optional LaTeX snippet for the table body + JSON stats for inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FILENAME_RE = re.compile(
    r"unbounded_sigma(?P<sigma>[\d.]+)_epsy(?P<eps_y>[\d.]+)_alpha(?P<alpha>[\d.]+)_"
)


def parse_run_path(path: Path) -> Optional[Tuple[float, float, float]]:
    m = FILENAME_RE.search(path.name)
    if not m:
        return None
    return float(m["sigma"]), float(m["eps_y"]), float(m["alpha"])


def load_summaries(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Any] = {}
    results = data.get("results") or {}
    for func_name, block in results.items():
        out[func_name] = block.get("summary") or {}
    return out


def _pick_best_sound_then_radius(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Prefer soundness==1; else max soundness; then max mean certified radius."""
    if not candidates:
        raise ValueError("empty candidate list")
    full = float(max(c["soundness"] for c in candidates))
    perfect = [c for c in candidates if c["soundness"] >= 1.0 - 1e-9]
    pool = perfect if perfect else [c for c in candidates if c["soundness"] >= full - 1e-9]
    return max(pool, key=lambda c: (c["mean_cert"],))


def best_cg_for_eps(
    rows: List[Dict[str, Any]], func: str, eps_y: float
) -> Dict[str, Any]:
    """Best σ for (C,G): dedupe α; prefer 100% sound then max mean certified radius."""
    cand = [r for r in rows if r["eps_y"] == eps_y and func in r["cg_by_func"]]
    if not cand:
        raise ValueError(f"No rows for {func} eps_y={eps_y}")
    by_sigma: Dict[float, Dict[str, Any]] = {}
    for r in cand:
        s = r["sigma"]
        s_obj = r["cg_by_func"][func]
        mr = s_obj.get("mean_radius_unbounded")
        snd = s_obj.get("soundness_unbounded")
        if mr is None or snd is None:
            continue
        prev = by_sigma.get(s)
        row = {
            "sigma": s,
            "mean_radius_unbounded": mr,
            "mean_ratio_unbounded": s_obj.get("mean_ratio_unbounded"),
            "soundness_unbounded": snd,
            "source": str(r["path"]),
            "mean_cert": mr,
            "soundness": float(snd),
        }
        if prev is None or row["mean_cert"] > prev["mean_cert"]:
            by_sigma[s] = row
    if not by_sigma:
        raise ValueError(f"No CG summaries for {func} eps_y={eps_y}")
    best = _pick_best_sound_then_radius(list(by_sigma.values()))
    best.pop("mean_cert", None)
    best.pop("soundness", None)
    return best


def best_alpha_for_eps(
    rows: List[Dict[str, Any]], func: str, eps_y: float
) -> Dict[str, Any]:
    """Best (σ, α): prefer 100% sound then max mean certified radius."""
    cand = [r for r in rows if r["eps_y"] == eps_y and func in r["alpha_by_func"]]
    if not cand:
        raise ValueError(f"No rows for {func} eps_y={eps_y}")
    candidates: List[Dict[str, Any]] = []
    for r in cand:
        a_obj = r["alpha_by_func"][func]
        mr = a_obj.get("mean_radius_alpha")
        snd = a_obj.get("soundness_alpha")
        if mr is None or snd is None:
            continue
        candidates.append(
            {
                "sigma": r["sigma"],
                "alpha": r["alpha_trim"],
                "mean_radius_alpha": mr,
                "mean_ratio_alpha": a_obj.get("mean_ratio_alpha"),
                "soundness_alpha": snd,
                "source": str(r["path"]),
                "mean_cert": mr,
                "soundness": float(snd),
            }
        )
    if not candidates:
        raise ValueError(f"No alpha summaries for {func} eps_y={eps_y}")
    best = _pick_best_sound_then_radius(candidates)
    best.pop("mean_cert", None)
    best.pop("soundness", None)
    return best


def collect_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        parsed = parse_run_path(p)
        if parsed is None:
            continue
        sigma, eps_y, alpha = parsed
        summaries = load_summaries(p)
        cg_by_func: Dict[str, Any] = {}
        alpha_by_func: Dict[str, Any] = {}
        for fn, summ in summaries.items():
            cg_by_func[fn] = {
                "mean_radius_unbounded": summ.get("mean_radius_unbounded"),
                "mean_ratio_unbounded": summ.get("mean_ratio_unbounded"),
                "soundness_unbounded": summ.get("soundness_unbounded"),
            }
            alpha_by_func[fn] = {
                "mean_radius_alpha": summ.get("mean_radius_alpha"),
                "mean_ratio_alpha": summ.get("mean_ratio_alpha"),
                "soundness_alpha": summ.get("soundness_alpha"),
            }
        rows.append(
            {
                "path": p,
                "sigma": sigma,
                "eps_y": eps_y,
                "alpha_trim": alpha,
                "cg_by_func": cg_by_func,
                "alpha_by_func": alpha_by_func,
            }
        )
    return rows


def latex_table_rows(
    func_order: List[str],
    eps_order: List[float],
    display_names: Dict[str, str],
    best_cg: Dict[Tuple[str, float], Dict[str, Any]],
    best_a: Dict[Tuple[str, float], Dict[str, Any]],
) -> str:
    """Body rows matching tables/SECTION_7.1_LATEX_UPDATED.tex (multirow layout)."""
    lines: List[str] = []
    last_func = func_order[-1]
    for func in func_order:
        dname = display_names.get(func, func)
        lines.append(f"    \\multirow{{4}}{{=}}{{{dname}}}")
        for i, eps in enumerate(eps_order):
            cg = best_cg[(func, eps)]
            al = best_a[(func, eps)]
            cg_t = cg["mean_ratio_unbounded"]
            cg_s = cg["soundness_unbounded"]
            al_t = al["mean_ratio_alpha"]
            al_s = al["soundness_alpha"]
            sig_cg = cg["sigma"]
            sig_a = al["sigma"]
            al_trim = al["alpha"]
            alpha_note = f" ($\\alpha={al_trim:.2f}$)"
            if i > 0:
                lines.append("        \\cmidrule(l){2-6}")
            pct_cg = int(round(100 * float(cg_s)))
            pct_al = int(round(100 * float(al_s)))
            lines.append(
                f"        & \\multirow{{2}}{{*}}{{{eps:.1f}}} & $(C, G)$ & {sig_cg:.2f} & {cg_t:.3f} & ({pct_cg}\\%) \\\\"
            )
            lines.append(
                f"        &                      & $\\alpha$-smoothing{alpha_note} & {sig_a:.2f} & {al_t:.3f} & ({pct_al}\\%) \\\\"
            )
        if func != last_func:
            lines.append("    \\midrule")
        else:
            lines.append("    \\bottomrule")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("unbounded_synthetic_experiments_results"),
        help="Directory containing unbounded_sigma*_epsy*_alpha*.json",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="unbounded_sigma*_epsy*_alpha*_202604*.json",
        help="Glob relative to input_dir; set to wider pattern if needed",
    )
    ap.add_argument(
        "--exclude_substr",
        action="append",
        default=["rerun"],
        help="Skip files whose path contains this substring (repeatable)",
    )
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--json_out", type=Path, default=None)
    ap.add_argument("--latex_fragment", type=Path, default=None)
    args = ap.parse_args()

    paths = sorted(args.input_dir.glob(args.glob))
    for ex in args.exclude_substr:
        paths = [p for p in paths if ex not in str(p)]
    if not paths:
        raise SystemExit(f"No JSON matched under {args.input_dir} / {args.glob}")

    rows = collect_rows(paths)
    func_order = [
        "unbounded_quadratic",
        "unbounded_slice",
        "unbounded_sandwich",
    ]
    eps_order = [0.2, 0.5]
    display_names = {
        "unbounded_quadratic": "Quadratic",
        "unbounded_slice": "Slice",
        "unbounded_sandwich": "Sandwich",
    }

    best_cg: Dict[Tuple[str, float], Dict[str, Any]] = {}
    best_a: Dict[Tuple[str, float], Dict[str, Any]] = {}
    csv_rows: List[Dict[str, Any]] = []

    for func in func_order:
        for eps in eps_order:
            bcg = best_cg_for_eps(rows, func, eps)
            bal = best_alpha_for_eps(rows, func, eps)
            best_cg[(func, eps)] = bcg
            best_a[(func, eps)] = bal
            csv_rows.append(
                {
                    "function": func,
                    "eps_y": eps,
                    "method": "C_G",
                    "best_sigma": bcg["sigma"],
                    "best_alpha": "",
                    "mean_certified_radius": bcg["mean_radius_unbounded"],
                    "mean_tightness": bcg["mean_ratio_unbounded"],
                    "soundness": bcg["soundness_unbounded"],
                    "source_json": bcg["source"],
                }
            )
            csv_rows.append(
                {
                    "function": func,
                    "eps_y": eps,
                    "method": "alpha_smoothing",
                    "best_sigma": bal["sigma"],
                    "best_alpha": bal["alpha"],
                    "mean_certified_radius": bal["mean_radius_alpha"],
                    "mean_tightness": bal["mean_ratio_alpha"],
                    "soundness": bal["soundness_alpha"],
                    "source_json": bal["source"],
                }
            )

    repo_tables = Path("tables")
    csv_path = args.csv or (repo_tables / "unbounded_synthetic_best_cv_summary.csv")
    json_path = args.json_out or (repo_tables / "unbounded_synthetic_best_cv_summary.json")
    latex_path = args.latex_fragment or (repo_tables / "SECTION_7.1_table_body.tex")
    repo_tables.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    payload = {
        "input_files": [str(p) for p in paths],
        "rule": "prefer_soundness_1_then_maximize_mean_certified_radius_per_method",
        "best_cg": {f"{k[0]}_eps{k[1]}": v for k, v in best_cg.items()},
        "best_alpha": {f"{k[0]}_eps{k[1]}": v for k, v in best_a.items()},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    frag = latex_table_rows(func_order, eps_order, display_names, best_cg, best_a)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(frag)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {latex_path}")
    print(
        "Note: tables/SECTION_7.1_LATEX_UPDATED.tex embeds a static copy of the tabular "
        "rows; paste tables/SECTION_7.1_table_body.tex into that file after regenerating."
    )
    for r in csv_rows:
        print(
            f"{r['function'][:12]:12} ε={r['eps_y']} {r['method']:16} σ*={r['best_sigma']} "
            f"α*={r.get('best_alpha', '')} tight={r['mean_tightness']:.3f} sound={r['soundness']:.0%}"
        )


if __name__ == "__main__":
    main()
