#!/usr/bin/env python
"""Reproduce Amine's robustness/clamping and intervention experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.robustness_clamping_and_intervention.metrics import (
    js_divergence,
    spearman_over_union,
    topk_overlap,
)
from experiments.robustness_clamping_and_intervention.tasks import load_benchmark


def resolve_robustness_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    candidate = ROBUSTNESS_DIR / path
    if candidate.exists():
        return candidate
    return candidate


def check_model_dir(path: str | Path, label: str) -> Path:
    model_dir = resolve_robustness_path(path)
    missing = [
        name
        for name in ["args.json", "model_state_dict.pth"]
        if not (model_dir / name).exists()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"{label} checkpoint is incomplete at {model_dir}. "
            f"Missing: {missing_list}. Download the large checkpoint files "
            "described in experiments/robustness_clamping_and_intervention/README.md "
            "before running this experiment."
        )
    return model_dir


def load_release_models(args: argparse.Namespace, include_baselines: bool):
    requested = [
        ("ProtoT", "protot", args.protot_path),
    ]
    if include_baselines:
        requested.extend([
            ("Mamba", "mamba", args.mamba_path),
            ("LLaMA", "llama", args.llama_path),
            ("DeltaNet", "deltanet", args.deltanet_path),
        ])

    resolved = [
        (name, kind, check_model_dir(path, name))
        for name, kind, path in requested
    ]

    from experiments.robustness_clamping_and_intervention.models import ModelSpec, load_model

    return [
        load_model(
            ModelSpec(name=name, kind=kind, path=str(path), device=args.device),
            compute_full_ppl=args.compute_full_ppl,
        )
        for name, kind, path in resolved
    ]


def load_pairs(name: str, path: str, args: argparse.Namespace):
    return load_benchmark(
        name,
        path=str(resolve_robustness_path(path)),
        n=args.n_per_slice,
        seed=args.seed,
        shuffle=True,
    )


def check_datasets(args: argparse.Namespace) -> None:
    perturbation = load_pairs("perturbation", args.perturbation_path, args)
    intervention = load_pairs("intervention", args.intervention_path, args)
    print(f"perturbation pairs: {len(perturbation)}")
    print(f"intervention pairs: {len(intervention)}")
    print("perturbation slices:", sorted({row[0] for row in perturbation}))
    print("intervention slices:", sorted({row[0] for row in intervention}))


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_get_routing(model, text):
    if hasattr(model, "get_last_routing"):
        return model.get_last_routing(text)
    return None


def try_forced_probs(model, text, force_write=None, force_read=None):
    if hasattr(model, "next_token_probs_forced"):
        return model.next_token_probs_forced(
            text,
            force_write=force_write,
            force_read=force_read,
        )
    return None


def pmr_on_pair(proto_model, text_a: str, text_b: str) -> dict:
    p_a = proto_model.next_token_probs(text_a)
    p_b = proto_model.next_token_probs(text_b)
    base = float(js_divergence(p_a, p_b).item())

    routing = try_get_routing(proto_model, text_a)
    if routing is None or base <= 0.0:
        return {
            "JS_base": base,
            "JS_read": None,
            "JS_write": None,
            "JS_clamped": None,
            "PMR": None,
            "PMR_read": None,
            "PMR_write": None,
        }

    p_b_read = try_forced_probs(
        proto_model,
        text_b,
        force_read=routing.get("read_weights", None),
    )
    js_read = float(js_divergence(p_a, p_b_read).item()) if p_b_read is not None else None
    pmr_read = (base - js_read) / base if js_read is not None else None

    p_b_write = try_forced_probs(
        proto_model,
        text_b,
        force_write=routing.get("write_weights", None),
    )
    js_write = float(js_divergence(p_a, p_b_write).item()) if p_b_write is not None else None
    pmr_write = (base - js_write) / base if js_write is not None else None

    candidates = [value for value in [js_read, js_write] if value is not None]
    js_clamped = min(candidates) if candidates else None
    pmr = (base - js_clamped) / base if js_clamped is not None else None

    return {
        "JS_base": base,
        "JS_read": js_read,
        "JS_write": js_write,
        "JS_clamped": js_clamped,
        "PMR": pmr,
        "PMR_read": pmr_read,
        "PMR_write": pmr_write,
    }


def run_pmr_raw(proto_model, pairs, progress_every: int = 250) -> pd.DataFrame:
    rows = []
    for idx, (slice_name, text_a, text_b) in enumerate(pairs, start=1):
        out = pmr_on_pair(proto_model, text_a, text_b)
        out.update({"slice": slice_name, "A": text_a, "B": text_b})
        rows.append(out)
        if progress_every and idx % progress_every == 0:
            print(f"PMR pairs evaluated: {idx}/{len(pairs)}")
    return pd.DataFrame(rows)


def summarize_pmr(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["PMR"].notna()].copy()
    grouped = valid.groupby("slice").agg(
        PMR_mean=("PMR", "mean"),
        PMR_std=("PMR", "std"),
        PMR_pos_rate=("PMR", lambda values: float((values > 0).mean())),
        JS_base_mean=("JS_base", "mean"),
        JS_clamped_mean=("JS_clamped", "mean"),
        n=("PMR", "count"),
    )
    total = df.groupby("slice").size().rename("n_total")
    summary = grouped.join(total, how="right").reset_index()
    summary["coverage"] = summary["n"] / summary["n_total"]
    return summary.sort_values("slice").reset_index(drop=True)


def pmr_call_table(
    pmr_summary: pd.DataFrame,
    pmr_mean_threshold: float,
    pmr_pos_threshold: float,
) -> pd.DataFrame:
    df = pmr_summary.copy()
    df["mediated"] = (
        (df["PMR_mean"] >= pmr_mean_threshold)
        & (df["PMR_pos_rate"] >= pmr_pos_threshold)
    )
    cols = [
        "slice",
        "PMR_mean",
        "PMR_pos_rate",
        "JS_base_mean",
        "JS_clamped_mean",
        "coverage",
        "n",
        "n_total",
        "mediated",
    ]
    return df[[col for col in cols if col in df.columns]].sort_values("slice")


def run_pmr_experiment(proto_model, args: argparse.Namespace, outdir: Path) -> None:
    pairs = load_pairs("perturbation", args.perturbation_path, args)
    print(f"Running PMR on {len(pairs)} perturbation pairs")
    raw = run_pmr_raw(proto_model, pairs)
    summary = summarize_pmr(raw)
    calls = pmr_call_table(
        summary,
        pmr_mean_threshold=args.pmr_mean_threshold,
        pmr_pos_threshold=args.pmr_pos_threshold,
    )

    raw.to_csv(outdir / "pmr_raw.csv", index=False)
    summary.to_csv(outdir / "pmr_summary.csv", index=False)
    calls.to_csv(outdir / "pmr_call_table.csv", index=False)
    write_markdown_table(calls, outdir / "pmr_call_table.md")
    print(calls.to_string(index=False))


@torch.no_grad()
def eval_intervention_pairs(model, pairs, k_top: int = 10) -> pd.DataFrame:
    rows = []
    for slice_name, text_a, text_b in pairs:
        p_a = model.next_token_probs(text_a)
        p_b = model.next_token_probs(text_b)
        idx_a = torch.topk(p_a, k_top).indices
        rows.append(
            {
                "slice": slice_name,
                "JS": float(js_divergence(p_a, p_b).item()),
                "topk_overlap": topk_overlap(p_a, p_b, k=k_top),
                "spearman": spearman_over_union(p_a, p_b, k=max(k_top, 20)),
                "top1_same": int(torch.argmax(p_a).item() == torch.argmax(p_b).item()),
                "mass_on_Ak_in_B": float(p_b[idx_a].sum().item()),
                "A": text_a,
                "B": text_b,
            }
        )
    return pd.DataFrame(rows)


def summarize_intervention(df: pd.DataFrame, model_name: str, ppl=None) -> pd.DataFrame:
    summary = df.groupby("slice").agg(
        JS_mean=("JS", "mean"),
        JS_std=("JS", "std"),
        overlap_mean=("topk_overlap", "mean"),
        overlap_std=("topk_overlap", "std"),
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        top1_same_rate=("top1_same", "mean"),
        mass_on_Ak_in_B_mean=("mass_on_Ak_in_B", "mean"),
        n=("JS", "count"),
    ).reset_index()
    summary["model"] = model_name
    if ppl is not None:
        summary["ppl"] = ppl
    return summary


def wide_intervention_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "JS_mean",
        "overlap_mean",
        "spearman_mean",
        "top1_same_rate",
        "mass_on_Ak_in_B_mean",
    ]
    wide = summary.pivot(index="slice", columns="model", values=metrics)
    wide.columns = [f"{metric}__{model}" for metric, model in wide.columns]
    return wide.reset_index().sort_values("slice")


def run_intervention_experiment(models, args: argparse.Namespace, outdir: Path) -> None:
    pairs = load_pairs("intervention", args.intervention_path, args)
    print(f"Running intervention evaluation on {len(pairs)} pairs")
    raw_tables = {}
    summaries = []

    for model in models:
        raw = eval_intervention_pairs(model, pairs, k_top=args.k_top)
        raw_tables[model.name] = raw
        raw.to_csv(outdir / f"intervention_raw_{model.name}.csv", index=False)
        summaries.append(summarize_intervention(raw, model.name, ppl=getattr(model, "ppl", None)))

    summary = pd.concat(summaries, ignore_index=True)
    wide = wide_intervention_table(summary)
    summary.to_csv(outdir / "intervention_summary_long.csv", index=False)
    wide.to_csv(outdir / "intervention_comparison_table.csv", index=False)
    write_markdown_table(wide, outdir / "intervention_comparison_table.md")
    print(wide.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Amine's ProtoT robustness and intervention experiments."
    )
    parser.add_argument("--mode", choices=["all", "pmr", "intervention", "check-data"], default="all")
    parser.add_argument("--protot-path", default="ProtoT")
    parser.add_argument("--mamba-path", default="Mamba")
    parser.add_argument("--llama-path", default="LLaMA")
    parser.add_argument("--deltanet-path", default="DeltaNet")
    parser.add_argument("--perturbation-path", default="perturbation_dataset/perturbation_benchmark_clean.jsonl")
    parser.add_argument("--intervention-path", default="intervention_dataset/intervention_benchmark_clean.jsonl")
    parser.add_argument("--outdir", default="results/amine_reproduction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-per-slice", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-top", type=int, default=10)
    parser.add_argument("--compute-full-ppl", action="store_true")
    parser.add_argument("--pmr-mean-threshold", type=float, default=0.001)
    parser.add_argument("--pmr-pos-threshold", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROBUSTNESS_DIR)

    if args.mode == "check-data":
        check_datasets(args)
        return

    outdir = resolve_robustness_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    include_baselines = args.mode in {"all", "intervention"}
    models = load_release_models(args, include_baselines=include_baselines)
    models_by_name = {model.name: model for model in models}

    if args.mode in {"all", "pmr"}:
        run_pmr_experiment(models_by_name["ProtoT"], args, outdir)
    if args.mode in {"all", "intervention"}:
        run_intervention_experiment(models, args, outdir)

    print(f"Wrote outputs to {outdir}")


if __name__ == "__main__":
    main()
