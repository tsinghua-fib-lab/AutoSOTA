# robustness/runner.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from robustness.metrics import js_divergence, topk_overlap, spearman_over_union


def eval_pairs(model, labeled_pairs, k=10):
    """Evaluate one model on perturbation pairs -> raw DataFrame."""
    rows = []
    for slice_name, a, b in labeled_pairs:
        pA, pB = model.next_token_probs(a), model.next_token_probs(b)
        rows.append({
            "slice": slice_name,
            "JS": float(js_divergence(pA, pB).item()),
            "topk_overlap": topk_overlap(pA, pB, k=k),
            "spearman": spearman_over_union(pA, pB, k=max(k, 20)),
            "A": a, "B": b
        })
    return pd.DataFrame(rows)


def summarize(df, model_name, ppl=None):
    """Aggregate metrics per slice for one model."""
    g = df.groupby("slice").agg(
        JS_mean=("JS", "mean"), JS_std=("JS", "std"),
        overlap_mean=("topk_overlap", "mean"), overlap_std=("topk_overlap", "std"),
        spearman_mean=("spearman", "mean"), spearman_std=("spearman", "std"),
        n=("JS", "count")
    ).reset_index()
    g["model"] = model_name
    if ppl is not None:
        g["ppl"] = ppl
    return g


def run_eval(models, pairs, names=None, ppls=None, k=10):
    """Evaluate multiple models and return combined summary DataFrame."""
    all_summaries = []
    for i, model in enumerate(models):
        name = names[i] if names is not None else getattr(model, "name", f"Model{i}")
        df = eval_pairs(model, pairs, k=k)

        # collect PPLs
        ppl_val = getattr(model, "ppl", None)
        saved_ppl = getattr(model, "saved_ppl", None)
        recomputed_ppl = getattr(model, "recomputed_ppl", None)

        g = summarize(df, name, ppl=ppl_val)
        g["saved_ppl"] = saved_ppl
        g["recomputed_ppl"] = recomputed_ppl
        all_summaries.append(g)

    # filter out empties and all-NA frames to avoid FutureWarning
    valid_summaries = [
        g for g in all_summaries
        if not g.empty and not g.isna().all().all()
    ]
    return pd.concat(valid_summaries, ignore_index=True)


def decision_rule(summary, model_a=None, model_b=None, threshold=0.9):
    """
    Symmetric decision rule: per slice, compare JS divergence of two models.
    - Winner = model with lower JS (if within threshold margin).
    - Tie if both are within ±(1-threshold) margin of each other.
    """
    unique_models = summary["model"].unique().tolist()
    if len(unique_models) != 2 and (model_a is None or model_b is None):
        raise ValueError(
            f"Expected exactly 2 models, found {unique_models}. "
            "Pass model_a/model_b explicitly."
        )

    if model_a is None or model_b is None:
        model_a, model_b = unique_models

    merged = summary.pivot(index="slice", columns="model", values="JS_mean").reset_index()
    merged = merged.dropna(subset=[model_a, model_b])

    wins_a, wins_b, ties, rows = [], [], [], []
    for _, row in merged.iterrows():
        js_a, js_b = row[model_a], row[model_b]
        ratio = js_a / js_b if js_b > 0 else float("inf")

        if js_a <= threshold * js_b:
            winner = model_a
            wins_a.append(row["slice"])
        elif js_b <= threshold * js_a:
            winner = model_b
            wins_b.append(row["slice"])
        else:
            winner = "tie"
            ties.append(row["slice"])

        rows.append({
            "slice": row["slice"],
            model_a: js_a,
            model_b: js_b,
            "ratio (A/B)": ratio,
            "winner": winner
        })

    # collect perplexities if available
    ppls = {}
    for m in [model_a, model_b]:
        if "ppl" in summary.columns:
            vals = summary.loc[summary["model"] == m, "ppl"].unique()
            if len(vals) == 1:
                ppls[m] = float(vals[0])

    # decide overall winner
    if len(wins_a) > len(wins_b):
        overall = model_a
    elif len(wins_b) > len(wins_a):
        overall = model_b
    else:
        overall = "tie"

    return {
        "overall_winner": overall,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "details": rows,
        "models": (model_a, model_b),
        "perplexities": ppls
    }


def make_report(summary, outdir="results"):
    """
    Make robustness reports:
    - CSV: full metrics (JS, overlap, spearman) per slice/model
    - LaTeX: compact JS-only table (for paper, best per slice bolded)
    - Plots: all metrics
    """
    os.makedirs(outdir, exist_ok=True)

    # full pivot (for CSV + plots)
    full_report = summary.pivot(
        index="slice", columns="model",
        values=["JS_mean", "overlap_mean", "spearman_mean"]
    ).round(6)

    # save full CSV
    full_report.to_csv(os.path.join(outdir, "robustness_summary_table.csv"))

    # JS-only pivot (for LaTeX in the paper)
    js_report = summary.pivot(
        index="slice", columns="model", values="JS_mean"
    ).round(6)

    # Bold the lowest value per row (work on string DataFrame to avoid dtype warnings)
    js_bold = pd.DataFrame(index=js_report.index, columns=js_report.columns, dtype=str)
    for i, row in js_report.iterrows():
        min_val = row.min()
        for col in js_report.columns:
            val = row[col]
            if val == min_val:
                js_bold.at[i, col] = f"\\textbf{{{val:.6f}}}"
            else:
                js_bold.at[i, col] = f"{val:.6f}"

    latex_path = os.path.join(outdir, "robustness_summary_table.tex")
    with open(latex_path, "w") as f:
        f.write(js_bold.to_latex(
            escape=False,  # allow bold text
            multicolumn=True,
            multirow=True,
            column_format="l" + "r" * len(js_report.columns),
            index_names=True,
            caption="Slice-level robustness: Jensen--Shannon divergence (lower is better). Best per slice in bold.",
            label="tab:robustness_slices"
        ))
    print(f"Saved LaTeX (JS-only) table with bold best scores to {latex_path}")

    # plots (still use full metrics)
    for m in ["JS_mean", "overlap_mean", "spearman_mean"]:
        plt.figure(figsize=(8, 5))
        try:
            sns.barplot(data=summary, x="slice", y=m, hue="model",
                        errorbar="sd", palette="Set2")
        except TypeError:  # fallback for older seaborn
            sns.barplot(data=summary, x="slice", y=m, hue="model",
                        ci="sd", palette="Set2")

        plt.title(f"{m} by slice")
        plt.ylabel(m)
        plt.xlabel("Slice type")
        plt.legend(title="Model")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

    return js_report