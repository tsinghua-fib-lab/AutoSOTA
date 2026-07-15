"""Reproduce SySRs identification accuracy at 12% budget across 15 datasets.

Settings: budget_percentage=12, n_runs=1000, sampling=without_replacement,
          decoding=greedy, n_datasets=15

Paper reports: SySRs @ 12% -> 95.1% identification accuracy (Table 2).
"""
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/repo")
import bai_algs
from bai_algs import smart_successive_rejects_wo_replacement_no_budget_limit as smart_sr

ALL_DATASETS = [
    "arc_challenge", "bbh", "commonsense", "gpqa", "gsm", "ifeval",
    "legalbench", "math", "med_qa", "mmlu", "mmlu_pro",
    "musr", "narrative_qa", "natural_qa", "wmt_14",
]

results = {}
for ds in ALL_DATASETS:
    pkl_path = Path("/repo/datasets") / ds / "model_accuracies_filtered.pkl"
    data = pickle.load(open(pkl_path, "rb"))
    bai_algs.model_accuracies = data
    n_models, n_tasks = data.shape
    raw = int(0.12 * n_models * n_tasks)
    budget = max(n_models, ((raw + n_models - 1) // n_models) * n_models)
    budget = min(budget, n_models * n_tasks)
    arms = smart_sr(n_items=budget, k=1000, verbose=False)
    acc = float((np.asarray(arms) == 0).mean()) * 100
    results[ds] = acc
    print(f"  {ds:20s}: {acc:.1f}%", flush=True)

avg = np.mean(list(results.values()))
print(f"\nSySRs Avg Identification Accuracy @ 12% budget: {avg:.1f}%")
print(f"Paper reports: 95.1%")
