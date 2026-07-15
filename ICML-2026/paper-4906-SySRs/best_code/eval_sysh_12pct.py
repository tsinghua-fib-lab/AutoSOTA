"""Evaluate SySH (Smart Sequential Halving) at 12% budget across 15 datasets."""
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/repo")
import bai_algs
from bai_algs import smart_sequential_halving as sysh

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
    budget = max(n_models, (int(0.12 * n_models * n_tasks) // n_models) * n_models)
    arms = sysh(n_items=budget, k=1000, verbose=False)
    acc = float((np.asarray(arms) == 0).mean()) * 100
    results[ds] = acc
    print(f"  {ds:20s}: {acc:.1f}%", flush=True)

avg = np.mean(list(results.values()))
print(f"\nSySH Avg Identification Accuracy @ 12% budget: {avg:.1f}%")
print(f"Baseline SySRs: 95.2%")
