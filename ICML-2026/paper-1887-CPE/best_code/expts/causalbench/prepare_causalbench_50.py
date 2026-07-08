# prepare_causalbench_50.py
# 	loads the exported .npz
# 	selects top-50 genes by variance (deterministic)
# 	searches inside data_directory for a likely ground-truth edge file and parses it
# 	saves a clean 50-node package: X_50, genes_50, A_ref_50, plus a meta JSON

# scripts/prepare_causalbench_50.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def load_export_npz(export_npz: Path) -> tuple[np.ndarray, List[str], List[str]]:
    d = np.load(export_npz, allow_pickle=True)
    X = np.asarray(d["X"], dtype=np.float32)
    genes = [str(x) for x in d["gene_names"].tolist()]
    intervs = [str(x) for x in d["interventions"].tolist()]
    return X, genes, intervs


def pick_topk_by_variance(X: np.ndarray, genes: List[str], k: int, seed: int = 0) -> List[int]:
    # variance across samples (axis 0)
    v = X.var(axis=0)
    # tie-break deterministically by gene name
    order = np.lexsort((np.array(genes, dtype=object), -v))
    idx = order[:k].tolist()
    idx.sort()
    return idx


def _looks_like_edge_file(path: Path) -> bool:
    name = path.name.lower()
    if any(tok in name for tok in ["gold", "truth", "ground", "benchmark", "edges", "network", "grn"]):
        return True
    return False


def _score_edge_file(path: Path) -> float:
    # heuristic scoring
    name = path.name.lower()
    score = 0.0
    for tok, w in [
        ("gold", 3.0),
        ("truth", 3.0),
        ("ground", 2.5),
        ("benchmark", 2.0),
        ("network", 1.5),
        ("edge", 1.5),
        ("grn", 1.0),
        ("directed", 1.0),
    ]:
        if tok in name:
            score += w
    # prefer small-ish text-like files
    if path.suffix.lower() in [".csv", ".tsv", ".txt"]:
        score += 1.0
    return score


def find_best_ground_truth_file(data_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.stat().st_size < 300_000_000:  # avoid huge blobs
            if _looks_like_edge_file(p):
                candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=_score_edge_file, reverse=True)
    return candidates[0]


def parse_edge_file(fp: Path) -> List[Tuple[str, str]]:
    """
    Very forgiving parser:
      - handles CSV/TSV/space separated
      - expects at least 2 columns per row: src, dst
      - ignores comments / headers if they don't parse
    """
    edges: List[Tuple[str, str]] = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # split on comma, tab, or whitespace
            parts = re.split(r"[,\t ]+", s)
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            # drop obvious header rows
            if a.lower() in {"source", "src", "from"} and b.lower() in {"target", "dst", "to"}:
                continue
            edges.append((a, b))
    # de-dup
    edges = list(dict.fromkeys(edges))
    return edges


def build_adjacency(nodes: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.int8)
    kept = 0
    for a, b in edges:
        if a in idx and b in idx and a != b:
            A[idx[a], idx[b]] = 1
            kept += 1
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_npz", type=str, required=True)
    ap.add_argument("--data_directory", type=str, required=True, help="same --data_directory you used for causalbench_run")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out_npz", type=str, default="data_causalbench/exports/k562_obs_top50.npz")
    ap.add_argument("--out_meta", type=str, default="data_causalbench/exports/k562_obs_top50_meta.json")
    args = ap.parse_args()

    export_npz = Path(args.export_npz)
    data_dir = Path(args.data_directory)
    out_npz = Path(args.out_npz)
    out_meta = Path(args.out_meta)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    X, genes, intervs = load_export_npz(export_npz)
    idx50 = pick_topk_by_variance(X, genes, k=args.k)

    genes50 = [genes[i] for i in idx50]
    X50 = X[:, idx50]

    gt_file = find_best_ground_truth_file(data_dir)
    if gt_file is None:
        raise FileNotFoundError(
            f"Could not find a likely ground-truth edge file under {data_dir}. "
            f"Try printing the cache tree and we’ll target it explicitly."
        )

    edges = parse_edge_file(gt_file)
    A_ref_50 = build_adjacency(genes50, edges)

    np.savez_compressed(out_npz, X=X50, gene_names=np.asarray(genes50, dtype=object), A_ref=A_ref_50)
    meta = {
        "export_npz": str(export_npz),
        "data_directory": str(data_dir),
        "k": args.k,
        "selected_rule": "topk_by_variance",
        "gt_edge_file_used": str(gt_file),
        "N": int(X50.shape[0]),
        "D": int(X50.shape[1]),
        "num_edges_in_ref_subgraph": int(A_ref_50.sum()),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[prepare_causalbench_50] wrote: {out_npz}")
    print(f"[prepare_causalbench_50] wrote: {out_meta}")
    print(f"[prepare_causalbench_50] ref edges in subgraph: {int(A_ref_50.sum())}")


if __name__ == "__main__":
    main()


