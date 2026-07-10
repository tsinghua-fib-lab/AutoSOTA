#!/usr/bin/env python3
"""Generate synthetic Noncircle graphs used in the PSAHS paper (3-class SBM + elliptical features)."""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from psahs.synthetic.noncircle import (
    create_pyg_graph_with_attributes_three_class,
    generate_source_graph_three_class,
    generate_target_graph_from_source_three_class,
    load_graph_core_three_class,
)


def main():
    parser = argparse.ArgumentParser(description="Generate Noncircle synthetic graphs.")
    parser.add_argument("--num_nodes", type=int, default=4000)
    parser.add_argument("--target_homophily", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "dataset", "noncircle"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    source_file = os.path.join(args.out_dir, f"source_graph_ICLR_{args.num_nodes}.pkl")
    target_dir = os.path.join(args.out_dir, f"ICLR_target_graphs_{args.num_nodes}")
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, f"target_graph_h{args.target_homophily:.1f}.pkl")

    if not os.path.exists(source_file):
        print(f"Generating source graph -> {source_file}")
        generate_source_graph_three_class(
            args.num_nodes,
            SIGMA=0.5,
            p=0.2,
            q=0.02,
            save_path=source_file,
            seed=args.seed,
        )
    else:
        print(f"Source graph exists: {source_file}")

    G, features, label, c0, c1, c2 = load_graph_core_three_class(source_file)
    print(f"Generating target graph (H={args.target_homophily}) -> {target_file}")
    generate_target_graph_from_source_three_class(
        (G, features, label, c0, c1, c2),
        args.target_homophily,
        seed=args.seed * 2,
        save_path=target_file,
    )
    graph = create_pyg_graph_with_attributes_three_class(G, features, label, c0, c1, c2)
    print(
        f"Done. nodes={graph.num_nodes}, edges={graph.num_edges}, "
        f"homophily={graph.homophily_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
