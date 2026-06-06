#!/usr/bin/env python3
import argparse
import random
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

import networkx as nx
from tqdm import tqdm

def worker(args):
    """
    args: (idx, min_n, max_n, m, save_dir, seed)
    """
    idx, min_n, max_n, m, save_dir, seed = args

    # reproducible per‐worker RNG
    random.seed(seed + idx)
    np.random.seed(seed + idx)

    # pick n such that n > m
    n = random.randint(max(min_n, m+1), max_n)
    G = nx.barabasi_albert_graph(n, m)

    # filename stub
    stub = f"BA_{min_n}_{max_n}_{m}_{idx}"
    out_path = save_dir / f"{stub}.gpickle"

    # write
    nx.write_gpickle(G, out_path)
    return out_path

def main():
    p = argparse.ArgumentParser(
        description="Parallel BA graph generator → .gpickle files"
    )
    p.add_argument("--min-n",    type=int, required=True,  help="Minimum number of nodes")
    p.add_argument("--max-n",    type=int, required=True,  help="Maximum number of nodes")
    p.add_argument("--m",        type=int, default=4,  help="Number of edges to attach per new node")
    p.add_argument("--num-graphs", type=int, required=True, help="Total graphs to generate")
    p.add_argument("--save-dir", type=Path, required=True,  help="Directory to write .gpickle files")
    p.add_argument("--seed",     type=int, default=0,        help="Base random seed")
    p.add_argument("--processes",type=int, default=cpu_count(),
                   help="Parallel worker count")
    args = p.parse_args()

    # prepare output directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # build job list
    jobs = [
        (idx, args.min_n, args.max_n, args.m, args.save_dir, args.seed)
        for idx in range(args.num_graphs)
    ]

    print(f"Saving → {args.save_dir}")
    print(f"Generating {args.num_graphs} BA graphs (n∈[{args.min_n},{args.max_n}], m={args.m}) "
          f"using {args.processes} processes…")

    with Pool(args.processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, jobs),
                      total=len(jobs),
                      desc="BA Graphs"):
            pass

    print("Done!")

if __name__ == "__main__":
    main()
