#!/usr/bin/env python3
import argparse
import random
import pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import networkx as nx
from xu_util import get_random_instance


def worker(args):
    """
    args: (idx, min_n, max_n, save_dir, seed)
    """
    idx, min_n, max_n, save_dir, seed = args

    # reproducible RNG per-worker
    random.seed(seed + idx)
    np.random.seed(seed + idx)

    stub = f"RB_{min_n}_{max_n}_{idx}"
    out_path = save_dir / f"{stub}.gpickle"

    # sample until node-count constraint is satisfied
    while True:
        g, _ = get_random_instance(size="small")  # size argument ignored now
        g.remove_nodes_from(list(nx.isolates(g)))
        n_nodes = g.number_of_nodes()
        if min_n <= n_nodes <= max_n:
            break

    # write to disk
    with open(out_path, 'wb') as f:
        pickle.dump(g, f, pickle.HIGHEST_PROTOCOL)

    return out_path

def main():
    p = argparse.ArgumentParser(description="Parallel RB graph generator → .gpickle files")
    p.add_argument('--num-graphs', type=int,   required=True,
                   help='Number of graphs to generate')
    p.add_argument('--min-n',       type=int,   required=True,
                   help='Minimum node count per graph')
    p.add_argument('--max-n',       type=int,   required=True,
                   help='Maximum node count per graph')
    p.add_argument('--seed',        type=int,   default=0,
                   help='Base random seed')
    p.add_argument('--save-dir',    type=str,   required=True,
                   help='Directory to write .gpickle files')
    p.add_argument('--processes',   type=int,   default=cpu_count(),
                   help='Number of parallel worker processes')
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.min_n > args.max_n:
        p.error("--min-n must be ≤ --max-n")

    # build jobs list
    jobs = [
        (i, args.min_n, args.max_n, save_dir, args.seed)
        for i in range(args.num_graphs)
    ]

    print(f"Saving → {save_dir}")
    print(f"Generating {args.num_graphs} RB graphs with n∈[{args.min_n},{args.max_n}] "
          f"using {args.processes} processes…")

    with Pool(args.processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, jobs),
                      total=len(jobs),
                      desc="RB Graphs"):
            pass

    print("Done!")

if __name__ == "__main__":
    main()
