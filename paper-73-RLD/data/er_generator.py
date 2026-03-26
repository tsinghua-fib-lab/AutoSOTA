#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import networkx as nx
from tqdm import tqdm

class ErdosRenyi:
    """Simple ER graph sampler."""
    def __init__(self, min_n: int, max_n: int, p: float):
        self.min_n = min_n
        self.max_n = max_n
        self.p = p

    def __str__(self):
        return f"ER_{self.min_n}_{self.max_n}_{self.p}"

    def generate_graph(self):
        n = random.randint(self.min_n, self.max_n)
        return nx.erdos_renyi_graph(n, self.p)

def _worker(idx: int, sampler: ErdosRenyi, output_dir: Path):
    """
    Generate one ER graph and write it to disk.
    """
    G = sampler.generate_graph()
    stub = f"{sampler}_{idx}"
    out_path = output_dir / f"{stub}.gpickle"
    # write as gpickle
    nx.write_gpickle(G, out_path)
    return out_path

def main():
    p = argparse.ArgumentParser(
        description="Parallel ER graph generator → .gpickle files"
    )
    p.add_argument("--min-n", type=int, required=True, help="Minimum number of nodes")
    p.add_argument("--max-n", type=int, required=True, help="Maximum number of nodes")
    p.add_argument("--p", type=float, default=0.15, help="Edge probability")
    p.add_argument(
        "--num-graphs", type=int, default=1, help="How many graphs to generate"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write .gpickle files",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel worker processes",
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampler = ErdosRenyi(args.min_n, args.max_n, args.p)
    worker_fn = partial(_worker, sampler=sampler, output_dir=args.output_dir)

    indices = list(range(args.num_graphs))
    with Pool(processes=args.processes) as pool:
        # imap_unordered + tqdm to get a live progress bar
        for _ in tqdm(pool.imap_unordered(worker_fn, indices), total=len(indices)):
            pass

    print(f"Done! {args.num_graphs} ER graphs written to {args.output_dir}")

if __name__ == "__main__":
    main()
