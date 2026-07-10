import os
os.environ["GEOMSTATS_BACKEND"] = "numpy"

import argparse

import numpy as np
from tqdm import tqdm

from memory.data import load_images
from memory.preprocess import (
    image_pca_and_rescale,
    to_hyperboloid,
    add_noise,
    add_noise_euclidean,
)
from memory.utils import log_result, save_result
from memory.recall import kfm, dam, mhn
from memory.hyperboloid import HyperboloidKappa


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M-min", type=int, default=10)
    ap.add_argument("--M-max", type=int, default=1000)
    ap.add_argument("--pca-dim", type=int, default=10)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "synthetic"],
    )
    ap.add_argument("--mem-R", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--noise_sigma", type=float, default=0.3)
    return ap.parse_args()


def generate_M_values(M_min: int, M_max: int, K: int = 15) -> np.ndarray:
    r = (M_max / M_min) ** (1 / K)
    return np.round(M_min * r ** np.arange(K + 1)).astype(int)


def run_one_trial(args, M: int):

    rng = np.random.default_rng()

    X = load_images(args.dataset, M, rng, dim=args.pca_dim, R=args.mem_R)
    H = HyperboloidKappa(dim=args.pca_dim, curvature=-1)

    X_red = image_pca_and_rescale(X, args.pca_dim, args.mem_R)
    points = to_hyperboloid(H, X_red)
    query_on_manifold = add_noise(H, args.noise_sigma, points, rng)
    query_euclidean = add_noise_euclidean(args.noise_sigma, X_red, rng)

    steps = 64
    cor_kfm = kfm(H, points, query_on_manifold, steps, beta=args.beta)
    cor_mhn = mhn(X_red, query_euclidean, steps, beta=args.beta)
    cor_dam = dam(X_red, query_euclidean, steps, beta=args.beta)
    tqdm.write(
        f"M={M} kfm: {cor_kfm}/{M} mhn: {cor_mhn}/{M} dam: {cor_dam}/{M}"
    )
    return cor_kfm, cor_mhn, cor_dam


def main():
    args = parse_args()

    if args.M_min <= 0:
        raise ValueError(f"--M-min must be positive, got {args.M_min}")
    if args.M_max < args.M_min:
        raise ValueError(
            f"--M-max ({args.M_max}) must be >= --M-min ({args.M_min})"
        )

    M_values = generate_M_values(args.M_min, args.M_max)
    tqdm.write(f"M values: {M_values}")

    results = {
        "recall rate":[],
        "model":[],
        "M":[]
    }

    for _ in range(args.n_runs):

        for M in tqdm(M_values, desc="capacity sweep"):
            cor_kfm, cor_mhn, cor_dam = run_one_trial(args, int(M))
            results = log_result(results, M, cor_kfm, cor_mhn, cor_dam)

    save_result(results, args)

if __name__ == "__main__":
    main()
