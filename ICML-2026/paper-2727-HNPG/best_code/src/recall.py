import os
from typing import Any
os.environ["GEOMSTATS_BACKEND"] = "numpy"

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

# from preprocess import *
# from utils import *

from geomstats.learning.frechet_mean import FrechetMean, GradientDescent


def weighted_karcher_flow(H, points, weights, max_iter=64, tol=1e-6, adaptive=False):

    method = "adaptive" if adaptive else "default"
    fm = FrechetMean(space=H, method=method)
    if not adaptive:
        fm.optimizer = GradientDescent(max_iter=max_iter, epsilon=tol)
    fm.fit(points, weights=weights)
    return fm.estimate_


def hyperboloid_pairwise_inner(queries, memory):
    """
    Pairwise Lorentz inner product on the unit hyperboloid.
    queries: (M_q, d+1), memory: (M_m, d+1), all on the manifold.
    Returns (M_q, M_m).
    """
    mem_signed = memory.copy()
    mem_signed[:, 0] *= -1
    inners = queries @ mem_signed.T
    return inners


def hyperboloid_pairwise_dist(queries, memory):
    """
    Pairwise geodesic distance on the unit hyperboloid.
    queries: (M_q, d+1), memory: (M_m, d+1), all on the manifold.
    Returns (M_q, M_m).
    """
    mem_signed = memory.copy()
    mem_signed[:, 0] *= -1
    inners = queries @ mem_signed.T

    arg = np.clip(-inners, 1.0 + 1e-12, None)
    return np.arccosh(arg)


def kfm(H, memory, query, steps, beta=1.0):

    weights = hyperboloid_pairwise_inner(query, memory)
    weights = softmax(beta * weights, axis=-1)

    errors = []

    cor = 0
    tol = 0.01

    for i in tqdm(range(len(weights)), desc="kfm recall", leave=False):

        recall = weighted_karcher_flow(H, memory, weights[i], max_iter=steps)
        errors.append(H.metric.dist(recall, memory[i]))

        if H.metric.dist(recall, memory[i]) <= tol:
            cor += 1

    return cor


def dam(memory, query, steps, order=10, beta=1.0):
    cor = 0
    tol = 0.01
    for i in tqdm(range(len(query)), desc="dam recall", leave=False):
        q = query[i].copy()
        for _ in range(steps):

            sim = memory @ q
            score = np.maximum(sim, 0) ** order
            w = score / (score.sum() + 1e-9)
            q = w @ memory

        if np.linalg.norm(q - memory[i]) <= tol:
            cor += 1

    return cor


def mhn(memory, query, steps, beta=1.0):
    cor = 0
    tol = 0.01
    for i in tqdm(range(len(query)), desc="mhn recall", leave=False):
        q = query[i].copy()
        for _ in range(steps):
            w = softmax(beta * (memory @ q), axis=0)
            q = w @ memory

        if np.linalg.norm(q - memory[i]) <= tol:
            cor += 1

    return cor

