import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
import torch

import pytest

ROOT = Path(__file__).resolve().parent

def load_module(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# --- Provide minimal package shims so baseline modules import cleanly in this sandbox ---
utils_local = load_module("_utils_local", "utils/utils.py")

pkg_utils = types.ModuleType("utils")
mod_utils_utils = types.ModuleType("utils.utils")
mod_utils_utils.softmax = utils_local.softmax
mod_utils_utils.entropy_categorical = utils_local.entropy_categorical
mod_utils_utils.ess = utils_local.ess
mod_utils_utils.normalize = utils_local.normalize
pkg_utils.utils = mod_utils_utils  # type: ignore

sys.modules["utils"] = pkg_utils
sys.modules["utils.utils"] = mod_utils_utils

# dag shim
dag_local = load_module("_dag_ops_local", "dag/dag_ops.py")
pkg_dag = types.ModuleType("dag")
mod_dag_ops = types.ModuleType("dag.dag_ops")
mod_dag_ops.is_acyclic = dag_local.is_acyclic
mod_dag_ops.random_dag = dag_local.random_dag
mod_dag_ops.sample_weights = dag_local.sample_weights
mod_dag_ops.would_create_cycle = dag_local.would_create_cycle
pkg_dag.dag_ops = mod_dag_ops  # type: ignore
sys.modules["dag"] = pkg_dag
sys.modules["dag.dag_ops"] = mod_dag_ops

# likelihood shim (wire to the local baseline implementation)
pref_local = load_module("_pref_local", "likelihood/preference_likelihood_threeway.py")
pkg_lik = types.ModuleType("likelihood")
mod_pref = types.ModuleType("likelihood.preference_likelihood_threeway")
mod_pref.bt_threeway_hier = pref_local.bt_threeway_hier
mod_pref.log_expert_likelihood_bt_threeway = pref_local.log_expert_likelihood_bt_threeway
pkg_lik.preference_likelihood_threeway = mod_pref  # type: ignore
sys.modules["likelihood"] = pkg_lik
sys.modules["likelihood.preference_likelihood_threeway"] = mod_pref

# inference shim for candidate_selection imports if needed (not used here)
pkg_infer = types.ModuleType("inference")
sys.modules["inference"] = pkg_infer

generation = load_module("generation", "generation/generation.py")
generation_gpu = load_module("generation_gpu", "generation/generation_gpu.py")
ParticlePosterior_mod = load_module("ParticlePosterior", "inference/ParticlePosterior.py")
ParticlePosterior_gpu_mod = load_module("ParticlePosterior_gpu", "inference/ParticlePosterior_gpu.py")
pref_gpu = load_module("pref_gpu", "likelihood/preference_likelihood_threeway_gpu.py")

def test_screen_pairs_uncertain_matches():
    rng = np.random.default_rng(0)
    D = 12
    M = rng.random((D, D))
    np.fill_diagonal(M, 0.0)
    top_k = 25

    a = generation.screen_pairs_uncertain(M, top_k=top_k)
    b = generation_gpu.screen_pairs_uncertain(M, top_k=top_k)
    assert a == b

def test_screen_pairs_uncertain_ordered_matches():
    rng = np.random.default_rng(1)
    D = 10
    M = rng.random((D, D))
    np.fill_diagonal(M, 0.0)
    top_k = 30

    a = generation.screen_pairs_uncertain_ordered(M, top_k=top_k)
    b = generation_gpu.screen_pairs_uncertain_ordered(M, top_k=top_k)
    assert a == b

def test_edge_marginals_matches():
    rng = np.random.default_rng(2)
    S, D = 50, 8
    particles = [rng.normal(size=(D, D)) * (rng.random((D, D)) < 0.2) for _ in range(S)]
    weights = rng.random(S)
    weights = weights / weights.sum()

    post_np = ParticlePosterior_mod.ParticlePosterior(particles, weights)
    post_gpu = ParticlePosterior_gpu_mod.ParticlePosterior(particles, weights, device="cpu")

    m_np = post_np.edge_marginals()
    m_gpu = post_gpu.edge_marginals()

    np.testing.assert_allclose(m_np, m_gpu, rtol=0, atol=1e-15)


def _numpy_predictive(particles, weights, i, j, beta_edge, beta_dir, lam=0.0):
    p = np.zeros(3, dtype=float)
    for w, W in zip(weights, particles):
        p += float(w) * pref_local.bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
    return p


def _numpy_eig(particles, weights, i, j, beta_edge, beta_dir, lam=0.0):
    pred = _numpy_predictive(particles, weights, i, j, beta_edge, beta_dir, lam)
    H_pred = utils_local.entropy_categorical(pred)
    H_cond = 0.0
    for w, W in zip(weights, particles):
        p = pref_local.bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
        H_cond += float(w) * utils_local.entropy_categorical(p)
    return float(H_pred - H_cond)


def test_bt_threeway_hier_batched_matches_baseline():
    rng = np.random.default_rng(3)
    S, D = 128, 9
    particles = rng.normal(size=(S, D, D)) * (rng.random((S, D, D)) < 0.25)
    i, j = 2, 7
    beta_edge, beta_dir, lam = 4.5, 6.25, 0.1

    # Baseline per-particle
    P_np = np.stack(
        [pref_local.bt_threeway_hier(particles[s], i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam) for s in range(S)],
        axis=0,
    )

    Wt = torch.as_tensor(particles, dtype=torch.float64)
    P_t = pref_gpu.bt_threeway_hier_torch(Wt, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
    np.testing.assert_allclose(P_np, P_t.detach().cpu().numpy(), rtol=0, atol=1e-12)


def test_eig_and_update_matches_numpy_reference():
    rng = np.random.default_rng(4)
    S, D = 200, 8
    particles = [rng.normal(size=(D, D)) * (rng.random((D, D)) < 0.2) for _ in range(S)]
    weights = rng.random(S)
    weights = weights / weights.sum()

    i, j = 1, 6
    beta_edge, beta_dir, lam = 5.0, 5.0, 0.0

    import torch
    post_gpu = ParticlePosterior_gpu_mod.ParticlePosterior(particles, weights, device="cpu", dtype=torch.float64)

    eig_np = _numpy_eig(particles, weights, i, j, beta_edge, beta_dir, lam)
    eig_gpu = post_gpu.eig_for_pair(i, j, beta_edge, beta_dir, lam)
    assert abs(eig_np - eig_gpu) < 1e-10

    # Update weights for a fixed response
    y = 0
    # numpy reference update
    new_w = np.empty_like(weights)
    for s, W in enumerate(particles):
        p = pref_local.bt_threeway_hier(W, i, j, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam)
        new_w[s] = weights[s] * p[y]
    new_w = utils_local.normalize(new_w)

    post_gpu.update_with_observation(i, j, y, beta_edge, beta_dir, lam)
    w_gpu = post_gpu.weights.detach().cpu().numpy()
    np.testing.assert_allclose(new_w, w_gpu, rtol=0, atol=1e-12)


def test_eig_for_pairs_matches_looping():
    rng = np.random.default_rng(5)
    S, D = 150, 10
    particles = [rng.normal(size=(D, D)) * (rng.random((D, D)) < 0.25) for _ in range(S)]
    weights = rng.random(S)
    weights = weights / weights.sum()

    beta_edge, beta_dir, lam = 4.0, 6.0, 0.05
    pairs = np.array([(0, 1), (2, 7), (3, 9), (5, 6), (8, 4), (1, 9), (7, 0)], dtype=np.int64)

    import torch
    post_gpu = ParticlePosterior_gpu_mod.ParticlePosterior(particles, weights, device="cpu", dtype=torch.float64)

    eig_loop = np.array([
        post_gpu.eig_for_pair(int(i), int(j), beta_edge, beta_dir, lam)
        for i, j in pairs
    ])
    eig_batch = post_gpu.eig_for_pairs(pairs, beta_edge=beta_edge, beta_dir=beta_dir, lam=lam, chunk_size=3)
    np.testing.assert_allclose(eig_loop, eig_batch, rtol=0, atol=1e-12)
