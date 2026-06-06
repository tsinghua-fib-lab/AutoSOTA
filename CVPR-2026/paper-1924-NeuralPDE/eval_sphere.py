#!/usr/bin/env python3
"""eval_sphere.py - Evaluate SurfNO Poisson solver on a sphere.

Solves the Poisson equation Δu = sin(ωx)sin(ωy)sin(ωz) on the unit sphere
using the SurfNO neural operator with residual mixing of the extension weights.

Metrics: nmae, nmaxe (normalized by solution range, lower is better).
Usage: export PYTHONPATH=/repo/src && python /repo/eval_sphere.py
"""

import numpy as np
import torch
import time
import sys
import os
import json

from scipy.spatial import KDTree
from model.SurfNO import SurfNO_weights_only
from utils.Laplacian_matrix import Laplacian_matrix
from utils.retrieve_neural_weights import retrieve_neural_weights
from scipy.sparse.linalg import spsolve
from utils.RBF_update import precompute_rbf_data, interpolate_from_precomputed
from utils.rot_update import function_update, turn_into_dict
from utils.rot_update import build_Global_dico, final_updated_function_value
from utils.define_band_points import define_band_points


RESIDUAL_ALPHA = float(os.environ.get("SPHERE_RESIDUAL_ALPHA", "1.0"))


def fibonacci_sphere(n_points):
    indices = np.arange(n_points)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def build_sphere_surface_features(surface_points):
    n = surface_points.shape[0]
    features = np.zeros((n, 12), dtype=np.float32)
    features[:, :3] = surface_points
    features[:, 3:6] = surface_points
    x, y, z = surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]
    eps = 1e-8
    r_xy = np.sqrt(x**2 + y**2)
    r_xy = np.maximum(r_xy, eps)
    e_theta = np.stack([x * z / r_xy, y * z / r_xy, -r_xy], axis=1)
    e_theta /= np.maximum(np.linalg.norm(e_theta, axis=1, keepdims=True), eps)
    features[:, 6:9] = e_theta
    e_phi = np.stack([-y, x, np.zeros_like(x)], axis=1)
    e_phi /= np.maximum(np.linalg.norm(e_phi, axis=1, keepdims=True), eps)
    features[:, 9:12] = e_phi
    return features


def rot_update_residual(neural_weights, u, all_local_band_indexes, all_distances_to_central,
                        temperature=0.0423, alpha=1.0):
    """Rotation-ensemble update with residual mixing of neural weight extension."""
    values = []
    for rot_num in range(len(neural_weights)):
        update = function_update(
            neural_weights[rot_num], u, all_local_band_indexes,
            all_distances_to_central, temperature=temperature
        )
        values.append(update)
    values = np.stack(values, axis=0)
    mean_all = values.mean(axis=0)

    min_vals = values.min(axis=0)
    max_vals = values.max(axis=0)
    sum_vals = values.sum(axis=0) - min_vals - max_vals
    mean_trimmed = sum_vals / max(values.shape[0] - 2, 1)

    return mean_all, mean_trimmed


def main():
    n_surface = int(os.environ.get("SPHERE_N_SURFACE", "10000"))
    delta_x = float(os.environ.get("SPHERE_DELTA_X", "0.05"))
    dist_to_surface = float(os.environ.get("SPHERE_DIST", "0.2"))
    local_size = int(os.environ.get("SPHERE_LOCAL_SIZE", "400"))
    omega = float(os.environ.get("SPHERE_OMEGA", "10"))
    rbf_k = int(os.environ.get("SPHERE_RBF_K", "8"))
    rbf_epsilon = float(os.environ.get("SPHERE_RBF_EPSILON", "1.0"))

    t0 = time.time()

    surface_points = fibonacci_sphere(n_surface)
    surface_features = build_sphere_surface_features(surface_points)
    tree_surface = KDTree(surface_points)

    threshold = 0.8 * dist_to_surface
    band_points, mask_threshold = define_band_points(
        delta_x, surface_points, tree_surface, dist_to_surface, threshold
    )
    tree_band = KDTree(band_points)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = os.path.join(os.path.dirname(__file__), "src", "model", "weights", "SurfNO_pretrained_weights.pth")
    model = SurfNO_weights_only()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        neural_weights, all_distances_to_central, all_local_band_indexes = retrieve_neural_weights(
            surface_points, band_points, local_size, model, tree_band, mask_threshold,
            surface_feature_TP=surface_features
        )

    Lap, denom = Laplacian_matrix(band_points, tree_band, delta_x)

    lhs_function = (
        np.sin(omega * band_points[:, 0])
        * np.sin(omega * band_points[:, 1])
        * np.sin(omega * band_points[:, 2])
    )
    lhs_function -= np.mean(lhs_function)

    # Apply neural weight extension
    lhs_nn, _ = rot_update_residual(
        neural_weights, lhs_function, all_local_band_indexes,
        all_distances_to_central, alpha=RESIDUAL_ALPHA
    )

    # Residual mixing
    if RESIDUAL_ALPHA < 1.0:
        lhs_function = (1.0 - RESIDUAL_ALPHA) * lhs_function + RESIDUAL_ALPHA * lhs_nn
    else:
        lhs_function = lhs_nn

    rhs = lhs_function * denom
    U = spsolve(Lap, rhs)
    U -= np.mean(U)

    neighbors_indices, factors, phi_vecs = precompute_rbf_data(
        band_points, tree_band, surface_points, k=rbf_k, epsilon=rbf_epsilon
    )
    solution = interpolate_from_precomputed(U, neighbors_indices, factors, phi_vecs, clipping=True)

    # Analytic ground truth for Δu = sin(ωx)sin(ωy)sin(ωz):
    # Δ(sin(ωx)sin(ωy)sin(ωz)) = -3ω² sin(ωx)sin(ωy)sin(ωz)
    # So u = -sin(ωx)sin(ωy)sin(ωz) / (3ω²)
    ground_truth = (
        -np.sin(omega * surface_points[:, 0])
        * np.sin(omega * surface_points[:, 1])
        * np.sin(omega * surface_points[:, 2])
        / (3.0 * omega * omega)
    )

    solution -= np.mean(solution)
    ground_truth -= np.mean(ground_truth)

    error = solution - ground_truth
    solution_range = float(np.max(ground_truth) - np.min(ground_truth))
    solution_max_abs = float(np.max(np.abs(ground_truth)))

    if solution_range > 1e-12:
        nmae = float(np.mean(np.abs(error)) / solution_range)
        nmaxe = float(np.max(np.abs(error)) / solution_range)
    else:
        nmae = float(np.mean(np.abs(error)))
        nmaxe = float(np.max(np.abs(error)))

    elapsed = time.time() - t0

    print(f"nmae: {nmae:.6e}")
    print(f"nmaxe: {nmaxe:.6e}")
    print(f"time_s: {elapsed:.2f}")
    print(f"solution_range: {solution_range:.6e}")
    print(f"solution_max_abs: {solution_max_abs:.6e}")
    print(f"n_surface_points: {n_surface}")
    print(f"n_band_points: {band_points.shape[0]}")

    try:
        results = {
            "nmae": nmae, "nmaxe": nmaxe, "time_s": elapsed,
            "solution_range": solution_range, "solution_max_abs": solution_max_abs,
            "n_surface_points": n_surface, "n_band_points": int(band_points.shape[0]),
            "residual_alpha": RESIDUAL_ALPHA,
        }
        with open("/tmp/eval_results.json", "w") as f:
            json.dump(results, f)
    except Exception:
        pass


if __name__ == "__main__":
    main()
