###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
from abc import ABC, abstractmethod
import pickle
from typing import Any, Dict, List, Literal, Union
from typing_extensions import override
import graph_tool.all as gt

##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import signal
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

import pygsp as pg
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh, sqrtm
from scipy.stats import chi2
from src.analysis.dist_helper import (
    compute_mmd,
    gaussian_emd,
    gaussian,
    emd,
    gaussian_tv,
)
from torch_geometric.utils import to_networkx

from src.metrics.abstract_metrics import compute_ratios
from src.datasets.tls_dataset import CellGraph
from src.metrics.tls_metrics import eval_fraction_novel_cell_graphs, eval_fraction_unique_novel_valid_cell_graphs
PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
    "eval_acc_lobster_graph",
]


# Define a timeout handler
def handler(signum, frame):
    raise TimeoutError


# Set the signal handler for the alarm
signal.signal(signal.SIGALRM, handler)


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


def degree_stats_rbf(graph_ref_list, graph_pred_list, is_parallel=True):
    """RBF MMD on degree distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)
    return mmd_dist


###############################################################################


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(
        np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False
    )
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigval_stats(
    eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_ref_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_pred_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing eig mmd: ", elapsed)
    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_pred_list))],
                [bound for i in range(len(eigval_pred_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_pred_list_remove_empty],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


def clustering_stats_rbf(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    """RBF MMD on clustering coefficient distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian, sigma=1.0 / 10)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()
    output = sp.check_output(
        [
            str(os.path.join(os.path.dirname(os.path.realpath(__file__)), "orca/orca")),
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def _get_orbit_counts(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for orbit_counts in executor.map(orca, graph_ref_list):
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / (
                orbit_counts.shape[0] if orbit_counts.shape[0] > 0 else 1
            )
            total_counts_ref.append(orbit_counts_graph)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for orbit_counts in executor.map(orca, graph_pred_list_remove_empty):
            orbit_counts_graph = np.sum(orbit_counts, axis=0) / (
                orbit_counts.shape[0] if orbit_counts.shape[0] > 0 else 1
            )
            total_counts_pred.append(orbit_counts_graph)

    return np.array(total_counts_ref), np.array(total_counts_pred)


def motif_stats(
    graph_ref_list,
    graph_pred_list,
    motif_type="4cycle",
    ground_truth_match=None,
    bins=100,
    compute_emd=False,
):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref, total_counts_pred = _get_orbit_counts(
        graph_ref_list, graph_pred_list
    )

    # mmd_dist = compute_mmd(
    #     total_counts_ref,
    #     total_counts_pred,
    #     kernel=gaussian,
    #     is_hist=False,
    #     sigma=30.0)

    # mmd_dist = compute_mmd(
    #         total_counts_ref,
    #         total_counts_pred,
    #         kernel=gaussian_tv,
    #         is_hist=False,
    #         sigma=30.0)

    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian,
            is_hist=False,
            sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    return mmd_dist


def orbit_stats_rbf(graph_ref_list, graph_pred_list):
    """RBF MMD on orbit counts."""
    total_counts_ref, total_counts_pred = _get_orbit_counts(
        graph_ref_list, graph_pred_list
    )
    mmd_dist = compute_mmd(
        total_counts_ref,
        total_counts_pred,
        kernel=gaussian,
        is_hist=False,
        sigma=30.0,
    )
    return mmd_dist


def fid_score(real_features, fake_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_fake, sigma_fake = np.mean(fake_features, axis=0), np.cov(
        fake_features, rowvar=False
    )

    # Epsilon to prevent singular covariance matrices
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    diff = mu_real - mu_fake
    # Matrix square root of C_real * C_fake
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = sqrtm((sigma_real + offset).dot(sigma_fake + offset))

    # Numerical stability for complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    return fid


def orbit_stats_fid(graph_ref_list, graph_pred_list):
    """FID on orbit counts."""
    total_counts_ref, total_counts_pred = _get_orbit_counts(
        graph_ref_list, graph_pred_list
    )
    return fid_score(total_counts_ref, total_counts_pred)


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    refinement_steps=100,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
                [refinement_steps for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
            )
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = f"data/all_grids.pt"
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if nx.is_isomorphic(G, G_grid):
                    return True
        return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (
            (node_counts > 40).sum() > 0
            or (node_counts < 20).sum() > 0
            or n_blocks > 5
            or n_blocks < 2
        ):
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            try:
                # Set the alarm for 60 seconds
                signal.alarm(60)
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
            except TimeoutError:
                print("Timeout: Skipping this iteration")
                continue
            finally:
                # Disable the alarm
                signal.alarm(0)
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid

######################################################
# Refactored Metric Calculation Framework
######################################################

class MetricCalculator(ABC):
    """
    Abstract base class defining a unified interface for all graph metric calculators.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric, e.g. 'degree', 'clustering'."""
        pass

    @abstractmethod
    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Union[Dict[str, Any], Any]:
        """
        Computes and returns metric results.
        The metric result can return a single value or a dictionary.
        If the metric result is a single value, the calculate function wraps it as a dictionary.
        If the metric result is a dictionary, the calculate function returns that dictionary directly.
        :param generated_graphs: List of generated graphs.
        :param reference_graphs: List of reference (real) graphs.
        :return: A dictionary containing one or more metric results.
        """
        pass

    def calculate(self, generated_graphs: List[nx.Graph], 
                  reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        result = self._calculate(generated_graphs, reference_graphs)
        if isinstance(result, dict):
            return result
        else:
            return {self.name: result}


class DegreeMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "degree"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing degree stats...")
        emd_distance = degree_stats(
            reference_graphs,
            generated_graphs,
            is_parallel=True,
            compute_emd=self._compute_emd,
        )
        return emd_distance


class RBFMMDDegreeMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "rbf_mmd_degree"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing RBF MMD on degree stats..")
        rbf_mmd_degree = degree_stats_rbf(
            reference_graphs,
            generated_graphs,
            is_parallel=True,
        )
        return rbf_mmd_degree


class WaveletMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "wavelet"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing wavelet stats...")
        ref_eigvals, ref_eigvecs = compute_list_eigh(reference_graphs)
        pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(generated_graphs)
        wavelet = spectral_filter_stats(
            eigvec_ref_list=ref_eigvecs,
            eigval_ref_list=ref_eigvals,
            eigvec_pred_list=pred_graph_eigvecs,
            eigval_pred_list=pred_graph_eigvals,
            is_parallel=False,
            compute_emd=self._compute_emd,
        )
        return wavelet


class ClusteringMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "clustering"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing clustering stats...")
        clustering = clustering_stats(
            reference_graphs,
            generated_graphs,
            bins=100,
            is_parallel=True,
            compute_emd=self._compute_emd,
        )
        return clustering

class SpectreMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "spectre"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing spectre stats...")
        spectre = spectral_stats(
            reference_graphs,
            generated_graphs,
            is_parallel=True,
            n_eigvals=-1,
            compute_emd=self._compute_emd,
        )
        return spectre


class RBFMMDClusteringMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "rbf_mmd_clustering"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing RBF MMD on clustering stats..")
        rbf_mmd_clustering = clustering_stats_rbf(
                reference_graphs,
                generated_graphs,
                bins=100,
                is_parallel=True,
        )
        return rbf_mmd_clustering


class MotifMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "motif"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing motif stats")
        motif = motif_stats(
            reference_graphs,
            generated_graphs,
            motif_type="4cycle",
            ground_truth_match=None,
            bins=100,
            compute_emd=self._compute_emd,
        )
        return motif


class OrbitMetric(MetricCalculator):
    def __init__(self, compute_emd: bool = False):
        self._compute_emd = compute_emd

    @property
    def name(self) -> str:
        return "orbit"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing orbit stats...")
        orbit = orbit_stats_all(
            reference_graphs,
            generated_graphs,
            compute_emd=self._compute_emd,
        )
        return orbit


class RBFMMDOrbitMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "rbf_mmd_orbit"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing RBF MMD on orbit stats..")
        rbf_mmd_orbit = orbit_stats_rbf(reference_graphs, generated_graphs)
        return rbf_mmd_orbit


class FIDOrbitMetric(MetricCalculator):

    @property
    def name(self) -> str:
        return "fid_orbit"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing FID on orbit stats..")
        fid_orbit = orbit_stats_fid(reference_graphs, generated_graphs)
        return fid_orbit

class SBMMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "sbm"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing SBM stats..")
        sbm = eval_acc_sbm_graph(generated_graphs, refinement_steps=100, strict=True)
        return sbm


class PlanarMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "planar"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing Planar stats..")
        planar = eval_acc_planar_graph(generated_graphs)
        return planar

class TreeMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "tree"

    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        print("Computing Tree stats..")
        tree = eval_acc_tree_graph(generated_graphs)
        return tree


class TLSMetric(MetricCalculator):
    @property
    def name(self) -> str:
        return "tls"
    
    def __init__(self, datamodule):
        super().__init__()
        self.cell_graphs = {}
        self.cell_graphs["train"] = self._loader_to_cell_graphs(datamodule.train_dataloader())
        self.cell_graphs["val"] = self._loader_to_cell_graphs(datamodule.val_dataloader())
        self.cell_graphs["test"] = self._loader_to_cell_graphs(datamodule.test_dataloader())

    def _loader_to_cell_graphs(self, loader):
        cell_graphs = []
        for batch in loader:
            for tg_graph in batch.to_data_list():
                cell_graph = CellGraph.from_torch_geometric(tg_graph)
                cell_graphs.append(cell_graph)

        return cell_graphs

    def is_cell_graph_valid(self, cg: CellGraph):
        return is_planar_graph(cg)

    @override
    def calculate(self, raw_generated_graphs: List[nx.Graph], 
                    labels=None) -> Dict[str, Any]:
            result = self._calculate(raw_generated_graphs, labels)
            if isinstance(result, dict):
                return result
            else:
                return {self.name: result}

    def _calculate(self, raw_generated_graphs: List[nx.Graph], 
                   labels=None) -> Dict[str, Any]:

        generated_cgs = []
        for graph in raw_generated_graphs:
            generated_cgs.append(CellGraph.from_dense_graph(graph))

        # TODO: Implement these metrics with torchmetrics for parallelization
        generated_labels = torch.tensor([cg.to_label() for cg in generated_cgs])
        ambiguous_gen_cgs = sum(
            [(cg_label == -1).int() for cg_label in generated_labels]
        ).item()
        if labels is not None:
            true_labels = torch.tensor(labels)
            high_tls_idxs = true_labels == 1
            low_tls_idxs = true_labels == 0
            total_tls_acc = (generated_labels == true_labels).float().mean().item()
            high_tls_acc = (
                (generated_labels[high_tls_idxs] == true_labels[high_tls_idxs])
                .float()
                .mean()
                .item()
            )
            low_tls_acc = (
                (generated_labels[low_tls_idxs] == true_labels[low_tls_idxs])
                .float()
                .mean()
                .item()
            )
        else:
            total_tls_acc = -1
            high_tls_acc = -1
            low_tls_acc = -1


        print("Computing uniqueness, novelty and validity for cell graphs...")
        frac_novel = eval_fraction_novel_cell_graphs(
            generated_cell_graphs=generated_cgs,
            train_cell_graphs=self.cell_graphs["train"],
        )
        (
            frac_unique,
            frac_unique_and_novel,
            frac_unique_and_novel_valid,
        ) = eval_fraction_unique_novel_valid_cell_graphs(
            generated_cell_graphs=generated_cgs,
            train_cell_graphs=self.cell_graphs["train"],
            valid_cg_fn=self.is_cell_graph_valid,
        )

        tls_to_log = {
            f"tls_metrics/total_tls_acc": total_tls_acc,
            f"tls_metrics/high_tls_acc": high_tls_acc,
            f"tls_metrics/low_tls_acc": low_tls_acc,
            f"tls_metrics/num_ambiguous_tls": ambiguous_gen_cgs,
            f"tls_metrics/frac_novel": frac_novel,
            f"tls_metrics/frac_unique": frac_unique,
            f"tls_metrics/frac_unique_and_novel": frac_unique_and_novel,
            f"tls_metrics/frac_unique_and_novel_valid": frac_unique_and_novel_valid,
        }
        return tls_to_log

class FractionMetric(MetricCalculator):
    _validity_func_dict = {
        "sbm": is_sbm_graph,
        "planar": is_planar_graph,
        "tree": nx.is_tree,
    }
    def __init__(self, reference_graphs, dataset_name):
        self.reference_graphs = reference_graphs
        self.validity_func = self._validity_func_dict[dataset_name]

    @property
    def name(self) -> str:
        return "fraction"
    
    def _calculate(self, generated_graphs: List[nx.Graph], 
                   reference_graphs: List[nx.Graph]) -> Dict[str, Any]:
        # if (
        #     "sbm" in self.metrics_list
        #     or "planar" in self.metrics_list
        #     or "tree" in self.metrics_list
        # ):
        """
        Note: reference_graphs is train_graphs
        So the reference_graphs passed as parameter is not used; use internal self.reference_graphs instead
        """
        print("Computing all fractions...")
        validity_func = self.validity_func
        (
            frac_unique,
            frac_unique_non_isomorphic,
            fraction_unique_non_isomorphic_valid,
        ) = eval_fraction_unique_non_isomorphic_valid(
            generated_graphs,
            self.reference_graphs,
            validity_func,
        )
        frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
            generated_graphs, self.reference_graphs
        )
        result = {}
        result.update(
            {
                "sampling/frac_unique": frac_unique,
                "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                "sampling/frac_non_iso": frac_non_isomorphic,
            }
        )
        return result


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, datamodule, compute_emd, metrics_list):
        super().__init__()
        self.reference_graph = {}
        self.reference_graph["train"] = self._loader_to_nx(datamodule.train_dataloader())
        self.reference_graph["val"] = self._loader_to_nx(datamodule.val_dataloader())
        self.reference_graph["test"] = self._loader_to_nx(datamodule.test_dataloader())
        # self.nums_graphs_in_test = len(self.reference_graph["test"])
        # self.nums_graphs_in_val = len(self.reference_graph["val"])
        self.compute_emd = compute_emd
        self.metrics_list = ["ratio"]  # must include ratio and fraction
        self.metrics_list.extend(metrics_list)
        self.metrics_calculator = self._init_metrics_calculator(datamodule)
        self.ref_metrics = self._init_ref_metrics(datamodule)

    def _init_metrics_calculator(self, datamodule) -> Dict[str, MetricCalculator]:
        metrics_with_emd = {
            "degree": DegreeMetric,
            "wavelet": WaveletMetric,
            "spectre": SpectreMetric,
            "motif": MotifMetric,
            "orbit": OrbitMetric,
            "clustering": ClusteringMetric,
        }

        metrics_without_emd = {
            "rbf_mmd_degree": RBFMMDDegreeMetric,
            "rbf_mmd_clustering": RBFMMDClusteringMetric,
            "rbf_mmd_orbit": RBFMMDOrbitMetric,
            "fid_orbit": FIDOrbitMetric,
            "sbm": SBMMetric,
            "planar": PlanarMetric,
            "tree": TreeMetric,
        }

        metrics_calculator = {}

        for name, metric_cls in metrics_with_emd.items():
            metrics_calculator[name] = metric_cls(
                compute_emd=self.compute_emd
            )

        for name, metric_cls in metrics_without_emd.items():
            metrics_calculator[name] = metric_cls()

        dataset_names = ["sbm", "planar", "tree"]
        for dataset_name in dataset_names:
            if dataset_name in self.metrics_list:
                metrics_calculator["fraction"] = FractionMetric(
                    self.reference_graph["train"], dataset_name
                )
                break

        if 'tls' in self.metrics_list:
            metrics_calculator["tls"] = TLSMetric(datamodule)
        return metrics_calculator

    def _loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
        return networkx_graphs

    def forward(
        self,
        generated_graphs: list,
        split: Literal["train", "val", "test"],
        labels=None
    ) -> Dict[str, Any]:
        networkx_graphs_of_generated_graphs = self._change_to_networkx_graphs(
            generated_graphs
        )
        reference_graphs = self.reference_graph[split]
        metircs_of_generated_graphs = {}
        metircs_of_generated_graphs.update(
            self._compute_metrics(networkx_graphs_of_generated_graphs, reference_graphs)
            )
        if "ratio" in self.metrics_list:
            metircs_of_generated_graphs.update(
                self._compute_ratio(
                    metircs_of_generated_graphs,
                    split=split
                    )
                )
        if "tls" in self.metrics_list:
            metircs_of_generated_graphs.update(
                self._compute_tls(generated_graphs, labels)
            )
        return metircs_of_generated_graphs

    def _compute_tls(self, raw_generated_graphs: list, labels: list):
        result = {}
        result.update(
            self.metrics_calculator["tls"].calculate(raw_generated_graphs, labels)
        )
        return result

    def _compute_metrics(self, networkx_graphs: list, reference_graphs: list):
        result = {}
        metrics_to_run = [m for m in self.metrics_list if m != "ratio" and m != "tls"]
        for metric_name in metrics_to_run:
            result.update(
                self.metrics_calculator[metric_name].calculate(
                    networkx_graphs, reference_graphs)
                )
        return result

    def _init_ref_metrics(self, datamodule):
        def save_pickle(array, path):
            with open(path, "wb") as f:
                pickle.dump(array, f)

        def load_pickle(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        def get_ref_metrics_path(datamodule):
            ref_metrics_path = os.path.join(
                datamodule.train_dataloader().dataset.root, "ref_metrics_new.pkl"
            )
            if hasattr(datamodule, "remove_h"):
                if datamodule.remove_h:
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_no_h.pkl")
                else:
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_h.pkl")
            return ref_metrics_path

        ref_metrics_path = get_ref_metrics_path(datamodule)
        if os.path.exists(ref_metrics_path):
            ref_metrics = load_pickle(ref_metrics_path)
            return ref_metrics

        ref_metrics = {}
        ref_metrics["train"] = self._compute_metrics(self.reference_graph['train'],
                                                     self.reference_graph['train'])
        ref_metrics["val"] = self._compute_metrics(self.reference_graph['val'],
                                                   self.reference_graph['train'])
        ref_metrics["test"] = self._compute_metrics(self.reference_graph['test'],
                                                    self.reference_graph['train'])
        save_pickle(ref_metrics, ref_metrics_path)
        return ref_metrics

    def _compute_ratio(self,
                       metircs_of_generated_graphs: Dict[str, Any],
                       split: Literal["train", "val", "test"]):
        result = compute_ratios(metircs_of_generated_graphs,
                                self.ref_metrics[split],
                                ["degree", "clustering", "orbit", "spectre", "wavelet"])
        return result

    def _change_to_networkx_graphs(self, generated_graphs: list) -> list:
        networkx_graphs = []
        # adjacency_matrices = []
        print("Building networkx graphs...")
        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            # adjacency_matrices.append(A)
            nx_graph = nx.from_numpy_array(A)
            networkx_graphs.append(nx_graph)
        return networkx_graphs

class OldSpectreSamplingMetrics(nn.Module):
    def __init__(self, datamodule, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.nums_graphs_in_test = len(self.test_graphs)
        self.nums_graphs_in_val = len(self.val_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

        # Store for wavelet computaiton
        self.val_ref_eigvals, self.val_ref_eigvecs = compute_list_eigh(self.val_graphs)
        self.test_ref_eigvals, self.test_ref_eigvecs = compute_list_eigh(
            self.test_graphs
        )

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(
                    to_networkx(
                        data,
                        node_attrs=None,
                        edge_attrs=None,
                        to_undirected=True,
                        remove_self_loops=True,
                    )
                )
        return networkx_graphs

    def forward(
        self,
        generated_graphs: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
        log_dict : dict =None,
    ):
        prefix=""
        if test:
            prefix = "test"
        else:
            prefix = "val"
        if log_dict is None:
            prefix=""
        reference_graphs = self.test_graphs if test else self.val_graphs
        if local_rank == 0:
            print(
                f"Computing sampling metrics between {len(generated_graphs)} generated graphs and {len(reference_graphs)}"
                f" test graphs -- emd computation: {self.compute_emd}"
            )
        networkx_graphs = []
        adjacency_matrices = []
        if local_rank == 0:
            print("Building networkx graphs...")
        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            adjacency_matrices.append(A)

            nx_graph = nx.from_numpy_array(A)
            networkx_graphs.append(nx_graph)

        to_log = {}
        np.savez("generated_adjs.npz", *adjacency_matrices)

        if "degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing degree stats..")
            degree = degree_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            # if wandb.run:
            #     wandb.run.summary["degree"] = degree
            to_log["degree"] = degree

        if "rbf_mmd_degree" in self.metrics_list:
            if local_rank == 0:
                print("Computing RBF MMD on degree stats..")
            rbf_mmd_degree = degree_stats_rbf(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
            )
            to_log["rbf_mmd_degree"] = rbf_mmd_degree

        # val_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.val]
        # train_eigvals = [graph["eigval"][1:self.k + 1].cpu().detach().numpy() for graph in self.train]

        # eigval_stats(eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False)
        # spectral_filter_stats(eigvec_ref_list, eigval_ref_list, eigvec_pred_list, eigval_pred_list, is_parallel=False,
        #                       compute_emd=False)          # This is the one called wavelet

        if "wavelet" in self.metrics_list:
            if local_rank == 0:
                print("Computing wavelet stats...")

            ref_eigvecs = self.test_ref_eigvecs if test else self.val_ref_eigvecs
            ref_eigvals = self.test_ref_eigvals if test else self.val_ref_eigvals

            pred_graph_eigvals, pred_graph_eigvecs = compute_list_eigh(networkx_graphs)
            wavelet = spectral_filter_stats(
                eigvec_ref_list=ref_eigvecs,
                eigval_ref_list=ref_eigvals,
                eigvec_pred_list=pred_graph_eigvecs,
                eigval_pred_list=pred_graph_eigvals,
                is_parallel=False,
                compute_emd=self.compute_emd,
            )
            to_log["wavelet"] = wavelet
            # if wandb.run:
            #     wandb.run.summary["wavelet"] = wavelet

        if "spectre" in self.metrics_list:
            if local_rank == 0:
                print("Computing spectre stats...")
            spectre = spectral_stats(
                reference_graphs,
                networkx_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )

            to_log["spectre"] = spectre
            # if wandb.run:
            #     wandb.run.summary["spectre"] = spectre

        if "clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing clustering stats...")
            clustering = clustering_stats(
                reference_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering
            # if wandb.run:
            #     wandb.run.summary["clustering"] = clustering

        if "rbf_mmd_clustering" in self.metrics_list:
            if local_rank == 0:
                print("Computing RBF MMD on clustering stats..")
            rbf_mmd_clustering = clustering_stats_rbf(
                reference_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
            )
            to_log["rbf_mmd_clustering"] = rbf_mmd_clustering

        if "motif" in self.metrics_list:
            if local_rank == 0:
                print("Computing motif stats")
            motif = motif_stats(
                reference_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif
            # if wandb.run:
            #     wandb.run.summary["motif"] = motif

        if "orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing orbit stats...")
            orbit = orbit_stats_all(
                reference_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit
            # if wandb.run:
            #     wandb.run.summary["orbit"] = orbit

        if "rbf_mmd_orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing RBF MMD on orbit stats..")
            rbf_mmd_orbit = orbit_stats_rbf(reference_graphs, networkx_graphs)
            to_log["rbf_mmd_orbit"] = rbf_mmd_orbit

        if "fid_orbit" in self.metrics_list:
            if local_rank == 0:
                print("Computing FID on orbit stats..")
            fid_orbit = orbit_stats_fid(reference_graphs, networkx_graphs)
            to_log["fid_orbit"] = fid_orbit

        if "sbm" in self.metrics_list:
            if local_rank == 0:
                print("Computing accuracy...")
            sbm_acc = eval_acc_sbm_graph(
                networkx_graphs, refinement_steps=100, strict=True
            )
            to_log["sbm_acc"] = sbm_acc
            # if wandb.run:
            #     wandb.run.summary["sbm_acc"] = sbm_acc

        if "planar" in self.metrics_list:
            if local_rank == 0:
                print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc
            # if wandb.run:
            #     wandb.run.summary["planar_acc"] = planar_acc

        if "tree" in self.metrics_list:
            if local_rank == 0:
                print("Computing tree accuracy...")
            tree_acc = eval_acc_tree_graph(networkx_graphs)
            to_log["tree_acc"] = tree_acc
            # if wandb.run:
            #     wandb.run.summary["tree_acc"] = tree_acc

        if (
            "sbm" in self.metrics_list
            or "planar" in self.metrics_list
            or "tree" in self.metrics_list
        ):
            if local_rank == 0:
                print("Computing all fractions...")
            if "sbm" in self.metrics_list:
                validity_func = is_sbm_graph
            elif "planar" in self.metrics_list:
                validity_func = is_planar_graph
            elif "tree" in self.metrics_list:
                validity_func = nx.is_tree
            else:
                validity_func = None
            (
                frac_unique,
                frac_unique_non_isomorphic,
                fraction_unique_non_isomorphic_valid,
            ) = eval_fraction_unique_non_isomorphic_valid(
                networkx_graphs,
                self.train_graphs,
                validity_func,
            )
            frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(
                networkx_graphs, self.train_graphs
            )
            to_log.update(
                {
                    "sampling/frac_unique": frac_unique,
                    "sampling/frac_unique_non_iso": frac_unique_non_isomorphic,
                    "sampling/frac_unic_non_iso_valid": fraction_unique_non_isomorphic_valid,
                    "sampling/frac_non_iso": frac_non_isomorphic,
                }
            )

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )
        to_log.update(ratios)

        if local_rank == 0:
            print("Sampling statistics", to_log)
        # if wandb.run:
        #     wandb.log(to_log, commit=False)
        if log_dict is not None:
            for key, value in to_log.items():
                if f"{prefix}/{key}" in log_dict:
                    raise ValueError(f"Key {key} already exists in log_dict")
                else:
                    log_dict[f"{prefix}/{key}"] = value
        return to_log

    def reset(self):
        pass


class Comm20SamplingMetrics(SpectreSamplingMetrics):

    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=True,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class CommunitySamplingMetrics(SpectreSamplingMetrics):

    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=True,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "rbf_mmd_degree",
                "rbf_mmd_clustering",
                "rbf_mmd_orbit",
                "fid_orbit",
            ],
        )


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "planar",
                'fraction'
            ],
        )

class TLSSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "planar", # Just for consistency
                "tls",
            ],
        )


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet", "sbm",'fraction'],
        )


class TreeSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "tree",
                'fraction'
            ],
        )


class EgoSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "wavelet",
                "rbf_mmd_degree",
                "rbf_mmd_clustering",
                "rbf_mmd_orbit",
                "fid_orbit",
            ],
        )


class ProteinSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


class IMDBSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(
            datamodule=datamodule,
            compute_emd=False,
            metrics_list=["degree", "clustering", "orbit", "spectre", "wavelet"],
        )


# ----------------- Minimal metrics for community/caveman/cora/breast/enzymes/ego-small -----------------

class DegreeClusteringOrbitMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule, compute_emd: bool = True):
        super().__init__(
            datamodule=datamodule,
            compute_emd=compute_emd,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
            ],
        )


class CommunitySmallSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)


class CavemanSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)


class CoraSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)


class BreastSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)


class EnzymesSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)


class EgoSmallSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)

class GridSamplingMetrics(DegreeClusteringOrbitMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule, compute_emd=True)