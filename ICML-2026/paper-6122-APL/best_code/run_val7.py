import math
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
import pandas as pd
import os
import time
import sys
from utility import get_acc
from utility import get_macro_f1
from method_PTBCC import PTBCC
import scipy.sparse as ssp_mv



def majority_vote_phi(tuples, num_items, num_workers, num_classes):
    """Compute majority-vote phi_ik from label tuples."""
    phi = np.zeros((num_items, num_classes), dtype=np.float64)
    for l in range(num_classes):
        selected = (tuples[:, 2] == l)
        coo = ssp_mv.coo_matrix(
            (np.ones(selected.sum()), tuples[selected, :2].T),
            shape=(num_items, num_workers), dtype=bool)
        phi[:, [l]] += coo.tocsr().sum(axis=-1)
    phi /= phi.sum(axis=1, keepdims=True)
    nan_rows = np.isnan(phi).any(axis=1)
    phi[nan_rows] = 1.0 / num_classes
    return phi


def CPBCC(dataset, tuples, S=2, max_iter=1000):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix(
            (np.ones(selected.sum()), tuples[selected, :2].T),
            shape=(num_items, num_workers),
            dtype=bool)
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    has_label = np.zeros((num_workers, num_items), dtype=bool)
    for k in range(num_classes):
        has_label |= y_is_one_lji[k].astype(bool).toarray()

    phi_ik = np.zeros((num_items, num_classes), dtype=np.float64)
    for l in range(num_classes):
        phi_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1)
    phi_ik /= phi_ik.sum(axis=1, keepdims=True)
    u_k = phi_ik.sum(axis=0)
    if dataset == 'valence5' or dataset == 'valence7':
        temp_ksl = np.full((num_classes, S, num_classes), 1, dtype=np.float64)
        k = np.arange(num_classes)
        temp_ksl[k, :, k] = 5
        temp_ksl /= temp_ksl.sum(axis=-1, keepdims=True)

        # Generalized prototype weight init for arbitrary S
        # Uses exponential decay: larger weights for primary prototypes
        temp_jks = np.zeros((num_workers, num_classes, S))
        if S == 2:
            weights = np.array([0.6, 0.1], dtype=np.float64)
        elif S == 3:
            weights = np.array([0.5, 0.2, 0.01], dtype=np.float64)
        elif S == 4:
            weights = np.array([0.5, 0.2, 0.1, 0.01], dtype=np.float64)
        elif S == 5:
            weights = np.array([0.5, 0.2, 0.1, 0.05, 0.01], dtype=np.float64)
        elif S == 6:
            weights = np.array([0.4, 0.2, 0.1, 0.08, 0.05, 0.01], dtype=np.float64)
        else:
            # Exponential decay for arbitrary S
            weights = np.exp(-0.8 * np.arange(S))
            weights /= weights.sum()
        for s in range(S):
            temp_jks[:, :, s] = weights[min(s, len(weights)-1)]

        theta_jiks = np.zeros((num_workers, num_items, num_classes, S))
        for l in range(num_classes):
            j_idx, i_idx = y_is_one_lji[l].nonzero()
            theta_jiks[j_idx, i_idx, :, :] = (temp_jks[j_idx] * temp_ksl[:, :, l])

        beta_jks = np.full((num_workers, num_classes, S), 1e-5, dtype=np.float64)
        j_idx, i_idx = np.where(has_label)
        phi = phi_ik[i_idx]  # (N, K)
        theta = theta_jiks[j_idx, i_idx]
        np.add.at(beta_jks, j_idx, np.einsum("nk,nks->nks", phi, theta))
    else:
        theta_jiks = np.full((num_workers, num_items, num_classes, S), 1e-5, dtype=np.float32)
        j_idx, i_idx = np.where(has_label)
        phi_sub = phi_ik[i_idx]
        theta_jiks[j_idx, i_idx, :, 0] += phi_sub
        theta_jiks[j_idx, i_idx, :, 1] += 0.35
        theta_jiks[j_idx, i_idx, :, 2] += 0.1

        theta_sum = theta_jiks.sum(axis=3, keepdims=True)
        theta_jiks /= np.where(theta_sum == 0, 1.0, theta_sum)

        beta_jks = np.zeros((num_workers, num_classes, S), dtype=np.float64)
        j_idx, i_idx = np.where(has_label)
        j_rev = j_idx[::-1]
        i_rev = i_idx[::-1]
        j_unique, pos = np.unique(j_rev, return_index=True)
        i_last = i_rev[pos]
        beta_jks[j_unique, :, :] = (phi_ik[i_last, :, None] * theta_jiks[j_unique, i_last, :, :])
        beta_jks += 1e-5

    a_ksl = np.zeros((num_classes, S, num_classes), dtype=np.float64)
    for l in range(num_classes):
        i_idx, j_idx = y_is_one_lij[l].nonzero()
        phi_sub = phi_ik[i_idx]
        theta_sub = theta_jiks[j_idx, i_idx]
        for k in range(num_classes):
            for s in range(S):
                a_ksl[k, s, l] += np.sum(phi_sub[:, k] * theta_sub[:, k, s])
    a_ksl += 1

    theta_sum = theta_jiks.sum(axis=3, keepdims=True)
    theta_jiks /= np.where(theta_sum == 0, 1.0, theta_sum)

    beta_jks *= 0.6
    a_ksl *= 0.9
    for it in range(max_iter):
        tau_k = u_k + phi_ik.sum(axis=0)
        j_idx, i_idx = np.where(has_label)
        phi = phi_ik[i_idx]  # (N, K)
        theta = theta_jiks[j_idx, i_idx]
        eta_jks = np.zeros((num_workers, num_classes, S))
        np.add.at(eta_jks, j_idx, np.einsum("nk,nks->nks", phi, theta))
        eta_jks += beta_jks

        mu_ksl = a_ksl.copy()
        for l in range(num_classes):
            y_lij = y_is_one_lij[l]
            rows, cols = y_lij.nonzero()
            phi_sub = phi_ik[rows]
            theta_sub = theta_jiks[cols, rows]
            mu_ksl[:, :, l] += np.einsum("ik,iks->ks", phi_sub, theta_sub)

        E_log_tau_k = digamma(tau_k) - digamma(np.sum(tau_k))
        E_log_pi_jks = digamma(eta_jks) - digamma(eta_jks.sum(axis=2, keepdims=True))
        E_log_v_ksl = digamma(mu_ksl) - digamma(mu_ksl.sum(axis=2, keepdims=True))

        theta_jiks.fill(0.0)
        theta_jiks += E_log_pi_jks[:, None, :, :]
        for l in range(num_classes):
            y_lji = y_is_one_lji[l]
            j_idx, i_idx = y_lji.nonzero()
            theta_jiks[j_idx, i_idx] += E_log_v_ksl[:, :, l]

        j_idx, i_idx = np.where(has_label)
        theta_sub = theta_jiks[j_idx, i_idx]
        theta_sub -= theta_sub.max(axis=2, keepdims=True)
        np.exp(theta_sub, out=theta_sub)
        theta_sub /= theta_sub.sum(axis=2, keepdims=True)
        theta_jiks[j_idx, i_idx] = theta_sub

        last_phi_ik = phi_ik.copy()
        log_phi = np.broadcast_to(E_log_tau_k[None, :],(num_items, num_classes)).copy()
        for l in range(num_classes):
            y_lij = y_is_one_lij[l]
            i_idx, j_idx = y_lij.nonzero()
            if len(i_idx) == 0:
                continue
            theta_sub = theta_jiks[j_idx, i_idx]
            contrib = (E_log_v_ksl[:, :, l][None, :, :] + E_log_pi_jks[j_idx])
            delta = np.sum(theta_sub * contrib, axis=2)
            np.add.at(log_phi, i_idx, delta)
        log_phi -= log_phi.max(axis=1, keepdims=True)
        np.exp(log_phi, out=log_phi)
        phi_ik = log_phi
        phi_ik /= phi_ik.sum(axis=1, keepdims=True)

        if np.allclose(last_phi_ik, phi_ik, atol=1e-3):
            if it < 10 or it % 50 == 0:
                print(f"  Converged at iter {it+1}", flush=True)
            break

    return phi_ik, num_classes


if __name__ == "__main__":
    dataset = 'valence7'
    iteration = 10
    sum_acc = 0
    sum_macrofscore = 0
    sum_time = 0
    accuracies = []
    macrofscores = []

    print(f"Running 4-way CPBCC Ensemble on {dataset}")
    print(f"Ensemble: 0.50*CPBCC(S=3) + 0.20*CPBCC(S=2) + 0.25*PTBCC(S=2) + 0.05*MajorityVote")
    print(f"Settings: S=3, max_iter=1000, n_runs={iteration}")
    print()

    truth_file = './datasets/' + dataset + '/truth.csv'
    datafile = "./datasets/" + dataset + "/label.csv"
    df_label = pd.read_csv(datafile)
    df_truth = pd.read_csv(truth_file)
    num_items = df_label.values[:, 0].max() + 1
    num_workers = df_label.values[:, 1].max() + 1
    num_classes = df_label.values[:, 2].max() + 1

    # Pre-compute majority vote phi (same every run)
    mv_phi = majority_vote_phi(df_label.values, num_items, num_workers, num_classes)

    for i in range(iteration):
        starttime = time.time()

        # Run CPBCC with S=3
        phi_cpbcc_s3, _ = CPBCC(dataset, df_label.values, S=3, max_iter=1000)

        # Run CPBCC with S=2
        phi_cpbcc_s2, _ = CPBCC(dataset, df_label.values, S=2, max_iter=1000)

        # Run PTBCC
        phi_ptbcc, _ = PTBCC(df_label.values, S=2, max_iter=1000)

        # 4-way weighted ensemble
        phi_ensemble = 0.50 * phi_cpbcc_s3 + 0.20 * phi_cpbcc_s2 + 0.25 * phi_ptbcc + 0.05 * mv_phi

        duration = time.time() - starttime
        accuracy = get_acc(phi_ensemble, df_truth)
        macrofscore = get_macro_f1(phi_ensemble, df_truth)
        print(f"  Run {i+1}: acc={accuracy:.4f} macro_f1={macrofscore:.4f} time={duration:.2f}s")
        accuracies.append(accuracy)
        macrofscores.append(macrofscore)
        sum_acc += accuracy
        sum_macrofscore += macrofscore
        sum_time += duration

    mean_acc = sum_acc / iteration
    mean_mf1 = sum_macrofscore / iteration
    mean_time = sum_time / iteration
    std_acc = np.std(accuracies)
    std_mf1 = np.std(macrofscores)

    print()
    print("=" * 60)
    print(f"RESULTS for {dataset} (n_runs={iteration}):")
    print(f"  Accuracy:  {mean_acc*100:.2f}%  (baseline: 48.00%)")
    print(f"  Macro-F1:  {mean_mf1*100:.2f}%  (baseline: 36.01%)")
    print(f"  Avg time:  {mean_time:.2f}s per run")
    print(f"  Acc std:   {std_acc:.6f}")
    print(f"  F1 std:    {std_mf1:.6f}")
    print(f"  Individual accuracies: {[f'{a*100:.2f}%' for a in accuracies]}")
    print(f"  Individual macro-F1s:  {[f'{f*100:.2f}%' for f in macrofscores]}")
    print("=" * 60)
