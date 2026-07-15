import math
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
import pandas as pd
import os
import time
from utility import get_acc
from utility import get_macro_f1



def CPBCC(dataset, tuples, S=2, max_iter=1000):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix(
            (np.ones(selected.sum()), tuples[selected, :2].T),
            shape=(num_items, num_workers),
            dtype=np.bool_)
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

        temp_jks = np.zeros((num_workers, num_classes, S))
        temp_jks[:, :, 0] = 0.5
        temp_jks[:, :, 1] = 0.2
        temp_jks[:, :, 2] = 0.01

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
    for _ in range(max_iter):
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
            break

    return phi_ik, num_classes


datasets = ['LabelMe', 'aircrowd6', 'valence5', 'valence7', 'CF', 'MS', 's4_Dog_data', 's4_Face_Sentiment_Identification',
                's5_AdultContent', 'web']


if __name__ == "__main__":
    iteration = 10
    total_accu = 0
    total_time = 0
    total_macrofscore = 0

    for dataset in datasets:
        print(dataset)
        sum_acc = 0
        sum_macrofscore = 0
        sum_time = 0
        accuracies = []
        macrofscores = []
        times = []
        for i in range(iteration):
            tempaccuracies = []
            tempmacrofscores = []
            temptime = []
            truth_file = './datasets/' + dataset + '/truth.csv'
            datafile = "./datasets/" + dataset + "/label.csv"
            starttime = time.time()
            df_label = pd.read_csv(datafile)
            df_truth = pd.read_csv(truth_file)

            phi_ik, num_classes = CPBCC(dataset, df_label.values, S=3,  max_iter=1000)

            duration = time.time() - starttime
            accuracy = get_acc(phi_ik, df_truth)
            macrofscore = get_macro_f1(phi_ik, df_truth)
            print("accu: " + str(accuracy) + "    macro_fscore: " + str(macrofscore) + "    duration: " + str(duration))
            sum_acc += accuracy
            sum_macrofscore += macrofscore
            sum_time += duration

        total_accu += (sum_acc / iteration)
        total_time += (sum_time / iteration)
        total_macrofscore += (sum_macrofscore / iteration)

    print("total_accu: " + str(total_accu / len(datasets)))
    print("total_macrofscore: " + str(total_macrofscore / len(datasets)))
    print("total_time: " + str(total_time / len(datasets)))