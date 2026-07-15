import math
import numpy as np
import scipy.sparse as ssp
from scipy.special import digamma, gammaln
import pandas as pd
import os
import time
np.set_printoptions(suppress=True, precision=6)
from utility import get_acc
from utility import get_macro_f1


def PTBCC(tuples, S=2, max_iter=1000):
    num_items, num_workers, num_classes = tuples.max(axis=0) + 1

    y_is_one_lij = []
    y_is_one_lji = []
    for k in range(num_classes):
        selected = (tuples[:, 2] == k)
        coo_ij = ssp.coo_matrix((np.ones(selected.sum()), tuples[selected, :2].T), shape=(num_items, num_workers),
                                dtype=np.bool_)
        y_is_one_lij.append(coo_ij.tocsr())
        y_is_one_lji.append(coo_ij.T.tocsr())

    phi_ik = np.zeros((num_items, num_classes), dtype=np.float64)
    for l in range(num_classes):
        phi_ik[:, [l]] += y_is_one_lij[l].sum(axis=-1)
    phi_ik /= phi_ik.sum(axis=1, keepdims=True)

    u_k_vec = phi_ik.sum(axis=0)

    skl_vec = np.zeros((S, num_classes, num_classes), dtype=np.float64)
    for s in range(S):
        if s == 0:
            skl_vec[s] = np.full((num_classes, num_classes), 1, dtype=np.float64)
            np.fill_diagonal(skl_vec[s], 5)
            skl_vec[s] /= skl_vec[s].sum(axis=1, keepdims=True)
        elif s == 1:
            skl_vec[s] = np.full((num_classes, num_classes), 1.35, dtype=np.float64)
            np.fill_diagonal(skl_vec[s], 1)
            skl_vec[s] /= skl_vec[s].sum(axis=1, keepdims=True)

    theta_jis_vec = np.zeros((num_workers, num_items, S), dtype=np.float64)
    for l in range(num_classes):
        y_l_ji = y_is_one_lji[l].tocoo()
        rows = y_l_ji.row
        cols = y_l_ji.col
        for k in range(num_classes):
            phi_i_k = phi_ik[cols, k]
            for s in range(S):
                weight = skl_vec[s, k, l]
                theta_jis_vec[rows, cols, s] += phi_i_k * weight

    beta_js_vec = theta_jis_vec.sum(axis=1)
    a_skl_vec = np.zeros((S, num_classes, num_classes), dtype=np.float64)
    for l in range(num_classes):
        y_l_ji = y_is_one_lji[l]
        for s in range(S):
            theta_s = theta_jis_vec[:, :, s]
            temp = y_l_ji.multiply(theta_s)
            result = temp.T @ np.ones(temp.shape[0])
            a_skl_vec[s, :, l] = phi_ik.T @ result
    sums = np.sum(theta_jis_vec, axis=2, keepdims=True)
    non_zero_mask = (sums.squeeze() != 0)
    theta_jis_vec[non_zero_mask] /= sums[non_zero_mask]

    beta_js_vec *= 0.4
    a_skl_vec *= 0.5

    for it in range(max_iter):
        nu_k_vec = u_k_vec + phi_ik.sum(axis=0)

        eta_js_vec = beta_js_vec.copy()
        worker_item_mask = sum(y_is_one_lji)
        for s in range(S):
            eta_js_vec[:, s] += worker_item_mask.multiply(theta_jis_vec[:, :, s]).sum(axis=1).A1

        mu_skl_vec = a_skl_vec.copy()
        for s in range(S):
            theta_s = theta_jis_vec[:, :, s]
            for l in range(num_classes):
                y_l_ji = y_is_one_lji[l]
                temp = y_l_ji.multiply(theta_s)
                result = temp.T @ np.ones(temp.shape[0])
                mu_skl_vec[s, :, l] += phi_ik.T @ result

        Eq_log_tau_k = digamma(nu_k_vec) - digamma(nu_k_vec.sum())
        Eq_log_pi_js = digamma(eta_js_vec) - digamma(eta_js_vec.sum(axis=-1, keepdims=True))
        Eq_log_v_skl = digamma(mu_skl_vec) - digamma(mu_skl_vec.sum(axis=-1, keepdims=True))

        theta_jis_vec[:] = Eq_log_pi_js[:,None,:] - 1
        for s in range(S):
            Eq_log_v_skl[s] = np.nan_to_num(Eq_log_v_skl[s], posinf=1e10, neginf=-1e10)
            phi_v_product = phi_ik.dot(Eq_log_v_skl[s])
            for l in range(num_classes):
                y_l_ji = y_is_one_lji[l].tocoo()
                rows = y_l_ji.row
                cols = y_l_ji.col
                theta_jis_vec[rows, cols, s] += phi_v_product[cols, l]

        theta_jis_vec = np.exp(theta_jis_vec)
        theta_jis_vec /= theta_jis_vec.sum(axis=2, keepdims=True)

        last_phi = phi_ik.copy()

        phi_ik[:] = Eq_log_tau_k[None, :] - 1
        for l in range(num_classes):
            y_l_ij = y_is_one_lij[l]
            for s in range(S):
                theta_jis = theta_jis_vec[:, :, s].T
                temp = y_l_ij.multiply(theta_jis)
                result = temp @ np.ones(temp.shape[1])
                phi_ik += np.outer(result, Eq_log_v_skl[s, :, l])
        phi_ik = np.exp(phi_ik)
        phi_ik /= phi_ik.reshape(num_items, -1).sum(axis=-1)[:, None]

        if np.allclose(last_phi, phi_ik, atol=1e-3):
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
            phi_ik, num_classes = PTBCC(df_label.values, S=2,  max_iter=1000)
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

