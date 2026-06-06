import copy

import numpy as np
import torch
from numpy import linalg as LA
from numpy.linalg import norm
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import utils_ood as utils

recall = 0.95


def neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, model_architecture_type, neco_dim):
    '''
    Prints the auc/fpr result for NECO method.

            Parameters:
                    feature_id_train (array): An array of training samples features
                    feature_id_val (array): An array of evaluation samples features
                    feature_ood (array): An array of OOD samples features
                    logit_id_val (array): An array of evaluation samples logits
                    logit_ood (array): An array of OOD samples logits
                    model_architecture_type (string): Module architecture used
                    neco_dim (int): ETF approximative dimmention for the tested case

            Returns:
                    None
    '''
    method = 'NECO'
    ss = StandardScaler()  # if NC1 is well verified, i.e a well seperated class clusters (case of cifar using ViT, its better not to use the scaler)
    complete_vectors_train = ss.fit_transform(feature_id_train)
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)

    pca_estimator = PCA(feature_id_train.shape[1])
    _ = pca_estimator.fit_transform(complete_vectors_train)
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)
    if model_architecture_type in ['deit', 'swin']:
        complete_vectors_train = feature_id_train
        complete_vectors_test = feature_id_val
        complete_vectors_ood = feature_ood

    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]
    l_ID = []
    l_OOD = []

    for i in range(cls_test_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_test[i, :]))
        sc = LA.norm(cls_test_reduced[i, :])
        sc_finale = sc/sc_complet
        l_ID.append(sc_finale)
    for i in range(cls_ood_reduced.shape[0]):
        sc_complet = LA.norm((complete_vectors_ood[i, :]))
        sc = LA.norm(cls_ood_reduced[i, :])
        sc_finale = sc/sc_complet
        l_OOD.append(sc_finale)
    l_OOD = np.array(l_OOD)
    l_ID = np.array(l_ID)
    #############################################################
    score_id = l_ID
    score_ood = l_OOD
    if model_architecture_type != 'resnet':
        score_id *= score_id_maxlogit
        score_ood *= score_ood_maxlogit
    auc_ood = utils.auc(score_id, score_ood)[0]
    recall = 0.95
    fpr_ood, _ = utils.fpr_recall(score_id, score_ood, recall)
    print(f' \n {method}: auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
