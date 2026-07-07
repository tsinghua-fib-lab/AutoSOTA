import os
import random
import argparse
import sys
from time import time
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import sklearn.metrics as metrics
from munkres import Munkres
from model import HAMC_Model

def data_load(args):
    #New dataset
    data_path = './Data/' + args.dataset + '.mat'
    data_mat = scipy.io.loadmat(data_path)
    data = []
    if args.dataset in ['Scene15', 'Caltech101', 'BDGP', 'Reuters', 'Animal', 'Wiki', 'NoisyMNIST','MNIST-USPS', 'CUB']:
        data.append(data_mat['X'][0][0])
        data.append(data_mat['X'][0][1])
        label = np.squeeze(data_mat['Y'])
    elif args.dataset in ['100Leaves', 'LandUse21',]:
        data.append(data_mat['X'][0][0])
        data.append(data_mat['X'][0][1])
        data.append(data_mat['X'][0][2])
        label = np.squeeze(data_mat['Y'])
    elif args.dataset in ['YoutubeFace', 'ALOI100']:
        data.append(data_mat['X'][0][0])
        data.append(data_mat['X'][0][1])
        data.append(data_mat['X'][0][2])
        data.append(data_mat['X'][0][3])
        label = np.squeeze(data_mat['Y'])
    else:
        raise NotImplementedError('Dataset not implemented: ' + args.dataset)
    return data, label

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('error')
        return 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    return acc

def calculate_entropy(distances):

    log_probs = F.log_softmax(-distances, dim=1)
    
    probs = torch.exp(log_probs)
    
    entropy = -torch.sum(probs * log_probs, dim=1)
    
    return entropy

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True) 
    
    os.environ['PYTHONHASHSEED'] = str(seed)
