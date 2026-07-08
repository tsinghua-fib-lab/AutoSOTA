import os
import random
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import networkx as nx
import uuid
from torch.utils.data import DataLoader, Dataset
from collections import Counter


def is_acyclic(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return nx.is_directed_acyclic_graph(G)

def adj2order(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return list(nx.topological_sort(G))

def full_DAG(top_order):
    d = len(top_order)
    A = np.zeros((d,d))
    for i, var in enumerate(top_order):
        A[var, top_order[i+1:]] = 1
    return A

def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order

def np_to_csv(array, save_path):
        """
        Convert np array to .csv
        array: numpy array
            the numpy array to convert to csv
        save_path: str
            where to temporarily save the csv
        Return the path to the csv file
        """
        id = str(uuid.uuid4())
        #output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')
        output = os.path.join(save_path, 'tmp_' + id + '.csv')

        df = pd.DataFrame(array)
        df.to_csv(output, header=False, index=False)

        return output

def normlize_data(data):
    mu = data.mean(axis=0)
    sigma = data.std(axis=0)
    data = (data-mu) / sigma
    return data

def get_parents_score(curr_H, full_H, i):
    full_H = torch.hstack([full_H[0:i], full_H[i+1:]])
    parents_score = np.abs(curr_H - full_H)
    parents_score = torch.cat([parents_score[:i], torch.tensor([0.0]), parents_score[i:]])
    return parents_score

def add_edge_with_parents_score(dag, parents_score, lambda2):
    avg_parent_score = (dag * parents_score.T).mean()
    add_edge = (parents_score.T >= lambda2 * avg_parent_score).astype(np.int32)
    idx = np.transpose(np.nonzero(add_edge))
    val = np.array([parents_score.T[i[0]][i[1]] for i in idx])
    sorted_idx = np.argsort(-val)
    idx = idx[sorted_idx]
    for i in idx:
        if dag[i[0], i[1]] == 1:
            continue
        else:
            dag[i[0], i[1]] = 1
            if is_acyclic(dag):
                continue
            else:
                dag[i[0], i[1]] = 0
    return dag

def pre_pruning_with_parents_score(init_dag, parents_score, lambda1):
    d = init_dag.shape[0]
    threshold = (np.max(parents_score, axis=1)/lambda1).reshape(d, 1)
    threshold = np.tile(threshold, (1, d))
    mask = (parents_score >= threshold).T.astype(np.int32)
    return init_dag * mask