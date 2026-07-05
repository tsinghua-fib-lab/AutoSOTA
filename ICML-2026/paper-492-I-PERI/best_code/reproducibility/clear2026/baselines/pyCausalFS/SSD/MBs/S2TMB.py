#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/10/13 19:01
 @File    : S^2TMB.py
 """
import numpy as np
from SSD.MBs.common.optimalnetwork import optimal_network
def S2TMB(data, target):
    # step 1:find the PC set
    _, kVar = np.shape(data)
    pc_t = []
    o_set = [i for i in range(kVar) if i != target]
    for x in o_set:
        Z = set([target,x]).union(pc_t)
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]

    # step2: remove false PC nodes and find spouses
    spouses_t = []
    varis_set = [i for i in range(kVar) if i != target and i not in pc_t]
    for x in varis_set:
        Z = set([target, x]).union(set(pc_t)).union(set(spouses_t))
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        spouses_t = [i for i in range(kVar) for j in range(kVar) if i != target and DAG[target, j] == 1 and DAG[i, j] == 1]

    MB = list(set(pc_t).union(set(spouses_t)))
    return pc_t, MB



def S2TMB_p(data, target):
    # step 1:find the PC set
    _, kVar = np.shape(data)
    pc_t = []
    o_set = [i for i in range(kVar) if i != target]
    for x in o_set:
        Z = set([target,x]).union(pc_t)
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]

    # step 2: remove false PC nodes and find spouses
    spouses_t = []
    varis_set = [i for i in range(kVar) if i != target and i not in pc_t]
    for x in varis_set:
        Z = set([target, x]).union(set(pc_t)).union(set(spouses_t))
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        sp = [i for i in range(kVar) for j in range(kVar) if i != target and DAG[target, j] == 1 and DAG[i, j] == 1]
        spouses_t = set(spouses_t).union(sp)

    # step 3: shrink spouses
    var_set = spouses_t.copy()
    spouses_t = []
    for x in var_set:
        Z = set([target, x]).union(pc_t).union(spouses_t)
        DAG = optimal_network(Z, data)
        pc_t = [i for i in range(kVar) if DAG[target, i] == 1 or DAG[i, target] == 1]
        spouses_t = [i for i in range(kVar) for j in range(kVar) if i != target and DAG[target, j] == 1 and DAG[i, j] == 1]

    MB = list(set(pc_t).union(set(spouses_t)))
    return pc_t, MB

# import pandas as pd
# data = pd.read_csv("C:/pythonProject/BN_PC_algorithm/data/child_s5000_v2.csv")
# target = 3
# res = S2TMB_p(data, target)
# print("res is: " + str(res))
#

