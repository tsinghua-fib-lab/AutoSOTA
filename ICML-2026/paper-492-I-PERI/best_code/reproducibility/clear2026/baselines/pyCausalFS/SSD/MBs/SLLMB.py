#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/9/17 8:48
 @File    : SLL.py
 """

from SSD.MBs.common.optimalnetwork import optimal_network
import numpy as np



def find_potential_neighbors(data,target):
    number, kVar = np.shape(data)
    O_set = [i for i in range(kVar) if i != target]
    ht_set = []
    for v in O_set:
        Z = list(set(ht_set).union(set([target, v])))
        A = optimal_network(Z, data)
        ht_set =[i for i in range(kVar) if i != target and A[i, target] == 1 or A[target, i] == 1]
    return ht_set

def find_neighbors(data, target):
    ht_set = find_potential_neighbors(data, target)
    ht_temp = ht_set.copy()
    for v in ht_temp:
        hv = find_potential_neighbors(data, v)
        if target not in hv:
            ht_set.remove(v)

    return ht_set

def find_potential_spouses(data, target, ht_set):
    _, kVar = np.shape(data)
    ht_neighbors = []
    for x in ht_set:
        vari_set = find_neighbors(data, x)
        ht_neighbors = set(ht_neighbors).union(set(vari_set))
    O_set = set(ht_neighbors).difference(set(ht_set).union(set([target])))
    st_set = []
    for v in O_set:
        Z = set([target, v]).union(ht_neighbors).union(st_set)
        A = optimal_network(Z, data)
        st_set = [i for i in range(kVar) for j in range(kVar) if i != target and A[target, j] == 1 and A[i, j] == 1]

    return st_set

def find_spouses(data, target, ht_set, hv_set):
    _, kVar = np.shape(data)
    st_set = find_potential_spouses(data, target,ht_set)
    vari_set = [i for i in range(kVar) if i != target and i not in ht_set]
    for v in vari_set:
        sv_set = find_potential_spouses(data, v, hv_set)
        if target in sv_set:
            st_set.append(v)

    return st_set

def SLL(data,target):
    pc_t = find_neighbors(data, target)
    # print("pc_t is: " + str(pc_t))
    hv_set = []
    for x in pc_t:
        h_x = find_neighbors(data,x)
        pc_x = find_potential_spouses(data, x, h_x)
        hv_set = list(set(hv_set).union(set(pc_x)))
    sp_t = find_spouses(data, target, pc_t, hv_set)
    MB = list(set(pc_t).union(set(sp_t)))

    return pc_t, MB


# import pandas as pd
# data = pd.read_csv("C:/pythonProject/BN_PC_algorithm/Alarm_data/Alarm1_s500_v1.csv")
# target = 5
# res = SLL(data, target)
# print("res is: " + str(res))












