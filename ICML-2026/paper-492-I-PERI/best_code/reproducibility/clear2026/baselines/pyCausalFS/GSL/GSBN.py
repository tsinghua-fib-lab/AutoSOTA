#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2020/7/30 8:55
 @File    : GSBN.py
 """
import numpy as np
from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.Meek import meek
from CBD.MBs.common.condition_independence_test import cond_indep_test

def GSMB(data, target, alaph, is_discrete):
    number, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    circulateFlag = True
    S_variables = [i for i in range(kVar) if i !=target]

    """grow phase"""
    # print("grow phase")
    while circulateFlag:
        circulateFlag = False
        for x in S_variables:
            ci_number += 1
            pval_gp, dep_gp = cond_indep_test(data, target, x, CMB, is_discrete)
            if pval_gp < alaph:
                # print("CMB append is: "+str(x))
                CMB.append(x)
                circulateFlag = True
                break
        S_variables = [i for i in range(kVar) if i != target and i not in CMB]

    """"shrink phase"""
    # print("shrink phase")
    circulateFlag = True
    while circulateFlag:
        circulateFlag = False
        CMB_temp = CMB.copy()
        for x in CMB_temp:
            subsets_CMB = [i for i in CMB if i != x]
            ci_number += 1
            pval_sp, dep_sp= cond_indep_test(data, target, x, subsets_CMB, is_discrete)
            if pval_sp > alaph :
                # print("CMB remove is: "+ str(x))
                CMB.remove(x)
                circulateFlag = True
                break

    return list(set(CMB)), ci_number


def GSBN(data, alpha, is_discrete):
    _, kvar = np.shape(data)
    max_k = 3
    all_MB = [[] for i in range(kvar)]
    all_neighbor = [[] for i in range(kvar)]
    DAG = np.zeros((kvar, kvar))
    # Compute Markov Blankets

    # Set initial cache value
    dict_cache = {}
    dict_cache.setdefault("cache", [0, 0])

    for i in range(kvar):
        MB, _ = GSMB(data, i, alpha, is_discrete)
        for j in MB:
            DAG[i, j] = 1

    # # AND Rule
    # for i in range(kvar):
    #     for j in range(0, i):
    #         if DAG[i, j] != DAG[j, i]:
    #             DAG[i, j] = 0
    #             DAG[j, i] = 0

    # OR Rule
    for i in range(kvar):
        for j in range(0, i):
            if DAG[i, j] != DAG[j, i]:
                DAG[i, j] = 1
                DAG[j, i] = 1

    for i in range(kvar):
        for j in range(kvar):
            if DAG[i, j] == 1:
                all_MB[i].append(j)

    # Compute Graph Structure
    for x in range(kvar):
        for y in all_MB[x]:
            vs = set(all_MB[x]).union(set(all_MB[y]))
            varis = [i for i in vs if i != x and i != y]
            k = 0
            break_flag = False
            while len(varis) > k and k <= max_k:
                ss = subsets(varis, k)
                for s in ss:
                    pval, _ = cond_indep_test(data, x, y, s, is_discrete)
                    if pval > alpha:
                        DAG[x, y] = 0
                        DAG[x, y] = 0
                        break_flag = True
                        break
                if break_flag:
                    break
                k += 1

    for i in range(kvar):
        for j in range(kvar):
            if DAG[x, y] == 1:
                all_neighbor[i].append(j)

    PP = DAG.copy()
    pdag = DAG.copy()
    G = DAG.copy()
    print("2")
    # Orient Edges
    for x in range(kvar):
        for y in all_neighbor[x]:

            PP[y, x] = -1
            nz_vars = [i for i in all_neighbor[x] if i != y and i not in all_neighbor[y] ]
            for z in nz_vars:

                vs_vars = set(all_neighbor[y]).union(all_neighbor[z])
                vs = [i for i in vs_vars if i != z and i != y]
                k = 0
                break_flag = False
                while len(vs) > k and k <= max_k:
                    ss = subsets(vs, k)

                    for s in ss:
                        con_set = [i for i in s]
                        con_set.append(x)
                        con_set = list(set(con_set))
                        pval, _ = cond_indep_test(data, y, z, con_set, is_discrete)
                        if pval > alpha:
                            PP[y, x] = 1
                            break_flag = True
                            break
                    if break_flag:
                        break

                    k += 1

                if PP[y, x] == -1:
                    pdag[y, x] = -1
                    pdag[x, y] = 0

                    G[y, x] = 1
                    G[x, y] = 0

    print("3")
    # Remove Cycles
    [DAG, pdag, G] = meek(DAG, pdag, G, kvar)

    return pdag



# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# data = pd.read_csv("D:/data/Alarm_data/Alarm1_s5000_v7.csv")
# print("the file read")
# import numpy as np
# num1, kvar = np.shape(data)
# alpha = 0.01
#
# pdag, dic = GSBN(data, alpha, True)
# print(pdag)
# for i in range(kvar):
#     for j in range(kvar):
#         if pdag[i, j] == -1:
#             print("i: ", i, " ,j: ", j)
# print(dic["cache"][0]/(dic["cache"][0]+dic["cache"][1]))