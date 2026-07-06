#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2020/8/14 10:50
 @File    : MB_GSL.py
 """
import numpy as np
from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.Meek import meek
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.HITON.HITON_MB import HITON_MB
from CBD.MBs.semi_HITON.semi_HITON_MB import semi_HITON_MB
from CBD.MBs.PCMB.PCMB import PCMB
from CBD.MBs.MMMB.MMMB import MMMB


def MBGSL(data, alpha, is_discrete, selected):
    _, kvar = np.shape(data)
    max_k = 3
    all_MB = [[] for i in range(kvar)]
    all_neighbor = [[] for i in range(kvar)]
    PP = np.zeros((kvar, kvar))
    num_CI = 0

    for i in range(kvar):
        if selected == 1:
            MB, n_c = MMMB(data, i, alpha, is_discrete)
        elif selected == 2:
            MB, n_c = HITON_MB(data, i, alpha, is_discrete)
        elif selected == 3:
            MB, n_c = semi_HITON_MB(data, i, alpha, is_discrete)
        else:
            MB, n_c, dict_cache = PCMB(data, i, alpha, is_discrete)
        num_CI += n_c
        for j in MB:
            PP[i, j] = 1

    # # AND Rule
    # for i in range(kvar):
    #     for j in range(0, i):
    #         if DAG[i, j] != DAG[j, i]:
    #             DAG[i, j] = 0
    #             DAG[j, i] = 0

    for i in range(kvar):
        for j in range(0, i):
            if PP[i, j] != PP[j, i]:
                PP[i, j] = 1
                PP[j, i] = 1

    for i in range(kvar):
        for j in range(kvar):
            if PP[i, j] == 1:
                all_MB[i].append(j)

    # removes the possible spouse links between linked variables X and Y
    for x in range(kvar):
        for y in all_MB[x]:
            vs = set(all_MB[x]).union(set(all_MB[y]))
            varis = list((set(all_MB[x]).difference([y])).union(set(all_MB[y]).difference([x])))
            k = 0
            break_flag = False
            while len(varis) > k and k <= max_k:
                ss = subsets(varis, k)
                for s in ss:

                    num_CI += 1
                    pval, _ = cond_indep_test(data, x, y, s, is_discrete)
                    if pval > alpha:
                        PP[x, y] = 0
                        PP[x, y] = 0
                        break_flag = True
                        break
                if break_flag:
                    break
                k += 1


    for i in range(kvar):
        for j in range(kvar):
            if PP[i, j] == 1:
                all_neighbor[i].append(j)

    DAG = PP.copy()
    pdag = DAG.copy()
    G = DAG.copy()

    # orient edges
    for x in range(kvar):
        for y in all_neighbor[x]:
            sz = list((set(all_neighbor[x]).difference(all_neighbor[y])).difference([y]))
            for z in sz:
                PP[y, x] = -1
                B = list((set(all_MB[y]).difference([z])).union(set(all_MB[z]).difference([y])))
                break_flag = False
                cutSetSize = 0
                while len(B) >= cutSetSize and cutSetSize == 0:
                    SS = subsets(B, cutSetSize)
                    for s in SS:
                        cond_s = list(set(s).union([x]))

                        num_CI += 1
                        pval, _ = cond_indep_test(
                            data, y, z, cond_s, is_discrete)
                        if pval > alpha:
                            PP[y, x] = 1
                            break_flag = True
                            break
                    if break_flag:
                        break
                    cutSetSize += 1
            if PP[y, x] == -1:
                pdag[y, x] = -1
                pdag[x, y] = 0
                G[y, x] = 1
                G[x, y] = 0
                break

    DAG, pdag, G = meek(DAG, pdag, G, kvar)

    return pdag, num_CI
