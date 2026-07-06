#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2020/8/7 10:54
 @File    : MBbyMB_new.py
 """



import numpy as np
from baselines.pyCausalFS.CBD.MBs.MMMB.MMMB import MMMB
from .common.condition_independence_test import cond_indep_test
from .common.Meek import meek
from .common.subsets import subsets
import math


def MBbyMB(data, target, alpha, is_discrete=True):

    num_ci = 0
    max_k = 3
    _, kvar = np.shape(data)
    DAG = np.zeros((kvar, kvar))
    pdag = DAG.copy()
    G = DAG.copy()
    mb_calcualted = [True for i in range(kvar)]
    all_pc = [[] for i in range(kvar)]
    all_mb = [[] for i in range(kvar)]
    all_can_spouse = [[] for i in range(kvar)]
    all_sepset = [[[]] * kvar for i in range(kvar)]
    Q = [target]
    tmp = []

    num_calculated = 0

    while len(tmp) <= kvar and len(Q) > 0:
        A = Q[0]
        del Q[0]
        if A in tmp:
            continue
        else:
            tmp.append(A)

        # get MB(A)
        if mb_calcualted[A]:
            all_mb[A], ntest = MMMB(data, A, alpha, is_discrete)
            num_ci += ntest
            mb_calcualted[A] = False

        all_pc[A] = all_mb[A].copy()

        for B in all_mb[A]:
            Q.append(B)
            DAG[A, B] = 1
            DAG[B, A] = 1
            if pdag[A, B] == 0 and pdag[B, A] == 0:
                pdag[A, B] = 1
                pdag[B, A] = 1
                G[A, B] = 1
                G[B, A] = 1

            cutSetSize = 0
            break_flag = False
            can_pc = [i for i in all_mb[A] if i != B]
            while len(can_pc) >= cutSetSize and cutSetSize <= max_k:
                SS = subsets(can_pc, cutSetSize)
                for z in SS:
                    num_ci += 1
                    pval, _ = cond_indep_test(data, B, A, z, is_discrete)

                    if pval > alpha:
                        all_sepset[A][B] = [i for i in z]
                        all_sepset[B][A] = [i for i in z]

                        DAG[A, B] = 0
                        DAG[B, A] = 0
                        pdag[A, B] = 0
                        pdag[B, A] = 0
                        G[A, B] = 0
                        G[B, A] = 0

                        all_pc[A] = [i for i in all_pc[A] if i != B]
                        all_can_spouse[A].append(B)

                        break_flag = True
                        break
                if break_flag:
                    break
                cutSetSize += 1
        # print("all_sepset: ", all_sepset)
        # find v-structures
        for C in all_can_spouse[A]:
            for B in all_pc[A]:

                # A->B<-C
                if B not in all_sepset[A][C]:
                    DAG[A, B] = 1
                    DAG[B, A] = 1

                    pdag[A, B] = -1
                    pdag[B, A] = 0

                    pdag[C, B] = -1
                    pdag[B, C] = 0

                    G[A, B] = 1
                    G[B, A] = 0

                    G[C, B] = 1
                    G[B, C] = 0

        [DAG, pdag, G] = meek(DAG, pdag, G, kvar)

        num_calculated += 1
        if num_calculated > len(all_mb[target]):
            if 1 not in pdag[target, :] and 1 not in pdag[:, target]:
                break

    parents = [i for i in range(kvar) if pdag[i, target] == -1]
    children = [i for i in range(kvar) if pdag[target, i] == -1]
    undirected = [i for i in range(kvar) if pdag[target, i] == 1]
    PC = list(set(parents).union(set(children)).union(set(undirected)))

    return parents, children, PC, undirected, num_ci


# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# data = pd.read_csv("D:/data/alarm_data/Alarm1_s5000_v6.csv")
# print("the file read")
# import numpy as np
# num1, kvar = np.shape(data)
# alaph = 0.01
#
# for target in range(kvar):
#     P, C, PC, und = MBbyMB(data, target, alaph, True)
#     print(target," -P: ", P, " ,C: ", C, " ,PC: ", PC, " ,undire: ",und)