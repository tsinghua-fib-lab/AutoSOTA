#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/9/24 9:09
 @File    : PCDbyPCD.py
 """

import numpy as np
from CBD.MBs.MMMB.MMPC import MMPC
from LSL.MBs.common.Meek import Meek


def PCDbyPCD(data, target, alaph, is_discrete=True):
    _, kVar = np.shape(data)
    DAG = np.zeros((kVar, kVar))
    pDAG = DAG.copy()
    G = DAG.copy()
    sepset_all = [[[]]] * kVar
    PCD_set_all = [[]] * kVar
    tmp = []
    Q = [target]
    parents = []
    children = []
    undirected = []
    lnum = 0
    num_ci = 0
    while len(tmp) <= kVar and Q != []:
        A = Q[0]
        # print("A is: " + str(A))
        del Q[0]
        if A in tmp:
            continue
        else:
            tmp.append(A)

        #Get PC(A)
        if PCD_set_all[A] == []:
            PCD_set_all[A], sepset_all[A], n_c = MMPC(data, A, alaph, is_discrete)
            num_ci += n_c
        for B in PCD_set_all[A]:
            # print("B is: " + str(B))
            Q.append(B)
            # if PCD_set_all[B] == []:
            #      PCD_set_all[B], sepset_all[B], _ = MMPC(data, B, alaph)
            if A not in PCD_set_all[B]:
                continue

            DAG[A, B] = 1
            DAG[B, A] = 1

            if pDAG[A, B] == 0 and pDAG[B, A] == 0:
                pDAG[A, B] = 1
                pDAG[B, A] = 1
                G[A, B] = 1
                G[B, A] = 1

            for C in PCD_set_all[B]:
                # if PCD_set_all[C] == []:
                #     PCD_set_all[C], sepset_all[C], _ = MMPC(data, C, alaph)

                # if B not in PCD_set_all[C]:
                #      continue
                #
                # DAG[C, B] = 1
                # DAG[B, C] = 1
                #
                # if pDAG[C, B] == 0 and pDAG[B, C] == 0:
                #     pDAG[C, B] = 1
                #     pDAG[B, C] = 1
                #     G[C, B] = 1
                #     G[B, C] = 1

                if C in PCD_set_all[A] or C == A:
                    continue

                # v-structure
                if DAG[C, B] == 1 and DAG[B, C] == 1:
                    if B not in sepset_all[A][C]:
                        pDAG[A, B] = -1
                        pDAG[B, A] = 0
                        pDAG[C, B] = -1
                        pDAG[B, C] = 0

                        G[A, B] = 1
                        G[B, A] = 0
                        G[C, B] = 1
                        G[B, C] = 0

        pDAG = Meek(DAG, pDAG, data)

        lnum += 1
        length_PCD_set = 0
        for i in range(kVar):
            if PCD_set_all[i] != []:
                length_PCD_set += 1
        # break condition
        if lnum > length_PCD_set:
            if np.all(pDAG[:, target] != 1) and np.all(pDAG[target, :] != 1):
                # print("break")
                break
    # print(pDAG)
    for i in range(kVar):
        if pDAG[i, target] == -1:
            parents.append(i)
        if pDAG[target, i] == -1:
            children.append(i)
        if pDAG[target, i] == 1:
            undirected.append(i)
    PC = list(set(parents).union(set(children)).union(set(undirected)))
    return parents, children, PC, undirected, num_ci

# import pandas as pd
# data = pd.read_csv("D:/pyCausalFS/LSL/data/Child_s5000_v1.csv")
# print("the file read")
#
# target = 1
# alaph = 0.01
#
# parents, children, undirected = PCDbyPCD(data,  target, alaph)
# print("\nparents is: " + str(parents))
# print("children is: " + str(children))
# print("undirected is: " + str(undirected))

