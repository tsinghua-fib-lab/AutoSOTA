# _*_code:utf_8_*_
#!/usr/bin/env python
# date:2019/9/6 20:21

import numpy as np
from ..common.Meek import Meek
from .CMB_subroutine import CMB_subroutine


def CMB(Data, T, alaph, is_discrete=True):
    n, p = np.shape(Data)
    DAG = np.zeros((p, p))
    pdag = np.zeros((p, p))
    G = np.zeros((p, p))
    Tmp = []
    Q = [T]
    all_idT3 = {}
    all_idT3_count = [0] * p
    already_calculated = [1] * p
    already_calculated_MB = [1] * p
    all_MB = {}
    break_flag = False
    num_ci = 0
    # Step 1:establish initial ID
    IDT = np.zeros((p, p))

    # if no element of IDT is equal to 3,break
    while len(Tmp) <= p and len(Q) != 0:
        A = Q[0]
        Q.remove(A)
        if A in Tmp:
            continue
        else:
            Tmp.append(A)
        if already_calculated[A]:
            IDT, all_idT3[A], all_idT3_count[A], pctemp, n_c  = CMB_subroutine(
                Data, A, alaph, IDT, already_calculated_MB, all_MB, is_discrete)
            num_ci += n_c
            already_calculated[A] = 0
        IDT_A_3 = [index for index, i in enumerate(IDT[A]) if i == 3]
        IDT_A_2 = [index for index, i in enumerate(IDT[A]) if i == 2]
        IDT_A_1 = [index for index, i in enumerate(IDT[A]) if i == 1]

        for i in IDT_A_3:
            DAG[A, i] = 1
            DAG[i, A] = 1
        for i in IDT_A_2:
            DAG[A, i] = 1
            DAG[i, A] = 1
        for i in IDT_A_1:
            DAG[A, i] = 1
            DAG[i, A] = 1

        for i in IDT_A_3:
            pdag[A, i] = 1
            pdag[i, A] = 1
        for i in IDT_A_2:
            pdag[A, i] = -1
            pdag[i, A] = 0
        for i in IDT_A_1:
            pdag[A, i] = 0
            pdag[i, A] = -1

        for i in IDT_A_3:
             G[A][i] = 1
             G[i][A] = 1
        for i in IDT_A_2:
            G[A][i] = 1
            G[i][A] = 0
        for i in IDT_A_1:
            G[A][i] = 0
            G[i][A] = 1

        if 1 not in pdag[T] and 1 not in pdag[:, T]:
            break

        # Step 3:resolve variable set with idT=3
        IDT3_count = [index for index, i in enumerate(IDT[A]) if i == 3]
        for z in range(len(IDT3_count)):
            X = IDT3_count[z]
            Q.append(X)
            if already_calculated[X]:
                IDT, all_idT3[X], all_idT3_count[X], pctemp, n_c2 = CMB_subroutine(
                    Data, X, alaph, IDT, already_calculated_MB, all_MB, is_discrete)
                already_calculated[X] = 0
                num_ci += n_c2
            # update IDT according to IDX
            if IDT[X, A] == 2:
                IDT[A, X] = 1
                for j in range(all_idT3_count[X]):
                    if all_idT3[X][j][0] == X:
                        Y = all_idT3[X][j][1]
                        IDT[A, Y] = 2
                    elif all_idT3[X][j][1] == X:
                        Y = all_idT3[X][j][0]
                        IDT[A, Y] = 2
            IDT_X_3 = [index for index, i in enumerate(IDT[X])if i == 3]
            IDT_X_2 = [index for index, i in enumerate(IDT[X]) if i == 2]
            IDT_X_1 = [index for index, i in enumerate(IDT[X]) if i == 1]

            for i in IDT_X_3:
                DAG[X, i] = 1
                DAG[i, X] = 1
            for i in IDT_X_2:
                DAG[X, i] = 1
                DAG[i, X] = 1
            for i in IDT_X_1:
                DAG[X, i] = 1
                DAG[i, X] = 1

            for i in IDT_X_3:
                pdag[X, i] = 1
                pdag[i, X] = 1
            for i in IDT_X_2:
                pdag[X, i] = -1
                pdag[i, X] = 0
            for i in IDT_X_1:
                pdag[X, i] = 0
                pdag[i, X] = -1

            for i in IDT_X_3:
                G[X, i] = 1
                G[i, X] = 1
            for i in IDT_X_2:
                G[X, i] = 1
                G[i, X] = 0
            for i in IDT_X_1:
                G[X, i] = 0
                G[i, X] = 1

            pdag = Meek(DAG, pdag, Data)
            if 1 not in pdag[T] and 1 not in pdag[:, T]:
                break_flag = 1
                break
        if break_flag:
            break

    parents = [i for i in range(p)if pdag[i, T] == -1]
    children = [j for j in range(p)if pdag[T, j] == -1]
    undirected = [i for i in range(p)if pdag[T, i] == 1]
    PC = list(set(parents).union(set(children)).union(set(undirected)))
    return parents, children, PC, undirected, num_ci


# import pandas as pd
# Data=pd.read_csv("D:/pyCausalFS/LSL/data/Child_s500_v1.csv")
# print("the file read")
# P, C, Undirected = CMB(Data, 13, 0.01)
# print(P)
# print(C)
# print(Undirected)
