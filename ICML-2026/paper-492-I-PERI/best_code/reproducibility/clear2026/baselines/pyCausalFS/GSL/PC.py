#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2020/7/24 10:27
 @File    : pc1.py
 """


import time
import numpy as np
from ..CBD.MBs.common.subsets import subsets
from ..CBD.MBs.common.Meek import meek
from ..CBD.MBs.common.condition_independence_test import cond_indep_test
# The output P is an adjacency matrix, in which
# P(i,j) = -1 if there is an i->j edge.
# P(i,j) = P(j,i) = 1 if there is an undirected edge i <-> j


def pc(Data, alpha, is_discrete : bool = False):
    time_start = time.time()


    ind_test = 0  # the number of condition independency test
    Num, NbVar = Data.shape

    sepset = [[[]] * NbVar for i in range(NbVar)]

    DAG = np.ones((NbVar, NbVar))
    for i in range(NbVar):
        DAG[i][i] = 0

    # stage 1: construct skeletons
    n = 0
    done = False
    while not done:
        done = True
        for x in range(NbVar):
            adjx = [i for i in range(NbVar) if DAG[x, i] == 1]
            if len(adjx) >= n:
                done = False
                for y in adjx:
                    cx_y = [i for i in adjx if i != y]
                    SS  = subsets(cx_y,n)
                    # cx_y = my_set_diff_two(adjx, y)  # Adj(c,x)\{y}
                    # SS = list(combinations(cx_y, n))
                    for S in SS:
                        #sub_data_script = [x, y] + list(map(int, S))
                        #xyz_data = Data[:, sub_data_script]  # (X, target, subset)
                        #pval = mi_test(xyz_data)  # use MI to test conditional independence
                        # _, pval, _, _ = chi_square_test(Data, x, y, list(map(int, S)))
                        pval, _ = cond_indep_test(
                            Data, x, y, list(map(int, S)), is_discrete=is_discrete)
                        ind_test = ind_test + 1
                        if pval > alpha:
                            DAG[x, y] = 0
                            DAG[y, x] = 0
                            if list(map(int, S)) not in sepset[x][y]:
                                sepset[x][y].append(list(map(int, S)))
                            if list(map(int, S)) not in sepset[y][x]:
                                sepset[y][x].append(list(map(int, S)))
                            break
        n += 1

    # stage 2: create V structures orient X-Y-Z => X -> Y <- Z
    print('stage 2')
    pDAG = DAG.copy()

    gtmp = DAG.copy()

    X = [i for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
    Y = [j for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        Z = [j for j in range(NbVar) if DAG[y, j] == 1 and j != x]

        for z in Z:
            if DAG[x, z] == 0 and [y] not in sepset[x][z]:
                pDAG[x, y] = -1; pDAG[y, x] = 0
                pDAG[z, y] = -1; pDAG[y, z] = 0

                gtmp[x, y] = 1; gtmp[y, x] = 0
                gtmp[z, y] = 1; gtmp[y, z] = 0
    # stage 3:edge oriented
    print('stage 3')
    old_pDAG = np.zeros((NbVar, NbVar))
    iter = 0
    while not (pDAG == old_pDAG).all():
        iter += 1
        old_pDAG = pDAG
        # rule 1: A->B--C ==>B->C
        A = [i for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == -1]
        B = [j for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == -1]
        for i in range(len(A)):
            a = A[i]; b = B[i]
            C = [j for j in range(NbVar) if pDAG[b][j] == 1 and pDAG[a][j] == 0]
            for c in C:
                pDAG[b][c] = -1; pDAG[c][b] = 0
                gtmp[b][c] = 1; gtmp[c][b] = 0

        # rule 2: A->C->B,A--B=>A->B
        A = [i for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
        B = [j for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
        for i in range(len(A)):
            a = A[i]; b = B[i]
            if np.any(np.multiply(np.array(pDAG[a, :] == -1), np.array(pDAG[:, b] == -1))):
                pDAG[a][b] = -1; pDAG[b][a] = 0
                gtmp[a][b] = 1; gtmp[b][a] = 0

        # rule 3:     % a--c->b, a--d->b, pDAG(c,d)=pDAG(d,c)=0, a--b  => a->b
        A = [i for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
        B = [j for i in range(NbVar) for j in range(NbVar) if DAG[i, j] == 1]
        for i in range(len(A)):
            a = A[i]; b = B[i]
            C = [j for j in range(NbVar) if pDAG[a][j] == 1 and pDAG[j][b] == -1]
            for c in C:
                for d in C:
                    if pDAG[c][d] == 0 and c != d:
                        pDAG[a][b] = -1; pDAG[b][a] = 0
                        gtmp[a][b] = 1; gtmp[b][a] = 0
                        break

    time_end = time.time()
    time_cost = time_end - time_start
    print('running time is:', time_cost, 's')
    return pDAG, ind_test










'''
matrix = [[[]]*5 for i in range(5)]

print(matrix[4][1])

s=[4,5]
t=[1,2,3]
s2=[4,5]
if s not in matrix[4][1]:
    matrix[4][1].append(s)
if t not in matrix[4][1]:
    matrix[4][1].append(t)
if s2 not in matrix[4][1]:
    matrix[4][1].append(s2)
print(matrix[4][1])

SS = list(combinations([1,2,3,4,5,6,8], 2))
print(list(map(int, SS[0])))

print([1])

x = np.array([[1, 1, -1, 0, -1],[0, -1, -1, 0, -1], [1, 0, 0, 0, 1], [1, 1, 1, -1, 1], [-1, 0, 0, 1, -1]])

print(x([1,2,3],[1,2,3]))

print(pc(x, 0.01))
print(np.array(x[:, 3] == -1))
print(np.array(x[2, :] == -1))
print(np.any(np.multiply(np.array(x[:, 3] == -1), np.array(x[2, :] == -1))))
'''

# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# data = pd.read_csv("D:/data/Alarm_data/Alarm1_s5000_v2.csv")
# print("the file read")
# num1, kvar = np.shape(data)
# alaph = 0.01
# dic = {}
# dic.setdefault("cache", [0, 0])
# DAG, _, dic = pc(data, alaph, dic)
# print("DAG:\n ", DAG)
# for i in range(kvar):
#     for j in range(kvar):
#         if DAG[i, j] == -1:
#             print(i, " -child: ", j)
# print(dic["cache"][0] / (dic["cache"][0] + dic["cache"][1]))