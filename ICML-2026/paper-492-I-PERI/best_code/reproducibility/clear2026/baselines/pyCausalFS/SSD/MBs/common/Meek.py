# _*_code:utf_8_*_
#!/usr/bin/env python
# date:2019/9/10 20:18
import numpy as np


def Meek(DAG, pdag, Data):
    n, p = np.shape(Data)
    old_pdag = np.zeros((p, p))

    while not (pdag == old_pdag).all():
        old_pdag = pdag.copy()
        # rule 1
        X = [i for i in range(p) for j in range(p) if pdag[i, j] == -1]
        Y = [i for i in range(p) for j in range(p) if pdag[i, j] == -1]
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            Z = [j for j in range(p) if pdag[y, j] == 1 and DAG[x, j] == 0]
            for z in Z:
                pdag[y, z] = -1
                pdag[z, y] = 0
                DAG[y, z] = -1
                DAG[z, y] = -1

        # rule 2
        X = [i for i in range(p) for j in range(p)if pdag[i, j] == 1]
        Y = [j for i in range(p)for j in range(p)if pdag[i, j] == 1]
        if len(X) == 0:
            break
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
        if np.any(np.multiply(
                np.array(pdag[x, :] == -1), np.array(pdag[:, y] == -1))):
            pdag[x, y] = -1
            pdag[y, x] = 0
            DAG[x, y] = -1
            DAG[y, x] = -1

        # rule 3
        X = [i for i in range(p) for j in range(p)if pdag[i, j] == 1]
        Y = [j for i in range(p)for j in range(p)if pdag[i, j] == 1]
        if len(X) == 0:
            break
        Z = [j for j in range(p)if pdag[x, j] == 1 and pdag[j, y] == -1]
        for i in Z:
            for j in Z:
                if i != j:
                    pdag[x, y] = -1
                    pdag[y, x] = 0
                    DAG[x, y] = -1
                    DAG[y, x] = -1

    return pdag


def meek(DAG, pdag, G, p):
    old_pdag = np.zeros((p, p))

    while not (pdag == old_pdag).all():
        old_pdag = pdag
        # rule 1: a->b-c ===>   a->b->c
        A = [i for i in range(p) for j in range(p) if pdag[i, j] == -1]
        B = [j for i in range(p) for j in range(p) if pdag[i, j] == -1]
        for i in range(len(A)):
            a = A[i]
            b = B[i]
            C = [j for j in range(p) if pdag[b, j] == 1 and DAG[a, j] == 0]
            for c in C:
                if DAG[a, c] == 0:
                    pdag[b, c] = -1
                    pdag[c, b] = 0
                    G[b, c] = -1
                    G[c, b] = -1

        # rule 2: a-->c-->b ,a-b===> a-->b
        A = [i for i in range(p) for j in range(p) if pdag[i, j] == 1]
        B = [j for i in range(p) for j in range(p) if pdag[i, j] == 1]
        if A == []:
            break
        for i in range(len(A)):
            a = A[i]
            b = B[i]
            if np.any(np.multiply(
                    np.array(pdag[a, :] == -1), np.array(pdag[:, b] == -1))):
                pdag[a, b] = -1
                pdag[b, a] = 0
                G[a, b] = -1
                G[b, a] = -1

        # rule 3:a-b,a - c -> b,a--d -> b ===> a->b
        for i in range(len(A)):
            a = A[i]
            b = B[i]
            C = [m for m in range(p) if pdag[m, b] == -1 and pdag[a, m] == 1]
            for c in C:
                for d in C:
                    if c != d and pdag[c, d] == 0 and pdag[d, c] == 0:
                        pdag[a, b] = -1
                        pdag[b, a] = 0
                        G[a, b] = -1
                        G[b, a] = -1

    return DAG, pdag, G

# C:\pythonProject\pyCausalFS\LSL\data\Child_s500_v1.csv
