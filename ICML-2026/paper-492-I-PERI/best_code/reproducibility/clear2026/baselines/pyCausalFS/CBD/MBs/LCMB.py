#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/29 17:22
 @File    : LRH.py
 """
import numpy as np
from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.condition_independence_test import cond_indep_test


def LRH(data, target, alaph, is_discrete=True):
    ci_number = 0
    number, kVar = np.shape(data)
    max_k = 3
    M = []
    while True:
        # selection
        M1 = []
        x_dep_set = []
        variables = [i for i in range(kVar) if i != target and i not in M]
        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, M, is_discrete)
            if pval <= alaph:
                M1.append(x)
                x_dep_set.append([x,dep])

        # exclusion
        if M1 == []:
            break
        elif len(M1) == 1:
            M.append(M1[0])
            continue
        M2 = []
        # print("M is: " + str(M))
        # print("M1 is: " + str(M1))
        for x in M1:
            # print("x is: " + str(x))
            NX = []
            vari_set = [i for i in M1 if i != x]
            for y in vari_set:
                ci_number += 1
                pval, _ = cond_indep_test(data, x, y, M, is_discrete)
                if pval <= alaph:
                    NX.append(y)
            # print("NX is:" + str(NX))
            Nlength = len(NX)
            if Nlength > max_k:
                Nlength = 3
            break_flag = False
            for j in range(Nlength+1):
                Z_set = subsets(NX, j)
                for Z in Z_set:
                    conditionset = list(set(Z).union(set(M)))
                    ci_number += 1
                    pval, _ = cond_indep_test(data, target, x, conditionset, is_discrete)
                    # print("pval is: " + str(pval) + " ,x is: " + str(x) + " ,conditionset is: " + str(conditionset))
                    if pval > alaph:
                        break_flag = True
                        break
                if break_flag:
                    break
            if not break_flag:
                M2.append(x)
                # print("M2 append is: " + str(M2))
        # print("M2 is: " + str(M2))
        Y = []

        if M2 == []:
            x_dep_set = sorted(x_dep_set, key=lambda x: x[1], reverse=True)
            # print("-x_dep_set is: " + str(x_dep_set))
            if x_dep_set != []:
                dep_max = x_dep_set[0][1]
                for m in x_dep_set:
                    if m[1] == dep_max:
                        Y.append(m[0])
                    else:
                        break
        else:
            x_dep_set = []
            for x in M2:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, x, M, is_discrete)
                if pval <= alaph:
                    x_dep_set.append([x, dep])
            x_dep_set = sorted(x_dep_set, key=lambda x: x[1], reverse=True)
            # print("--x_dep_set is: " + str(x_dep_set))
            if x_dep_set != []:
                dep_max = x_dep_set[0][1]
                for m in x_dep_set:
                    if m[1] == dep_max:
                        Y.append(m[0])
                    else:
                        break

        # M3 = [i for i in M1 if i not in M2]
        M = list(set(M).union(set(Y)))

    # print("-M is: " + str(M))
    M_temp = M.copy()
    for x in M_temp:
        conditionset = [i for i in M if i != x]
        ci_number += 1
        pval, _ = cond_indep_test(data, target, x, conditionset, is_discrete)
        # print("pval is: " + str(pval) + " , x is: " + str(x))
        if pval > alaph:
            M.remove(x)

    return M, ci_number



# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 19
# alaph = 0.01
#
# MB = LRH(data, target, alaph)
# print("MBs is: " + str(MB))

# 500

# F1 is: 0.76
# Precision is: 0.85
# Recall is: 0.76
# Distance is: 0.34
# ci_number is: 331.96
# time is: 43.03

# 5000

# F1 is: 0.91
# Precision is: 0.90
# Recall is: 0.94
# Distance is: 0.14
# ci_number is: 1.00
# time is: 238.92