#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/21 9:30
 @File    : FBEDk.py
 """
import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test

def one_run(data, target, S, alaph, is_discrete):
    ci_number = 0
    number, kVar = np.shape(data)
    R = [i for i in range(kVar) if i not in S and i != target]
    while len(R) > 0:
        vari_dep_set = []
        for x in R:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, S, is_discrete)
            # print("x is: " + str(x) + " ,S is: " + str(S) + " ,pval is: " + str(pval) + " ,dep is: " + str(dep))
            if pval <= alaph:
                vari_dep_set.append([x, dep])
        vari_dep_set = sorted(vari_dep_set, key=lambda x:x[1], reverse=True)
        # print("varidepset have: "  + str(vari_dep_set))
        if vari_dep_set != []:
            S.append(vari_dep_set[0][0])
            # print("S have: " + str(S))
            del vari_dep_set[0]
            R = [vari_dep_set[i][0] for i in range(len(vari_dep_set)) ]
            # print("R have: " + str(R))
        else:
            R = []
    return S, ci_number


def FBED(data, target, k, alaph, is_discrete=True):
    S = []
    k_cur = 0
    s_change_flag = True
    ci_number = 0

    # Forward phase
    while k_cur <= k and s_change_flag == True:
        S_last = S.copy()
        S, ci_num = one_run(data, target, S, alaph, is_discrete)
        k_cur += 1
        ci_number += ci_num

        if set(S_last) == set(S):
            s_change_flag = False

    # Backward phase
    # print("now S have: " + str(S))
    S_temp = S.copy()
    for x in S_temp:
        condition_set = [i for i in S if i != x]
        ci_number += 1
        pval, _ = cond_indep_test(data, target, x, condition_set, is_discrete)
        # print("x is: " + str(x) + " ,conditionset is:" + str(condition_set))
        if pval > alaph:
            S.remove(x)

    return list(set(S)), ci_number


# import pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 19
# k=1
# alaph = 0.01
#
# MB, ci_number = FBED(data, target, k, alaph)
# print("MBs is: "+ str(MB) + ", " + str(ci_number))

# 500
#
# F1 is: 0.79
# Precision is: 0.89
# Recall is: 0.78
# Distance is: 0.30
# ci_number is: 50.27
# time is: 7.73

# 5000
#
# F1 is: 0.90
# Precision is: 0.90
# Recall is: 0.94
# Distance is: 0.15
# ci_number is: 63.38
# time is: 28.76

