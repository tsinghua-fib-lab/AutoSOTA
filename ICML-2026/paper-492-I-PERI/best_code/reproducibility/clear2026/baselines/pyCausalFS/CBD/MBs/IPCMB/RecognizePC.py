# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/21 9:42
desc:
"""
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets
import numpy as np


def RecognizePC(data, target, ADJT, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    NonPC = []
    cutSetSize = 0
    sepset = [[] for i in range(kVar)]
    ci_number = 0
    while len(ADJT) > cutSetSize:
        for x in ADJT:
            ADJT_X = [i for i in ADJT if i != x]
            SSubset = subsets(ADJT_X, cutSetSize)
            for S in SSubset:
                ci_number += 1
                pval_gp, dep_gp = cond_indep_test(
                    data, target, x, S, is_discrete)
                if pval_gp > alaph:
                    NonPC.append(x)
                    sepset[x] = [i for i in S]
                    break
        if len(NonPC) > 0:
            ADJT = [i for i in ADJT if i not in NonPC]
            cutSetSize += 1
            NonPC = []
        else:
            break

    return ADJT, sepset, ci_number

# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v3.csv")
# print("the file read")
# _,k = np.shape(data)
# target = 12
# alaph = 0.05
# ADJT = [i for i in range(k) if i != target]
# MBs,sepset=RecognizePC(data,target,ADJT,alaph)
# print("MBs is: "+str(MBs))
# print(sepset)
