# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/26 14:31
desc:
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets


def IAMBnPC(data, target, alaph, is_discrete=True):
    CMB = []
    ci_number = 0
    number, kVar = np.shape(data)

    while True:
        variDepSet = []
        Svariables = [i for i in range(kVar) if i != target and i not in CMB]
        # print(Svariables)
        for x in Svariables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, CMB, is_discrete)
            # print("pval: " + str(pval))
            if pval <= alaph:
                variDepSet.append([x, dep])
        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
        # print(variDepSet)
        if variDepSet == []:
            break
        else:
            CMB.append(variDepSet[0][0])
            # print(CMB)

    """shrinking phase"""
    TestMB = CMB.copy()
    # whether or not sorted TestMB  is not influence,just for elegant!
    TestMB = sorted(TestMB)
    p = len(TestMB)
    DAG = np.ones((1, p))
    size = 0
    continueFlag = True
    # conditionSet maximum set 3
    max_k = 3
    # target_index = TestMB.index(target)
    while continueFlag:
        # Candidate of MBs traverse
        for y in range(p):
            if DAG[0, y] == 0:
                continue
            conditionAllSet = [i for i in range(
                p) if i != y and DAG[0, i] == 1]
            conditionSet = subsets(conditionAllSet, size)
            for S in conditionSet:
                condtionVari = [TestMB[i] for i in S]
                ci_number += 1
                pval_sp, _ = cond_indep_test(
                    data, target, TestMB[y], condtionVari, is_discrete)
                if pval_sp >= alaph:
                    DAG[0, y] = 0
                    # print("pDAG: \n" + str(DAG))
                    break
        # print("test: \n" + str(DAG))
        size += 1
        continueFlag = False

        # circulate will be continue if condition suited
        if np.sum(DAG[0, :] == 1) >= size and size <= max_k:
            continueFlag = True
    # end while

    # print("DAG is: \n" + str(DAG))
    MB = [TestMB[i] for i in range(p) if DAG[0, i] == 1]

    return MB, ci_number


# data = pd.read_csv(
# )
# print("the file read")
#
# target = 11
# alaph = 0.05
#
# MBs = interIAMBnPC(data,target,alaph)
# print("MBs is: "+str(MBs))


# F1 is: 0.8206423576423579
# Precision is: 0.9254166666666666
# Recall is: 0.7850833333333331
# time is: 21.96171875


# 5000

# F1 is: 0.93
# Precision is: 0.99
# Recall is: 0.88
# Distance is: 0.12
# ci_number is: 125.915
# time is: 73.69
