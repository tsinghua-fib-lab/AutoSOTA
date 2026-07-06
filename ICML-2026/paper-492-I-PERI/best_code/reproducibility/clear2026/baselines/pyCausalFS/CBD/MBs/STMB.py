# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/20 10:46
desc:
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets
from CBD.MBs.IPCMB.RecognizePC import RecognizePC


def STMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    ci_number = 0
    PCT = [i for i in range(kVar) if i != target]
    PCT, sepset, ci_num2 = RecognizePC(data, target, PCT, alaph, is_discrete)
    ci_number += ci_num2

    spouse = [[] for i in range(kVar)]
    remove = []
    for y in PCT:
        X_set = [i for i in range(kVar) if i != target and i not in PCT]
        # print("y: " + str(y) + " ,X_set is:" + str(X_set))
        breakFlag = False
        for x in X_set:
            conditionsSet = [i for i in sepset[x]]
            conditionsSet.append(y)
            conditionsSet = list(set(conditionsSet))

            ci_number += 1
            pval_xt, dep_xt = cond_indep_test(
                data, target, x, conditionsSet, is_discrete)
            # print("x is: " + str(x) + " conditionSet is: " + str(conditionsSet) + "pval_xt is: " + str(pval_xt))
            if pval_xt <= alaph:
                Zset = [i for i in PCT]
                Zset.append(x)
                Zset = list(set(Zset))
                if y in Zset:
                    Zset.remove(y)

                if len(Zset) >= 3:
                    Zlength = 3
                else:
                    Zlength = len(Zset)
                # Zlength +1 is important!
                for j in range(Zlength + 1):
                    Zsubsets = subsets(Zset, j)
                    for Z in Zsubsets:
                        ci_number += 1
                        pval_yt, dep_yt = cond_indep_test(
                            data, target, y, Z, is_discrete)
                        if pval_yt > alaph:
                            # print("remove append is: " + str(y))
                            remove.append(y)
                            breakFlag = True
                            break
                    if breakFlag:
                        break
                if breakFlag:
                    break
                else:
                    spouse[y].append(x)

    PCT = [i for i in PCT if i not in remove]

    for y in range(len(spouse)):
        if spouse[y] != []:
            spouseY_temp = spouse[y].copy()
            for x in spouseY_temp:
                testSet = [
                    i for i in range(kVar) if i in PCT or i in spouse[y]]
                testSet = list(set(testSet))
                # print("testSet has: " + str(testSet))
                if x in testSet:
                    testSet.remove(x)

                ci_number += 1
                pval_xt_testset, _ = cond_indep_test(
                    data, target, x, testSet, is_discrete)
                if pval_xt_testset > alaph:
                    # print("spouse[y] had: " + str(spouse[y]))
                    spouse[y].remove(x)
                    # print("spouse[y] now has: " + str(spouse[y]))

    M_variSet = PCT.copy()
    # print("M_variSet is:" + str(M_variSet))
    for x in M_variSet:
        conditionsVariSet = [i for j in range(len(spouse)) for i in spouse[j]]
        conditionsVariSet = list(set(conditionsVariSet).union(set(PCT)))
        # print("conditionsVariSet is: " + str(conditionsVariSet))
        if x in conditionsVariSet:
            conditionsVariSet.remove(x)

        ci_number += 1
        pval_final, _ = cond_indep_test(
            data, target, x, conditionsVariSet, is_discrete)
        if pval_final > alaph:
            PCT.remove(x)

    spouse = [i for j in range(len(spouse)) for i in spouse[j]]
    MB = list(set(PCT).union(set(spouse)))

    return MB, ci_number


# import  pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v3.csv")
# print("the file read")
#
# target = 11
# alaph = 0.05
#
# MBs=STMB(data, target, alaph, is_discrete=False)
# print("MBs is: "+str(MBs))

# F1 is: 0.7526467421467425
# Precision is: 0.8019166666666667
# Recall is: 0.7789583333333334
# time is: 11.730078125

# 5000

# F1 is: 0.86
# Precision is: 0.86
# Recall is: 0.87
# Distance is: 0.21
# ci_number is: 142.295
# time is: 70.24
