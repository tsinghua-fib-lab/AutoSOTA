# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/17 17:08
desc:
"""

import numpy as np

from baselines.pyCausalFS.CBD.MBs.common.condition_independence_test import cond_indep_test
from baselines.pyCausalFS.CBD.MBs.common.subsets import subsets


def HITON_PC(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    sepset = [[] for i in range(kVar)]
    variDepSet = []
    candidate_PC = []
    PC = []
    ci_number = 0
    noAdmissionSet = []
    max_k = 3

    # use a list to store variables which are not condition independence with
    # target,and sorted by dep max to min
    candidate_Vars = [i for i in range(kVar) if i != target]
    for x in candidate_Vars:
        ci_number += 1
        pval_gp, dep_gp = cond_indep_test(
            data, target, x, [], is_discrete)
        if pval_gp <= alaph:
            variDepSet.append([x, dep_gp])

    # sorted by dep from max to min
    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    # print(variDepSet)

    # get number by dep from max to min
    for i in range(len(variDepSet)):
        candidate_PC.append(variDepSet[i][0])
    # print(candidate_PC)

    """ sp """
    for x in candidate_PC:

        PC.append(x)
        PC_index = len(PC)
        # if new x add will be removed ,test will not be continue,so break the
        # following circulate to save time ,but i don't not why other index
        # improve
        breakFlagTwo = False

        while PC_index >= 0:
            #  reverse traversal PC,and use PC_index as a pointer of PC
            PC_index -= 1
            y = PC[PC_index]
            breakFlag = False
            conditions_Set = [i for i in PC if i != y]

            if len(conditions_Set) >= max_k:
                Slength = max_k
            else:
                Slength = len(conditions_Set)

            for j in range(Slength + 1):
                SS = subsets(conditions_Set, j)
                for s in SS:
                    ci_number += 1
                    conditions_test_set = [i for i in s]
                    pval_rm, dep_rm = cond_indep_test(
                        data, target, y, conditions_test_set, is_discrete)
                    if pval_rm > alaph:
                        sepset[y] = [i for i in conditions_test_set]
                        # if new x add will be removed ,test will not be
                        # continue
                        if y == x:
                            breakFlagTwo = True
                        PC.remove(y)
                        breakFlag = True
                        break

                if breakFlag:
                    break
            if breakFlagTwo:
                break

    return list(set(PC)), sepset, ci_number


# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 1
# alaph = 0.05
#
# MBs,sepset,_=HITON_PC(data,target,alaph)
# print("MBs is: "+str(MBs))
# print(sepset)
