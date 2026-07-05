# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/19 21:37
desc: 
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets


def getPCD(data, target, alaph, is_discrete):
    number, kVar = np.shape(data)
    max_k = 3
    PCD = []
    ci_number = 0

    # use a list of sepset[] to store a condition set which can make target and the variable condition independence
    # the above-mentioned variable will be remove from CanPCD or PCD
    sepset = [[] for i in range(kVar)]

    while True:
        variDepSet = []
        CanPCD = [i for i in range(kVar) if i != target and i not in PCD]
        CanPCD_temp = CanPCD.copy()

        for vari in CanPCD_temp:
            breakFlag = False
            dep_gp_min = float("inf")
            vari_min = -1

            if len(PCD) >= max_k:
                Plength = max_k
            else:
                Plength = len(PCD)

            for j in range(Plength+1):
                SSubsets = subsets(PCD, j)
                for S in SSubsets:
                    ci_number += 1
                    pval_gp, dep_gp = cond_indep_test(data, target, vari, S, is_discrete)

                    if pval_gp > alaph:
                        vari_min = -1
                        CanPCD.remove(vari)
                        sepset[vari] = [i for i in S]
                        breakFlag = True
                        break
                    elif dep_gp < dep_gp_min:
                        dep_gp_min = dep_gp
                        vari_min = vari

                if breakFlag:
                    break

            # use a list of variDepset to store list, like [variable, its dep]
            if vari_min in CanPCD:
                variDepSet.append([vari_min, dep_gp_min])

        # sort list of variDepSet by dep from max to min
        variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)

        # if variDepset is null ,that meaning PCD will not change
        if variDepSet != []:
            y =variDepSet[0][0]
            PCD.append(y)
            pcd_index = len(PCD)
            breakALLflag = False
            while pcd_index >=0:
                pcd_index -= 1
                x = PCD[pcd_index]
                breakFlagTwo = False

                conditionSetALL = [i for i in PCD if i != x]
                if len(conditionSetALL) >= max_k:
                    Slength = max_k
                else:
                    Slength = len(conditionSetALL)

                for j in range(Slength+1):
                    SSubsets = subsets(conditionSetALL, j)
                    for S in SSubsets:
                        ci_number += 1
                        pval_sp, dep_sp = cond_indep_test(data, target, x, S, is_discrete)

                        if pval_sp > alaph:

                            PCD.remove(x)
                            if x == y:
                                breakALLflag = True

                            sepset[x] = [i for i in S]
                            breakFlagTwo = True
                            break
                    if breakFlagTwo:
                        break

                if breakALLflag:
                    break
        else:
            break
    return list(set(PCD)), sepset, ci_number


def getPC(data, target, alaph, is_discrete):
    ci_number = 0
    PC = []
    PCD, sepset, ci_num2 = getPCD(data, target, alaph, is_discrete)
    ci_number += ci_num2
    for x in PCD:
        variSet, _, ci_num3 = getPCD(data, x, alaph, is_discrete)
        ci_number += ci_num3
        # PC of target ,whose PC also has the target, must be True PC
        if target in variSet:
            PC.append(x)

    return list(set(PC)), sepset, ci_number