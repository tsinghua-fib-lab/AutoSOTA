# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/12 17:04
desc: 
"""

import numpy as np
from baselines.pyCausalFS.CBD.MBs.common.condition_independence_test import cond_indep_test
from baselines.pyCausalFS.CBD.MBs.common.subsets import subsets


def getMinDep(data, target, x, CPC, alpha, is_discrete):

    """this function is to chose min dep(association) about Target,x|(subsets of CPC)"""

    ci_number = 0
    dep_min = float("inf")
    max_k = 3
    # 在这图中很少一个节点的Perents或child(其中一个)超过三个,即最多图中a->b,c,d->z,所以最多条件集三个(a,z)|(b,c,d)
    # 便可测试出(a,z)是否独立,可极大得减少时间复杂度
    if len(CPC) > max_k:
        k_length = max_k
    else:
        k_length = len(CPC)
    for i in range(k_length+1):
        SS = subsets(CPC, i)
        for S in SS:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, S, is_discrete)
            # this judge about target and x whether or not is condition independence ,if true,dep must be zero,
            # and end operating of function of getMinDep
            if pval > alpha:
                return 0, S, ci_number
            if dep_min > dep:
                dep_min = dep
    return dep_min, None, ci_number


def MMPC(data, target, alpha, is_discrete):
    number, kVar = np.shape(data)
    ci_number = 0
    CPC = []
    deoZeroSet = []
    sepset = [[] for i in range(kVar)]

    while True:
        M_variables = [i for i in range(kVar) if i != target and i not in CPC and i not in deoZeroSet]
        vari_all_dep_max = -float("inf")
        vari_chose = 0

        # according to pseudocode, <F,assocF> = MaxMinFeuristic(T;CPC)
        for x in M_variables:
            # use a function of getMinDep to chose min dep of x
            x_dep_min, sepset_temp, ci_num2 = getMinDep(data, target, x, CPC, alpha, is_discrete)
            ci_number += ci_num2
            # print(str(x)+" dep min is: " + str(x_dep_min))

            # if x chose min dep is 0, it never append to CPC and should not test from now on,
            if x_dep_min == 0:
                deoZeroSet.append(x)
                if sepset_temp is None:
                    sepset[x] = []
                else:
                    sepset[x] = list(sepset_temp)

            elif x_dep_min > vari_all_dep_max:
                vari_chose = x
                vari_all_dep_max = x_dep_min

        # print("x chosed is: " + str(vari_chose)+" and its dep is: " + str(vari_all_dep_max))
        if vari_all_dep_max >= 0:
            # print("CPC append is: "+ str(vari_chose))
            CPC.append(vari_chose)
        else:
            # CPC has not changed(In other world,CPC not append new), circulate should be break
            break
    # print("CPC is:" +str(CPC))
    """phaseII :Backward"""
    # print("shrinking phase begin")

    CPC_temp = CPC.copy()
    max_k = 3
    for a in CPC_temp:
        C_subsets = [i for i in CPC if i != a]

        # please see explanation of the function of getMinDep() explanation
        # the chinese annotation ,if you see,you will know.
        if len(C_subsets) > max_k:
            C_length = max_k
        else:
            C_length = len(C_subsets)

        breakFlag = False
        for length in range(C_length+1):
            if breakFlag:
                break
            SS = subsets(C_subsets, length)
            for S in SS:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, a, S, is_discrete)
                if pval > alpha:
                    CPC.remove(a)
                    breakFlag = True
                    break

    return list(set(CPC)), sepset, ci_number




# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v3.csv")
# print("the file read")
#
# target = 3
# alaph = 0.05
#
# MBs,sepset=MMPC(data,target,alaph)
# print(MBs)
# print(sepset)


# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# data = pd.read_csv("C:/pythonProject/BN_PC_algorithm/CBD/data/child_s5000_v2.csv")
# print("the file read")
# import numpy as np
# num1, kvar = np.shape(data)
# alaph = 0.01
#
# import time
# start_time = time.process_time()
# for target in range(kvar):
#     print("target is: ", target)
#     PC,sepset,ntest = MMPC(data, target, alaph, True)
#
# end_time = time.process_time()
#
# print("time is: ", end_time - start_time)

