# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/12 8:50
desc: 
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test


def GSMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    circulateFlag = True
    S_variables = [i for i in range(kVar) if i !=target]

    """grow phase"""
    # print("grow phase")
    while circulateFlag:
        circulateFlag = False
        for x in S_variables:
            ci_number += 1
            pval_gp, dep_gp = cond_indep_test(data, target, x, CMB, is_discrete)
            if pval_gp < alaph:
                # print("CMB append is: "+str(x))
                CMB.append(x)
                circulateFlag = True
                break
        S_variables = [i for i in range(kVar) if i != target and i not in CMB]

    """"shrink phase"""
    # print("shrink phase")
    circulateFlag = True
    while circulateFlag:
        circulateFlag = False
        CMB_temp = CMB.copy()
        for x in CMB_temp:
            subsets_CMB = [i for i in CMB if i != x]
            ci_number += 1
            pval_sp, dep_sp = cond_indep_test(data, target, x, subsets_CMB, is_discrete)
            if pval_sp > alaph :
                # print("CMB remove is: "+ str(x))
                CMB.remove(x)
                circulateFlag = True
                break

    return list(set(CMB)), ci_number

# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v2.csv")
# print("the file read")
#
# target = 1
# alaph = 0.05
#
# MBs=GSMB(data,target,alaph)
# print(MBs)


# F1 is: 0.7752499444999447
# Precision is: 0.8426666666666667
# Recall is: 0.78825
# time is: 18.858984375


# 5000
#
# F1 is: 0.92
# Precision is: 0.93
# Recall is: 0.94
# Distance is: 0.12
# ci_number is: 41.40
# time is: 48.02