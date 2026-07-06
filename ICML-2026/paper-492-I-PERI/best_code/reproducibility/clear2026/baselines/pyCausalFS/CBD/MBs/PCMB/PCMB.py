# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/17 15:23
desc:
"""

from CBD.MBs.PCMB.getPC import getPC
from CBD.MBs.common.condition_independence_test import cond_indep_test


def PCMB(data, target, alaph, is_discrete=True):
    ci_number = 0
    PC, sepset, ci_num2 = getPC(data, target, alaph, is_discrete)
    ci_number += ci_num2
    # print(PC)
    # print(sepset)
    MB = PC.copy()

    for x in PC:
        PCofPC_temp, _, ci_num3 = getPC(data, x, alaph, is_discrete)
        ci_number += ci_num3
        # print(" pc of pc_temp is: " + str(PCofPC_temp))
        PCofPC = [i for i in PCofPC_temp if i != target and i not in MB]
        # print(" pc of pc is: " + str(PCofPC))
        for y in PCofPC:
            conditionSet = [i for i in sepset[y]]
            conditionSet.append(x)
            conditionSet = list(set(conditionSet))
            ci_number += 1
            pval, dep = cond_indep_test(
                data, target, y, conditionSet, is_discrete)
            if pval <= alaph:
                MB.append(y)
                break
    return list(set(MB)), ci_number


# import pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v2.csv")
# print("the file read")
#
# target = 19
# alaph = 0.05
#
# MBs=PCMB(data,target,alaph)
# print("MBs is: "+str(MBs))

# 500

# F1 is: 0.7966175213675215
# Precision is: 0.8985357142857144
# Recall is: 0.7550416666666664
# time is: 148.834140625


# F1 is: 0.7989160561660563
# Precision is: 0.902702380952381
# Recall is: 0.7550416666666664
# time is: 167.22234375

# 5000

# F1 is: 0.99
# Precision is: 0.99
# Recall is: 0.99
# Distance is: 0.02
# ci_number is: 8189.81
# time is: 630.02
