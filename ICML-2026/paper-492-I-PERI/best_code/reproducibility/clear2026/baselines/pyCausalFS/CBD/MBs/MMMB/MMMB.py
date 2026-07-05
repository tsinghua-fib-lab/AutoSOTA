# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/15 9:57
desc:
"""
from baselines.pyCausalFS.CBD.MBs.MMMB.MMPC import MMPC
from baselines.pyCausalFS.CBD.MBs.common.condition_independence_test import cond_indep_test


def MMMB(data, target, alaph, is_discrete=True):
    ci_number = 0
    PC, sepset, ci_num2 = MMPC(data, target, alaph, is_discrete)
    ci_number += ci_num2
    MB = PC.copy()
    for x in PC:
        PCofPC, _, ci_num3 = MMPC(data, x, alaph, is_discrete)
        ci_number += ci_num3
        for y in PCofPC:
            if y != target and y not in PC:
                conditions_Set = [i for i in sepset[y]]
                conditions_Set.append(x)
                conditions_Set = list(set(conditions_Set))
                ci_number += 1
                pval, dep = cond_indep_test(
                    data, target, y, conditions_Set, is_discrete)
                if pval <= alaph:
                    MB.append(y)
                    break
    return list(set(MB)), ci_number

# import pandas as pd
# data = pd.read_csv("C:/pythonProject/BN_PC_algorithm/CBD/data/child_s5000_v1.csv")
# print("the file read")
#
# target = 10
# alaph = 0.05
#
# MBs=MMMB(data,target,alaph)
# print("MBs is: "+str(MBs))


# F1 is: 0.7791296481296481
# Precision is: 0.8212976190476191
# Recall is: 0.796958333333333
# time is: 36.05296875

# 5000

# F1 is: 0.93 ±0.30
# Precision is: 0.90±0.26
# Recall is: 0.99±0.45
# Distance is: 0.11±0.39
# ci_number is: 1097.80±2499.40
# time is: 93.20±347.69
#
