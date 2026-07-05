# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/21 9:42
desc:
"""
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.IPCMB.RecognizePC import RecognizePC
import numpy as np


def IPC_MB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    CanADJT = [i for i in range(kVar) if i != target]
    PC, sepset, ci_number = RecognizePC(
        data, target, CanADJT, alaph, is_discrete)
    # print("pc is: " + str(PC))
    # print("sepset is: " + str(sepset))
    MB = PC.copy()

    for x in PC:
        CanADJT_X = [i for i in range(kVar) if i != x]
        CanSP, _, ci_num2 = RecognizePC(data, x, CanADJT_X, alaph, is_discrete)
        ci_number += ci_num2
        # print("CanSP:" + str(CanSP))
        if target not in CanSP:
            MB.remove(x)
            continue
        for y in CanSP:
            if y != target and y not in MB:
                conditionsSet = [i for i in sepset[y]]
                conditionsSet.append(x)
                conditionsSet = list(set(conditionsSet))
                ci_number += 1
                pval, dep = cond_indep_test(
                    data, target, y, conditionsSet, is_discrete)
                if pval <= alaph:
                    # print("append is:" + str(y)+" conditinSet: " + str(conditionsSet))
                    MB.append(y)

    return list(set(MB)), ci_number


# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v3.csv")
# print("the file read")
#
# target = 6
# alaph = 0.05
#
# MBs=IPC_MB(data,target,alaph)
# print("MBs is: "+str(MBs))


# F1 is: 0.7997213203463205
# Precision is: 0.893875
# Recall is: 0.7637083333333331
# time is: 26.190546875


# 5000

# F1 is: 0.96
# Precision is: 0.94
# Recall is: 1.0
# Distance is: 0.06
# ci_number is: 486.135
# time is: 18.63
