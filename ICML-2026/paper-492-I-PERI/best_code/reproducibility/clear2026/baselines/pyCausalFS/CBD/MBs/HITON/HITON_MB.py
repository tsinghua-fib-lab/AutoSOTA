# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/17 17:00
desc:
"""
from baselines.pyCausalFS.CBD.MBs.common.condition_independence_test import cond_indep_test
from baselines.pyCausalFS.CBD.MBs.HITON.HITON_PC import HITON_PC


def HITON_MB(data, target, alaph, is_discrete=True):

    PC, sepset, ci_number = HITON_PC(data, target, alaph, is_discrete)
    # print("PC is:" + str(PC))
    currentMB = PC.copy()
    for x in PC:
        # print("x is: " + str(x))
        PCofPC, _, ci_num2 = HITON_PC(data, x, alaph, is_discrete)
        ci_number += ci_num2
        # print("PCofPC is " + str(PCofPC))
        for y in PCofPC:
            # print("y is " + str(y))
            if y != target and y not in PC:
                conditions_Set = [i for i in sepset[y]]
                conditions_Set.append(x)
                conditions_Set = list(set(conditions_Set))
                ci_number += 1
                pval, dep = cond_indep_test(
                    data, target, y, conditions_Set, is_discrete)
                if pval <= alaph:
                    # print("append is: " + str(y))
                    currentMB.append(y)
                    break

    return list(set(currentMB)), ci_number


# alpha = 0.01
# start_time = time.process_time()
# for target in range(kvar):
#     print("target:", target)
#     MBs, ci_number = HITON_MB(data, target, alpha, True)
#     print("ci_number : ", ci_number)
#     # print(dic["cache"][0], "-", dic["cache"][1],
#     #   "-", (dic["cache"][0] + dic["cache"][1]))
#     # print(dic["cache"][0] / (dic["cache"][0] + dic["cache"][1]))
#
# end_time = time.process_time()
# print("run time = ", end_time - start_time)

# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 4
# alaph = 0.05
#
# MBs=HITON_MB(data,target,alaph)
# print("MBs is: "+str(MBs))
