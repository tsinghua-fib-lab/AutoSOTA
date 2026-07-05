# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/9 15:00
desc:
"""
import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets


def pc_simple(data, target, alaph, isdiscrete):
    number, kVar = np.shape(data)
    ciTest = 0
    k = 0

    # chose all variables except target itself
    PC = [i for i in range(kVar) if i != target]

    while len(PC) > k:

        PC_temp = PC.copy()
        for x in PC_temp:
            # see number of circulate
            condition_subsets = [i for i in PC_temp if i != x]
            if len(condition_subsets) >= k:
                # get a difinite number of subsets of condition_subsets
                css = subsets(condition_subsets, k)
                for s in css:
                    # every k length of subsets should test chi square and if
                    # make x and target CI,x removed
                    pval, dep = cond_indep_test(data, x, target, s, isdiscrete)
                    ciTest += 1
                    if pval > alaph:

                        PC.remove(x)
                        break  # end circulate of s
        k += 1

    return PC, ciTest


# use demo to test pc_simple
# data = pd.read_csv("C:\pythonProject\pyCausalFS\CBD\data\Child_s500_v1.csv")
# target=12
# alaph=0.01
# pc=pc_simple(data,target,alaph, True)
# print(pc)
