# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/9 20:10
desc: 
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test

def inter_IAMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    ci_number = 0
    MB=[]
    circulateFlag = True
    removeSet = []
    rmNumberSet = [0 for i in range(kVar)]
    while circulateFlag:
        circulateFlag =False
        # print("MBs is:" + str(MBs))
        dep_temp = - float("inf")
        pval_temp = 1
        max_s = None

        # remove target element from set before test
        variables =[i for i in range(kVar) if i != target and i not in MB and i not in removeSet]

        # growing phase
        for s in variables:
            ci_number += 1
            # print(numberOfCirculate)

            pval_gp, dep_gp = cond_indep_test(data, target,s, MB, is_discrete)

            if dep_gp > dep_temp:
                dep_temp = dep_gp
                max_s = s
                pval_temp = pval_gp

        if pval_temp <= alaph:
            # if any changes ,circulate should be continue
            circulateFlag = True
            MB.append(max_s)
            # print("BT append vari is:" + str(max_s))

        # if not append any variables to BT before this,the shirnking phase must not delete any variables.
        # save time
        if circulateFlag == False:
            break


        # print("----> shrinking phase")
        # use mb_index ,to be pointer
        mb_index = len(MB)
        # 逆序
        while mb_index >= 0:
            mb_index -= 1
            x = MB[mb_index]

            ci_number += 1

            subsets_Variables = [i for i in MB if i != x]
            pval_sp, dep_sp = cond_indep_test(data, target, x, subsets_Variables, is_discrete)
            if pval_sp > alaph:
                MB.remove(x)
                # remove the variables while have be append to MBs just,lead to circulation break
                if x == max_s:
                    break

                rmNumberSet[x] += 1
                if rmNumberSet[x] > 10:
                    removeSet.append(x)
                # print("BT remove vari is: "+ str(x) + " ,rmNumberSet[x] is:" + str(rmNumberSet[x]))
                # if any changes,circulate should be contine
                # circulateFlag = True

    return list(set(MB)), ci_number


# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 4
# alaph = 0.05
#
# CMB=inter_IAMB(data,target,alaph)
#
# print(CMB)


# F1 is: 0.7603808691308696
# Precision is: 0.7835000000000002
# Recall is: 0.8212083333333333
# time is: 21.37359375


#5000

# F1 is: 0.91
# Precision is: 0.89
# Recall is: 0.95
# Distance is: 0.14
# ci_number is: 120.48
# time is: 68.37

