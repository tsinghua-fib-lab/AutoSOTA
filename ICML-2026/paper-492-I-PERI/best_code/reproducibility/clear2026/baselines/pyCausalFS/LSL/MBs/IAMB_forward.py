# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/8 8:41
desc: 
"""
import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
# import common.subsets as subsets


def IAMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    CMB = []
    ci_number = 0
    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        # if not change, forward phase of IAMB is finished.
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y = None
        variables = [i for i in range(kVar) if i != target and i not in CMB]

        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, CMB, is_discrete)
            # print("target is:",target,",x is: ", x," CMB is: ", CMB," ,pval is: ",pval," ,dep is: ", dep)

            # chose maxsize of f(X:T|CMB)
            if pval <= alaph:
                if dep > temp_dep:
                    temp_dep=dep
                    y=x

        # if not condition independence the node,appended to CMB
        if y is not None:
            # print('appended is :'+str(y))
            CMB.append(y)
            circulate_Flag = True


    return list(set(CMB)), ci_number



# F1 is: 0.75430044955045
# Precision is: 0.8198333333333335
# Recall is: 0.7885833333333332
# time is: 22.64546875

# F1 is: 0.81
# Precision is: 0.89
# Recall is: 0.79
# Distance is: 0.28
# ci_number is: 77.25
# time is: 15.16



# 5000
#
# F1 is: 0.92 ±0.40
# Precision is: 0.94±0.53
# Recall is: 0.94±0.30
# Distance is: 0.12±0.56
# ci_number is: 95.82±38.47
# time is: 74.56±188.69

# F1 is: 0.89
# Precision is: 0.88
# Recall is: 0.94
# Distance is: 0.16
# ci_number is: 97.85
# time is: 88.89

# import  pandas as pd
# data = pd.read_csv("../data/Alarm1_s1000_v1.csv")
# print("the file read")
#
# target = 2
# alaph = 0.01
#
# MBs=IAMB(data, target, alaph, is_discrete=True)
# print("MBs is: "+str(MBs))