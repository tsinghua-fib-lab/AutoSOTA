#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/23 10:08
 @File    : IAMB.py
 """
from CBD.MBs.common.condition_independence_test import cond_indep_test


def IAMB(data, target, alaph, attribute, is_discrete):
    CMB = []
    ci_number = 0

    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y =None
        variables=[i for i in attribute if i != target and i not in CMB]

        for x in variables:
            ci_number += 1
            pival, dep = cond_indep_test(data, target, x, CMB, is_discrete)

            # chose maxsize of f(X:T|CMB)
            if pival <= alaph:
                if dep > temp_dep:
                    temp_dep = dep
                    y = x

        # if not condition independence the node,appended to CMB
        if y is not None:
            CMB.append(y)
            circulate_Flag = True

    # backward circulate phase
    CMB_temp = CMB.copy()
    for x in CMB_temp:
        # exclude variable which need test p-value
        condition_Variables=[i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(data, target, x, condition_Variables, is_discrete)
        if pval > alaph:
            CMB.remove(x)

    return sorted(CMB), ci_number
