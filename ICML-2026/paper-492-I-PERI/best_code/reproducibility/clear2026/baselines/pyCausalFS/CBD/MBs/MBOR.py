#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/21 21:19
 @File    : MBOR.py
 """

import numpy as np
from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.condition_independence_test import cond_indep_test
# from MBOR.IAMB import IAMB


def IAMB(data, target, alaph, attribute, is_discrete):
    CMB = []
    ci_number = 0

    # forward circulate phase
    circulate_Flag = True
    while circulate_Flag:
        circulate_Flag = False
        # tem_dep pre-set infinite negative.
        temp_dep = -(float)("inf")
        y = None
        variables = [i for i in attribute if i != target and i not in CMB]

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
        condition_Variables = [i for i in CMB if i != x]
        ci_number += 1
        pval, dep = cond_indep_test(
            data, target, x, condition_Variables, is_discrete)
        if pval > alaph:
            CMB.remove(x)

    return CMB, ci_number

# Algorithm 2. PCSuperSet


def PCSuperSet(data, target, alaph, is_discrete):
    ci_number = 0
    d_sep = dict()
    _, kVar = np.shape(data)
    PCS = [i for i in range(kVar) if i != target]
    PCS_temp = PCS.copy()
    for x in PCS_temp:
        ci_number += 1
        pval, _ = cond_indep_test(data, target, x, [], is_discrete)
        if pval > alaph:
            PCS.remove(x)
            d_sep.setdefault(x, [])

    PCS_temp = PCS.copy()
    for x in PCS_temp:
        PCS_rmX = [i for i in PCS if i != x]
        for y in PCS_rmX:
            ci_number += 1
            pval, _ = cond_indep_test(data, target, x, [y], is_discrete)
            if pval > alaph:
                PCS.remove(x)
                d_sep.setdefault(x, [y])
                break

    return PCS, d_sep, ci_number


# Algorithm 3. SPSuperSet

def SPSuperSet(data, target, PCS, d_sep, alaph, is_discrete):
    ci_number = 0
    _, kVar = np.shape(data)
    SPS = []
    for x in PCS:
        SPS_x = []
        vari_set = [i for i in range(kVar) if i != target and i not in PCS]
        for y in vari_set:
            conditon_set = [i for i in d_sep[y]]
            conditon_set.append(x)
            conditon_set = list(set(conditon_set))
            ci_number += 1
            pval, _ = cond_indep_test(
                data, target, y, conditon_set, is_discrete)
            if pval <= alaph:
                SPS_x.append(y)

        SPS_x_temp = SPS_x.copy()
        for y in SPS_x_temp:
            SPS_x_rmy = [i for i in SPS_x if i != y]
            for z in SPS_x_rmy:
                ci_number += 1
                pval, _ = cond_indep_test(data, target, y, [x, z], is_discrete)
                if pval > alaph:
                    SPS_x.remove(y)
                    break

        SPS = list(set(SPS).union(set(SPS_x)))

    return SPS, ci_number


# Algorithm 4. MBtoPC

def MBtoPC(data, target, alaph, attribute, is_discrete):
    max_k = 3
    ci_number = 0
    MB, ci_num = IAMB(data, target, alaph, attribute, is_discrete)
    ci_number += ci_num
    PC = MB.copy()
    for x in MB:
        break_flag = False
        condtion_sets_all = [i for i in MB if i != x]
        c_length = len(condtion_sets_all)
        if c_length > max_k:
            c_length = max_k
        for j in range(c_length + 1):
            condtion_sets = subsets(condtion_sets_all, j)
            for Z in condtion_sets:
                ci_number += 1
                pval, _ = cond_indep_test(data, target, x, Z, is_discrete)
                if pval > alaph:
                    PC.remove(x)
                    break_flag = True
                    break
            if break_flag:
                break
    return PC, ci_number


# Algorithm 1. MBOR

def MBOR(data, target, alaph, is_discrete=True):
    _, kVar = np.shape(data)
    max_k = 3
    ci_number = 0

    PCS, d_sep, ci_num = PCSuperSet(data, target, alaph, is_discrete)
    ci_number += ci_num
    SPS, ci_num = SPSuperSet(data, target, PCS, d_sep, alaph, is_discrete)
    ci_number += ci_num
    MBS = list(set(PCS).union(set(SPS)))

    # drop_data_attribute = [str(i) for i in range(
    #     kVar) if i != target and i not in MBS]
    # data_new = data.drop(drop_data_attribute, axis=1)
    data_attribute = [i for i in range(kVar) if i == target or i in MBS]

    PC, ci_num = MBtoPC(data, target, alaph, data_attribute, is_discrete)
    ci_number += ci_num
    PCS_rmPC = [i for i in PCS if i not in PC]
    for x in PCS_rmPC:
        x_pcset, ci_num = MBtoPC(
            data, x, alaph, data_attribute, is_discrete)

        ci_number += ci_num
        if target in x_pcset:
            PC.append(x)

    SP = []
    for x in PC:
        data_attribute = [i for i in range(kVar) if i != target]
        x_pcset, ci_num = MBtoPC(data, x, alaph, data_attribute, is_discrete)
        ci_number += ci_num
        vari_set = [i for i in x_pcset if i != target and i not in PC]
        for y in vari_set:
            break_flag = False
            condition_all_set = [i for i in MBS if i != target and i != y]
            clength = len(condition_all_set)
            if clength > max_k:
                clength = max_k
            for j in range(clength + 1):
                condition_set = subsets(condition_all_set, j)
                for Z in condition_set:
                    ci_number += 1
                    pval, _ = cond_indep_test(data, target, y, Z, is_discrete)
                    if pval > alaph:
                        if break_flag:
                            break
                        else:
                            # Find minimal Z ⊂ MBS\{T ∪ Y } such that T ⊥ Y |Z
                            break_flag = True
                            condition_varis = [i for i in Z]
                            condition_varis.append(x)
                            condition_varis = list(set(condition_varis))
                            ci_number += 1
                            pval, _ = cond_indep_test(
                                data, target, y, condition_varis, is_discrete)
                            if pval <= alaph:
                                SP.append(y)
                if break_flag:
                    break

    MB = list(set(PC).union(set(SP)))
    return MB, ci_number


# import pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 19
# alaph = 0.01
#
# MB = MBOR(data, target, alaph, True)
# print("MBs is: " + str(MB))


# 500
#
# F1 is: 0.85
# Precision is: 0.92
# Recall is: 0.82
# Distance is: 0.23
# ci_number is: 381.90
# time is: 61.39


# 5000
#
# F1 is: 0.97
# Precision is: 0.96
# Recall is: 0.99
# Distance is: 0.05
# ci_number is: 765.37
# time is: 371.77
