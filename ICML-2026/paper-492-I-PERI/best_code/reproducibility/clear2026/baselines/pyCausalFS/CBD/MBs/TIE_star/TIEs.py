#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/9/10 20:58
 @File    : TIEs.py
 """

from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.TIE_star.IAMB import IAMB
from CBD.MBs.TIE_star.eva_classifier import eva_classifier
import numpy as np


def TIE(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    M = []
    G = []
    max_k = 3
    not_in_set = []
    possible_subests = []
    variable = [i for i in range(kVar)]
    MB, _ = IAMB(data, target, alaph, variable, is_discrete)
    M.append(MB)
    G.append([])
    index = 0
    s_index = 0
    MB_new_set = []
    while True:
        length = len(M[index])
        if length > max_k:
            length = max_k
        for j in range(length+1):
            if j == 0:
                continue
            varis_set = subsets(M[index], j)
            for x in varis_set:
                break_Flag = False
                for y in not_in_set:
                    if set(x).issuperset(set(y)):
                        break_Flag = True
                        break
                if not break_Flag:
                    vari_one = list(set(x).union(set(G[index])))

                    if vari_one not in possible_subests:
                        possible_subests.append(vari_one)


        if s_index < len(possible_subests):
            excpet_varis_set = possible_subests[s_index]
            s_index += 1
        else:
            break


        variable_new = [i for i in range(kVar) if i not in excpet_varis_set]
        MB_new, _ = IAMB(data, target, alaph, variable_new, is_discrete)
        different_set = list(set(MB).difference(set(MB_new)))
        if different_set == [] or MB_new == []:
            continue
        break_Flag = False
        for x in different_set:
            pval, _ = cond_indep_test(data, target, x, MB_new, is_discrete)
            if pval <= alaph:
                break_Flag = True
                not_in_set.append(excpet_varis_set)
                possible_subests_temp = possible_subests.copy()
                for y in possible_subests_temp:
                    if excpet_varis_set != y and set(y).issuperset(excpet_varis_set):
                        possible_subests.remove(y)
                break
        if not break_Flag:
            MB_new_set.append(MB_new)
            M.append(MB_new)
            G.append(excpet_varis_set)
            index += 1
    return MB_new_set


def TIE_p(data, target, alaph, isdiscrete):
    number, kVar = np.shape(data)
    M = []
    G = []
    max_k = 3
    not_in_set = []
    possible_subests = []
    variable = [i for i in range(kVar)]
    MB, _ = IAMB(data, target, alaph, variable, isdiscrete)

    accurary_MB = eva_classifier(data, target, MB)
    M.append(MB)
    G.append([])
    index = 0
    s_index = 0
    MB_new_set = []
    while True:
        length = len(M[index])
        if length > max_k:
            length = max_k
        for j in range(length+1):
            if j == 0:
                continue
            varis_set = subsets(M[index], j)
            # print("varis_set is: " + str(varis_set))
            for x in varis_set:
                break_Flag = False
                for y in not_in_set:
                    if set(x).issuperset(set(y)):
                        # print("x is: " + str(x) + " , y is: " + str(y))
                        break_Flag = True
                        break
                if not break_Flag:
                    vari_one = list(set(x).union(set(G[index])))
                    # print("vari_one is: " + str(vari_one))
                    if vari_one not in possible_subests:
                        possible_subests.append(vari_one)
                        # print("possible_subsets is: " + str(possible_subests))


        if s_index < len(possible_subests):
            excpet_varis_set = possible_subests[s_index]
            s_index += 1
        else:
            break


        variable_new = [i for i in range(kVar) if i not in excpet_varis_set]
        MB_new, _ = IAMB(data, target, alaph, variable_new, isdiscrete)
        if MB_new == [] or MB_new in MB_new_set:
            continue

        # accurary_MB = eva_classifier(data, target, MB)
        accurary_MB_new = eva_classifier(data, target, MB_new)

        if accurary_MB <= accurary_MB_new:
            MB_new_set.append(MB_new)
            M.append(MB_new)
            G.append(excpet_varis_set)
            index += 1
    return MB_new_set


# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v1.csv")
# print("the file read")
#
# target = 4
# alaph = 0.01
#
# MB = TIE(data, target, alaph, False)
# print("MBs is: " + str(MB))


