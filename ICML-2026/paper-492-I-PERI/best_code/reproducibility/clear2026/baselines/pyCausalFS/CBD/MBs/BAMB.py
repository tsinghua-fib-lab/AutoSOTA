#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/8/6 20:18
 @File    : BAMB.py
 """


import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
from CBD.MBs.common.subsets import subsets


def BAMB(data, target, alaph, is_discrete=True):
    ci_number = 0
    number, kVar = np.shape(data)
    max_k = 3
    CPC = []
    TMP = [i for i in range(kVar) if i != target]
    sepset = [[] for i in range(kVar)]
    CSPT = [[] for i in range(kVar)]
    variDepSet = []
    SP = [[] for i in range(kVar)]
    PC = []

    for x in TMP:
        ci_number += 1
        pval_f, dep_f = cond_indep_test(data, target, x, [], is_discrete)
        if pval_f > alaph:
            sepset[x] = []
        else:
            variDepSet.append([x, dep_f])

    variDepSet = sorted(variDepSet, key=lambda x: x[1], reverse=True)
    """step one: Find the candidate set of PC and candidate set of spouse"""

    # print("variDepSet" + str(variDepSet))
    for variIndex in variDepSet:
        A = variIndex[0]
        # print("A is: " + str(A))
        Slength = len(CPC)
        if Slength > max_k:
            Slength = 3
        breakFlag = False
        for j in range(Slength + 1):
            ZSubsets = subsets(CPC, j)
            for Z in ZSubsets:
                ci_number += 1
                convari = [i for i in Z]
                pval_TAZ, dep_TAZ = cond_indep_test(
                    data, target, A, convari, is_discrete)
                if pval_TAZ > alaph:
                    sepset[A] = convari
                    breakFlag = True
                    # print("ZZZ")
                    break
            if breakFlag:
                break

        if not breakFlag:
            CPC_ReA = CPC.copy()
            B_index = len(CPC_ReA)
            CPC.append(A)
            breakF = False
            while B_index > 0:
                B_index -= 1
                B = CPC_ReA[B_index]
                flag1 = False

                conditionSet = [i for i in CPC_ReA if i != B]
                Clength = len(conditionSet)
                if Clength > max_k:
                    Clength = max_k
                for j in range(Clength + 1):
                    CSubsets = subsets(conditionSet, j)
                    for Z in CSubsets:
                        ci_number += 1
                        convari = [i for i in Z]
                        pval_TBZ, dep_TBZ = cond_indep_test(
                            data, target, B, convari, is_discrete)
                        # print("pval_TBZ: " + str(pval_TBZ))
                        if pval_TBZ >= alaph:

                            CPC.remove(B)
                            CSPT[B] = []
                            sepset[B] = convari

                            flag1 = True
                            if B == A:
                                breakF = True
                    if flag1:
                        break
                if breakF:
                    break

            CSPT[A] = []
            pval_CSPT = []

            # add candidate of spouse

            # print("sepset: " + str(sepset))
            for C in range(kVar):
                if C == target or C in CPC:
                    continue
                conditionSet = [i for i in sepset[C]]
                conditionSet.append(A)
                conditionSet = list(set(conditionSet))

                ci_number += 1
                pval_CAT, _ = cond_indep_test(
                    data, target, C, conditionSet, is_discrete)
                if pval_CAT <= alaph:
                    CSPT[A].append(C)
                    pval_CSPT.append([C, pval_CAT])

            """step 2-1"""

            pval_CSPT = sorted(pval_CSPT, key=lambda x: x[1], reverse=False)
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("pval_CSPT is: " + str(pval_CSPT))

            for pCSPT_index in pval_CSPT:
                E = pCSPT_index[0]
                # print("E is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " +str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False
                    # print("x is:" + str(x))

                    ZAllconditionSet = [i for i in SP[A] if i != x]
                    # print("ZAllconditionSet is:" + str(ZAllconditionSet))
                    for Z in ZAllconditionSet:
                        conditionvari = [Z]
                        if A not in conditionvari:
                            conditionvari.append(A)
                        ci_number += 1
                        pval_TXZ, _ = cond_indep_test(
                            data, target, x, conditionvari, is_discrete)
                        # print("x is: " + str(x) + "conditionvari: " + str(conditionvari) + " ,pval_TXZ is: " + str(pval_TXZ))
                        if pval_TXZ > alaph:
                            # print("spa is: " + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(conditionvari))
                            SP[A].remove(x)
                            breakFlag = True

                            if x == E:
                                breakflag_spa = True
                            break
                    if breakFlag:
                        break
                if breakflag_spa:
                    break

            """step 2-2"""
            # remove x from pval_CSPT
            pval_CSPT_new = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_new.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " + str(CSPT))
            # print("2222222pval_CSPT_new is: " + str(pval_CSPT_new))

            for pCSPT_index in pval_CSPT_new:
                E = pCSPT_index[0]
                # print("E2 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]

                    breakFlag = False
                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    # print("CPC is: " + str(CPC) + " , SP[A] is: " + str(SP[A]) + " ,A is" + str(A) + " ,x is:" + str(x) + " ,ZA is: " + str(ZAllSubsets))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    # print("-ZALLSubsets has: " + str(ZAllSubsets))
                    Zalength = len(ZAllSubsets)
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        for Z in ZaSubsets:
                            Z = [i for i in Z]
                            ci_number += 1
                            pval_TXZ, _ = cond_indep_test(
                                data, A, x, Z, is_discrete)
                            # print("Z is: " + str(Z) + " ,A is: " + str(A) + " ,x is: " + str(x) + " ,pval_txz is: " + str(pval_TXZ))
                            if pval_TXZ > alaph:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                breakFlag = True
                                if x == E:
                                    breakflag_spa = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break

            """ step 2-3"""
            pval_CSPT_fin = []
            plength = len(pval_CSPT)
            for i in range(plength):
                if pval_CSPT[i][0] in SP[A]:
                    pval_CSPT_fin.append(pval_CSPT[i])

            CSPT[A] = SP[A]
            SP[A] = []
            # print("CSPT-: " +str(CSPT))
            # print("2222222pval_CSPT_fin is: " + str(pval_CSPT_fin))

            for pCSPT_index in pval_CSPT_fin:
                E = pCSPT_index[0]
                # print("E3 is:" + str(E))

                SP[A].append(E)
                index_spa = len(SP[A])
                breakflag_spa = False
                # print("SP[A] is: " + str(SP[A]))
                while index_spa >= 0:
                    index_spa -= 1
                    x = SP[A][index_spa]
                    breakFlag = False

                    # print("x is:" + str(x))
                    ZAllSubsets = list(set(CPC).union(set(SP[A])))
                    ZAllSubsets.remove(x)
                    ZAllSubsets.remove(A)
                    Zalength = len(ZAllSubsets)
                    # print("=-ZALLSubsets has: " + str(ZAllSubsets))
                    if Zalength > max_k:
                        Zalength = max_k
                    for j in range(Zalength + 1):
                        ZaSubsets = subsets(ZAllSubsets, j)
                        # print("ZzSubsets is: " + str(ZaSubsets))
                        for Z in ZaSubsets:
                            Z = [i for i in Z]
                            Z.append(A)
                            # print("Z in ZaSubsets is: " + str(Z))
                            ci_number += 1
                            pval_TXZ, _ = cond_indep_test(
                                data, target, x, Z, is_discrete)
                            # print("-Z is: " + str(Z) + " ,x is: " + str(x) + " ,pval_txz is: " + str(
                            #     pval_TXZ))
                            if pval_TXZ >= alaph:
                                # print("spa is:" + str(SP[A]) + " .remove x is: " + str(x) + " ,Z is: " + str(Z))
                                SP[A].remove(x)
                                if x == E:
                                    breakflag_spa = True
                                breakFlag = True
                                break
                        if breakFlag:
                            break
                    if breakflag_spa:
                        break
            # print("SP[A]------: " + str(SP[A]))
            CSPT[A] = SP[A]
            # print("CSPT is: " + str(CSPT))

            """step3: remove false positives from the candidate set of PC"""

            CPC_temp = CPC.copy()
            x_index = len(CPC_temp)
            A_breakFlag = False
            # print("-CPC-: " + str(CPC))
            while x_index >= 0:
                x_index -= 1
                x = CPC_temp[x_index]
                flag2 = False
                ZZALLsubsets = [i for i in CPC if i != x]
                # print("xx is: " + str(x) + ", ZZALLsubsets is: " + str(ZZALLsubsets ))
                Zlength = len(ZZALLsubsets)
                if Zlength > max_k:
                    Zlength = max_k
                for j in range(Zlength + 1):
                    Zzsubsets = subsets(ZZALLsubsets, j)
                    for Z in Zzsubsets:
                        conditionSet = [
                            i for y in Z for i in CSPT[y] if i not in CPC]
                        conditionSet = list(set(conditionSet).union(set(Z)))
                        # print("conditionSet: " + str(conditionSet))
                        ci_number += 1
                        pval, _ = cond_indep_test(
                            data, target, x, conditionSet, is_discrete)
                        if pval >= alaph:
                            # print("remove x is: " + str(x) + " , pval is: " + str(pval) + " ,conditionset is: " + str(conditionSet))
                            CPC.remove(x)
                            CSPT[x] = []
                            flag2 = True
                            if x == A:
                                A_breakFlag = True
                            break
                    if flag2:
                        break
                if A_breakFlag:
                    break

    # print("SP is:" + str(SP))
    spouseT = [j for i in CPC for j in CSPT[i]]
    MB = list(set(CPC).union(set(spouseT)))
    return MB, ci_number


# import  pandas as pd
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s5000_v1.csv")
# print("the file read")
#
# target = 19
# alaph = 0.01
#
# import time
# start_time = time.process_time()
# MB, _ = BAMB(data, target, alaph, is_discrete=False)
# end_time = time.process_time()
#
# print(end_time - start_time)
# print("MBs is: "+str(MB))


# 5000
#
# F1 is: 0.97
# Precision is: 0.99
# Recall is: 0.96
# Distance is: 0.05
# ci_number is: 512.58
# time is: 62.22
