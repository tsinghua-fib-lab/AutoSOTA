#_*_code:utf_8_*_
#!/usr/bin/env python
#date:2019/8/13 16:27
from CBD.MBs.common.chi_square_test import chi_square_test
from CBD.MBs.common.subsets import subsets
import numpy as np
import pandas as pd

def interIAMBnPC(data,target,alaph):
    n,p=np.shape(data)
    BT=[]
    ntest=0
    #depmax=-float("inf")
    #feature=-1
    #pval_temp=1.0
    length=-1
    removeSet = []
    rmNumberSet = [0 for i in range(p)]
    while len(BT)!=length:
        depmax = -float("inf")
        feature = -1
        pval_temp = 1.0
        length=len(BT)
        #growing phase
        S=[i for i in range(p)if i!=target and i not in BT and i not in removeSet]
        for X in S:
            ntest += 1
            pval, dep = chi_square_test(data,X,target,BT)
            if dep>depmax:
                depmax=dep
                feature=X
                pval_temp=pval
        if pval_temp<=alaph:
            BT.append(feature)

         #shrinking phase
        mb_index = len(BT)
        # 逆序
        while mb_index >= 0:
            mb_index -= 1
            x = BT[mb_index]

            ntest += 1

            conditionvars= [i for i in BT if i != x]
            pval_sp, dep_sp = chi_square_test(data, target, x, conditionvars)
            if pval_sp > alaph:
                BT.remove(x)
                # remove the variables while have be append to MBs just,lead to circulation break
                if x == feature:
                    break

                rmNumberSet[x] += 1
                if rmNumberSet[x] > 10:
                    removeSet.append(x)


    """shrinking phase"""
    TestMB = BT.copy()
    p = len(TestMB)
    DAG = np.ones((1, p))
    size = 0
    continueFlag = True
    # conditionSet maximum set 3
    max_k = 3
    while continueFlag:
        # Candidate of MB traverse
        for y in range(p):
            if DAG[0, y] == 0:
                continue
            conditionSet = [i for i in range(p) if i != y and DAG[0,i] == 1]
            SS = subsets(conditionSet, size)
            for S in SS:
                condtionVari = [TestMB[i] for i in S]
                ntest += 1
                pval, _ = chi_square_test(data, target, TestMB[y], condtionVari)
                if pval > alaph:
                    DAG[0, y] = 0
                    break
        size += 1
        continueFlag = False

        # circulate will be continue if condition suited
        if np.sum(DAG[0, :] == 1) >= size and size <= max_k:
            continueFlag = True
    # end while
    MB = [TestMB[i] for i in range(p) if DAG[0,i] == 1 ]

    return MB, ntest
# data=pd.read_csv("E:/python/pycharm/algorithm/data/Child_s500_v9.csv")
# BT,ntest=interIAMBnPC(data,2,0.01)
# print(BT)
# print(ntest)

# Child_s500
# F1 is: 0.64
# Precision is: 0.86
# Recall is: 0.57
# Distance is: 0.49
# ci_number is: 68.92
# time is: 6.04
