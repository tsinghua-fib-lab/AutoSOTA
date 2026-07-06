from CBD.MBs.common.condition_independence_test import cond_indep_test
import numpy as np
import random


def KIAMB(data, target, alaph, k, is_discrete=True):
    n, p = np.shape(data)
    MB = []
    ci_number = 0
    flag = True
    while flag:
        x_dep = [0 for i in range(p)]
        flag = False
        CanMB = []
        variables = [i for i in range(p) if i != target and i not in MB]
        for x in variables:
            ci_number += 1
            pval, dep = cond_indep_test(data, target, x, MB, is_discrete)
            if pval <= alaph:
                CanMB.append(x)
                x_dep[x] = dep
        if len(CanMB) == 0:
            break
        CanMB2 = random.sample(CanMB, max(1, int(len(CanMB) * k)))
        max_dep = -float("inf")
        Y = None
        for x in CanMB2:
            if x_dep[x] > max_dep:
                Y = x
                max_dep = x_dep[x]
        if Y is not None:
            MB.append(Y)
            flag = True

    # remove false positives from MB
    MB_temp = MB.copy()
    for x in MB_temp:
        condition_set = [i for i in MB if i != x]
        ci_number += 1
        pval, _ = cond_indep_test(data, target, x, condition_set, is_discrete)
        if pval > alaph:
            MB.remove(x)

    return list(set(MB)), ci_number

# data = pd.read_csv("F:\cai_algorithm\data\Child_s500_v1.csv")
# MB,num = KIAMB(data,1,0.01,0.8)
# print(MB)
# print(num)

# 500 0.01 k = 0.8
# F1 is: 0.7437202519702523
# Precision is: 0.8664166666666666
# Recall is: 0.7202083333333331
# time is: 24.29546875


# F1 is: 0.75
# Precision is: 0.85
# Recall is: 0.73
# Distance is: 0.37
# ci_number is: 87.23
# time is: 24.73

# 5000 0.01 k = 0.8
# F1 is: 0.8891481990231993
# Precision is: 0.9209940476190475
# Recall is: 0.8887500000000002
# time is: 215.52859375


# 5000 0.01 k = 0.8
# F1 is: 0.89
# Precision is: 0.91
# Recall is: 0.90
# Distance is: 0.17
# ci_number is: 113.84
# time is: 218.20
