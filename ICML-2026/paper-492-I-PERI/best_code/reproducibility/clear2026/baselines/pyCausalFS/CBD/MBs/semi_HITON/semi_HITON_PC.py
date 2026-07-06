#from common.chi_square_test import chi_square_test
from CBD.MBs.common.subsets import subsets
from CBD.MBs.common.condition_independence_test import cond_indep_test
import numpy as np


def semi_HITON_PC(data, target, alaph, is_disrete=True):
    n, p = np.shape(data)
    ci_number = 0
    candidate_pc = []
    S = []
    current_pc = []
    sep = [[] for i in range(p)]
    con = [i for i in range(p) if i != target]
    for x in con:
        ci_number += 1
        pval, dep = cond_indep_test(data, target, x, [], is_disrete)
        if pval <= alaph:
            S.append([x, dep])

    depset = sorted(S, key=lambda x: x[1], reverse=True)
    for i in range(len(depset)):
        candidate_pc.append(depset[i][0])  # RANK

    for x in candidate_pc:
        breakflag = False
        current_pc.append(x)
        conditions_set = [i for i in current_pc if i != x]
        # print("conditions_set is " + str(conditions_set))
        if len(conditions_set) >= 3:
            Slength = 3
        else:
            Slength = len(conditions_set)
        for j in range(Slength + 1):
            SS = subsets(conditions_set, j)
            for s in SS:
                ci_number += 1
                pval, _ = cond_indep_test(data, x, target, s, is_disrete)
                if pval > alaph:
                    sep[x] = [i for i in s]
                    current_pc.remove(x)
                    breakflag = True
                    break
            if breakflag:
                break

    # backforward phase except the last add variable
    Last_added = None
    if len(current_pc) > 0:
        Last_added = current_pc[-1]

    current_temp = current_pc.copy()
    for x in current_temp:
        flag = False
        if x == Last_added:
            continue
        con_set = [i for i in current_pc if i != x]
        if len(con_set) >= 3:
            leng = 3
        else:
            leng = len(con_set)
        for j in range(leng + 1):
            SS = subsets(con_set, j)
            for s in SS:
                ci_number += 1
                pval, _ = cond_indep_test(data, x, target, s, is_disrete)
                if pval > alaph:
                    current_pc.remove(x)
                    sep[x] = [i for i in s]
                    flag = True
                    break
            if flag:
                break
    return list(set(current_pc)), sep, ci_number


# data = pd.read_csv("F:\cai_algorithm\data\Child_s500_v1.csv")
# PC,sep,ci = semi_HITON_PC(data,18,0.01)
# print(PC)
# print(sep)
