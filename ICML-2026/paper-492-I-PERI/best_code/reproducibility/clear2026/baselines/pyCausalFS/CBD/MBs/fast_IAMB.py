# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/10 9:49
desc:
"""

import numpy as np
from CBD.MBs.common.condition_independence_test import cond_indep_test
def ns(data, MB):
    qi = []
    # if MBs == []:
    #     qi.append(1)
    for i in MB:
        i = str(i)
        q_temp = len(np.unique(data[i]))
        qi.append(q_temp)
    return qi


def fast_IAMB(data, target, alaph, is_discrete=True):
    number, kVar = np.shape(data)
    ci_number = 0

    #BT present B(T) and set null,according to pseudocode
    MB = []

    # set a dictionary to store variables and their pval,but it temporary memory
    S_variables=[]
    MBvariables = [i for i in range(kVar) if i != target ]
    repeat_in_set = [0 for i in range(kVar)]
    num_reapeat = 10
    no_in_set = []
    for x in MBvariables:
        ci_number += 0
        pval, dep = cond_indep_test(data, target, x, MB, is_discrete)
        if(pval <= alaph):
            S_variables.append([x,dep])
    BT_temp = -1

    """iteritems() 得到的[(键，值)]的列表， 通过sorted方法，指定排序的键值key是原来字典中的value属性，其中
    用到了匿名函数lambda， 参数为t列表，返回第二个元素t[1]，也就是每个键值对中的value，  从小到大排序时 reverse=False，
    从大到小排序是True！
     output is [(key,value),...],which is sorted, and other aim is turn dictionary into this structrue [(key,value)]"""

    # preset value
    attributes_removed_Flag = False

    while S_variables != []:
        flag_repeat_set = [False for i in range(kVar)]
        # S sorted according to pval
        S_variables = sorted(S_variables, key=lambda x: x[1], reverse=True)
        # print(S_variables)

        """Growing phase"""
        # print("growing phase begin!")
        S_length=len(S_variables)
        insufficient_data_Flag=False
        attributes_removed_Flag = False
        for y in range(S_length):
            x = S_variables[y][0]
            # number = number
            # print("MBs is: " + str(MBs))
            qi = ns(data, MB)
            # print("qi is: " + str(qi))
            tmp = [1]
            temp1 = []
            if len(qi) > 1:
                temp1 = np.cumprod(qi[0:-1])
            # print("temp1 is: " + str(temp1))
            for i in temp1:
                tmp.append(i)
            # qs = 1 + ([i-1 for i in qi]) * tmp

            # qs = np.array([i-1 for i in qi])* np.array(tmp).reshape(len(tmp),1) + 1
            # print("qi is: " + str(qi) + " ,tmp is: " + str(tmp))
            qs = 0
            if qi == []:
                qs = 0
            else:
                for i in range(len(qi)):
                    qs += (qi[i]-1)*tmp[i]
                qs += 1

            # print("qs is: " + str(qs))
            qxt = ns(data, [x, target])
            # print("length of qs is:" + str(len(list(qs))))
            # print("qxt is: " + str(qxt))
            if qs == 0 :
                df = np.prod(np.mat([i-1 for i in qxt])) * np.prod(np.mat(qi))
                # print("1 = " + str(np.prod(np.array([i-1 for i in qxt]))) + " , 2 = " + str(np.prod(np.array(qi))))
            else:
                df = np.prod(np.mat([i-1 for i in qxt])) * qs
                # print("1 = " + str(np.prod(np.array([i-1 for i in qxt])))+" , 22 = " + str(qs))
            # print("df = " + str(df))
            if number >= 5 * df:
                # S_sort = [(key,value),....],and BT append is key
                MB.append(S_variables[y][0])
                flag_repeat_set[S_variables[y][0]] =True
                # print("BT append is: " + str(S_variables[y][0]))
            else:
                # print('1')
                insufficient_data_Flag=True
                # due to insufficient data, then go to shrinking phase
                break

        """shrinking phase"""
        # print("shrinking phase begin")
        if BT_temp == MB:
            break
        BT_temp = MB.copy()
        # print(BT)
        for x in BT_temp:

            subsets_BT = [i for i in MB if i != x]
            ci_number += 1
            pval_sp, dep_sp = cond_indep_test(data, target, x, subsets_BT, is_discrete)

            if pval_sp > alaph:
                MB.remove(x)
                if flag_repeat_set[x] == True:
                    repeat_in_set[x] += 1
                    if repeat_in_set[x] > num_reapeat:
                        no_in_set.append(x)
                        # print("x not in again is: " + str(x))
                # print("BT remove is: "+str(x))
                attributes_removed_Flag = True

        # if no variable will add to S_variables, circulate will be break,and output the result
        if (insufficient_data_Flag == True) and (attributes_removed_Flag == False):
            # print("circulate end!")
            break
        else:
            # set a new S_variables ,and add variable which match the condition
            S_variables = []
            # print("circulate should continue,so S_variable readd variables")
            BTT_variables =[i for i in range(kVar) if i != target and i not in MB and i not in no_in_set]
            # print(BTT_variables)
            for x in BTT_variables:
                ci_number += 1
                pval, dep = cond_indep_test(data, target, x, MB, is_discrete)
                if pval <= alaph:
                    # print([x,dep])
                    S_variables.append([x,dep])
                    # print("sv is: " + str(S_variables))



    return list(set(MB)), ci_number

# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s500_v4.csv")
# print("the file read")
#
# target = 15
# alaph = 0.01
#
# MBs=fast_IAMB(data,target,alaph)
# print(MBs)


# 500
#
# F1 is: 0.67
# Precision is: 0.79
# Recall is: 0.65
# Distance is: 0.48
# ci_number is: 36.02±177.29
# time is: 2.74±5.20

# 5000

# F1 is: 0.88 ±0.61
# Precision is: 0.88±0.82
# Recall is: 0.92±0.54
# Distance is: 0.18±0.88
# ci_number is: 52.73±132.08
# time is: 19.79±125.25

# F1 is: 0.88
# Precision is: 0.88
# Recall is: 0.92
# Distance is: 0.18
# ci_number is: 52.73
# time is: 19.54



