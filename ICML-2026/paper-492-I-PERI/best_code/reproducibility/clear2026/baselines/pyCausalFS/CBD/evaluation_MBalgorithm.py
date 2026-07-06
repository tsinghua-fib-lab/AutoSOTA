# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/18 15:38
desc:
"""
import numpy as np
import pandas as pd
import time
from CBD.MBs.common.realMB import realMB

from CBD.MBs.MMMB.MMMB import MMMB
from CBD.MBs.HITON.HITON_MB import HITON_MB
from CBD.MBs.PCMB.PCMB import PCMB
from CBD.MBs.IPCMB.IPCMB import IPC_MB
from CBD.MBs.GSMB import GSMB
from CBD.MBs.IAMB import IAMB
from CBD.MBs.fast_IAMB import fast_IAMB
from CBD.MBs.inter_IAMB import inter_IAMB
from CBD.MBs.IAMBnPC import IAMBnPC
from CBD.MBs.interIAMBnPC import interIAMBnPC
from CBD.MBs.KIAMB import KIAMB
from CBD.MBs.STMB import STMB
from CBD.MBs.BAMB import BAMB
from CBD.MBs.FBEDk import FBED
from CBD.MBs.MBOR import MBOR
from CBD.MBs.LCMB import LRH


def evaluation(
        method,
        path,
        all_number_Para,
        target_list,
        real_graph_path,
        is_discrete,
        filenumber=10,
        alaph=0.01,
        k=1):

    # pre_set variables is zero
    Precision = 0
    Recall = 0
    F1 = 0
    Distance = 0
    use_time = 0
    ci_number = 0
    realmb, realpc = realMB(all_number_Para, real_graph_path)
    length_targets = len(target_list)
    for m in range(filenumber):
        completePath = path + str(m + 1) + ".csv"
        data = pd.read_csv(completePath)
        number, kVar = np.shape(data)
        ResMB = [[]] * length_targets
        # print("\ndata set is: " + str(m+1) + ".csv")
        for i, target in enumerate(target_list):
            # print("target is: " + str(target))
            if method == "MMMB":
                start_time = time.process_time()
                MB, ci_num = MMMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "IAMB":
                start_time = time.process_time()
                MB, ci_num = IAMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "KIAMB":
                start_time = time.process_time()
                MB, ci_num = KIAMB(data, target, alaph, k, is_discrete)
                end_time = time.process_time()
            elif method == "IAMBnPC":
                start_time = time.process_time()
                MB, ci_num = IAMBnPC(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "inter_IAMB":
                start_time = time.process_time()
                MB, ci_num = inter_IAMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "interIAMBnPC":
                start_time = time.process_time()
                MB, ci_num = interIAMBnPC(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "fast_IAMB":
                start_time = time.process_time()
                MB, ci_num = fast_IAMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "GSMB":
                start_time = time.process_time()
                MB, ci_num = GSMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "HITON_MB":
                start_time = time.process_time()
                MB, ci_num = HITON_MB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "PCMB":
                start_time = time.process_time()
                MB, ci_num = PCMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "IPCMB":
                start_time = time.process_time()
                MB, ci_num = IPC_MB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "STMB":
                start_time = time.process_time()
                MB, ci_num = STMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "IAMBnPC":
                start_time = time.process_time()
                MB, ci_num = IAMBnPC(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "BAMB":
                start_time = time.process_time()
                MB, ci_num = BAMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "FBEDk":
                start_time = time.process_time()
                MB, ci_num = FBED(data, target, k, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "MBOR":
                start_time = time.process_time()
                MB, ci_num = MBOR(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "LRH":
                start_time = time.process_time()
                MB, ci_num = LRH(data, target, alaph, is_discrete)
                end_time = time.process_time()
            else:
                raise Exception("method input error!")

            use_time += (end_time - start_time)
            ResMB[i] = MB
            ci_number += ci_num

        for n, target in enumerate(target_list):
            # print("target is: " + str(target) + " , n is: " + str(n))
            true_positive = list(
                set(realmb[target]).intersection(set(ResMB[n])))
            length_true_positive = len(true_positive)
            length_RealMB = len(realmb[target])
            length_ResMB = len(ResMB[n])
            if length_RealMB == 0:
                if length_ResMB == 0:
                    precision = 1
                    recall = 1
                    distance = 0
                    F1 += 1
                else:
                    F1 += 0
                    precision = 0
                    distance = 2 ** 0.5
                    recall = 0
            else:
                if length_ResMB != 0:
                    precision = length_true_positive / length_ResMB
                    recall = length_true_positive / length_RealMB
                    distance = ((1 - precision) ** 2 + (1 - recall) ** 2) ** 0.5
                    if precision + recall != 0:
                        F1 += 2 * precision * recall / (precision + recall)
                else:
                    F1 += 0
                    precision = 0
                    recall = 0
                    distance = 2 ** 0.5
            Distance += distance
            Precision += precision
            Recall += recall

        # print("current average Precision is: " + str(Precision / ((m+1) * (numberPara))))
        # print("current average Recall is: " + str(Recall / ((m+1) * (numberPara))))

    commonDivisor = length_targets * filenumber

    # 标准差

    return F1 / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, Distance / \
        commonDivisor, ci_number / commonDivisor, use_time / commonDivisor


# test main
if __name__ == '__main__':
    method = input("algorithm name: ")
    K_flag = False
    if method == "KIAMB":
        K = float(input("k: "))
        K_flag = True
    elif method == "FBEDk":
        K = int(input("k: "))
        K_flag = True

    real_graph_path = input("real graph path: ")
    if real_graph_path == "default":
        real_graph_path = "./data/child_graph.txt"

    data_path = input("data: ")
    if data_path == "default":
        data_path = "./data/Child_s5000_v"
    file_number = int(input("file number: "))
    _, num_para = np.shape(pd.read_csv(data_path + '1.csv'))

    list_t = input("target variable index: ").split(",")
    list_target = []
    if list_t[0] == "all":
        list_target = [i for i in range(num_para)]
    else:
        for i in list_t:
            list_target.append(int(i))

    alpha = float(input("alpha: "))
    isdiscrete = input("is_discrete: ")
    if isdiscrete == "1":
        isdiscrete = True
    elif isdiscrete == "0":
        isdiscrete = False
    print("\n")
    if K_flag:
        F1, Precision, Recall, Distance, ci_number, time = evaluation(
            method, data_path, num_para, list_target, real_graph_path, isdiscrete, file_number, alpha, K)
    else:
        F1, Precision, Recall, Distance, ci_number, time = evaluation(
            method, data_path, num_para, list_target, real_graph_path, isdiscrete, file_number, alpha)

    print("F1 is: " + str("%.2f " % F1))
    print("Precision is: " + str("%.2f" % Precision))
    print("Recall is: " + str("%.2f" % Recall))
    print("Distance is: " + str("%.2f" % Distance))
    print("ci_number is: " + str("%.2f" % ci_number))
    print("Running time is: " + str("%.2f" % time))
    with open(r".\output\indicator.txt", "w") as file:
        file.write(str(method) + ": ")
        file.write("F1 is: " + str("%.2f " % F1) + "\n")
        file.write("Precision is: " + str("%.2f" % Precision) + "\n")
        file.write("Recall is: " + str("%.2f" % Recall) + "\n")
        file.write("Distance is: " + str("%.2f" % Distance) + "\n")
        file.write("ci_number is: " + str("%.2f" % ci_number) + "\n")
        file.write("Running time is:" + str("%.2f" % time) + "\n")
