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
from CBD.MBs.MMMB import MMMB
from CBD.MBs.HITON import HITON_MB
from CBD.MBs import PCMB
from CBD.MBs import IPC_MB
from CBD.MBs.GSMB import GSMB
from CBD.MBs import IAMB
from CBD.MBs.fast_IAMB import fast_IAMB
from CBD.MBs import inter_IAMB
from CBD.MBs import IAMBnPC
from CBD.MBs import STMB
from CBD.MBs import BAMB
from CBD.MBs import FBED
from CBD.MBs import MBOR
from CBD.MBs.LCMB import LRH


def evaluation(method, path, numberPara, filenumber=10, alaph=0.01, k=1):
    # pre_set variables is zero
    Precision = 0
    Recall = 0
    F1 = 0
    Distance = 0
    use_time = 0
    ci_number = 0
    realmb, realpc = realMB(numberPara)
    for m in range(filenumber):
        completePath = path + str(m+1) + ".csv"
        data = pd.read_csv(completePath)
        number, kVar = np.shape(data)
        ResMB = [[] for i in range(kVar)]
        print("\ndata set is: " + str(m+1) + ".csv")
        for target in range(kVar):
            print("target is: " + str(target))
            if method == "MMMB":
                start_time = time.process_time()
                MB, ci_num = MMMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "IAMB":
                start_time = time.process_time()
                MB, ci_num = IAMB(data, target, alaph)
                end_time = time.process_time()
            elif method =="inter_IAMB":
                start_time = time.process_time()
                MB, ci_num = inter_IAMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "fast_IAMB":
                start_time = time.process_time()
                MB, ci_num = fast_IAMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "GSMB":
                start_time = time.process_time()
                MB, ci_num = GSMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "HITON_MB":
                start_time = time.process_time()
                MB, ci_num = HITON_MB(data, target, alaph)
                end_time = time.process_time()
            elif method == "PCMB":
                start_time = time.process_time()
                MB, ci_num = PCMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "IPCMB":
                start_time = time.process_time()
                MB, ci_num = IPC_MB(data, target, alaph)
                end_time = time.process_time()
            elif method == "STMB":
                start_time = time.process_time()
                MB, ci_num = STMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "IAMBnPC":
                start_time = time.process_time()
                MB, ci_num = IAMBnPC(data, target, alaph)
                end_time = time.process_time()
            elif method == "BAMB":
                start_time = time.process_time()
                MB, ci_num = BAMB(data, target, alaph)
                end_time = time.process_time()
            elif method == "FBED":
                start_time = time.process_time()
                MB, ci_num = FBED(data, target, k, alaph)
                end_time = time.process_time()
            elif method == "MBOR":
                start_time = time.process_time()
                MB, ci_num = MBOR(data, target, alaph)
                end_time = time.process_time()
            elif method == "LRH":
                start_time = time.process_time()
                MB, ci_num = LRH(data, target, alaph)
                end_time = time.process_time()
            else:
                raise Exception("method input error!")

            use_time += (end_time - start_time)
            ResMB[target] = MB
            ci_number += ci_num

        for n in range(kVar):
            true_positive = list(set(realmb[n]).intersection(set(ResMB[n])))
            length_true_positive = len(true_positive)
            length_RealMB = len(realmb[n])
            length_ResMB = len(ResMB[n])
            if length_RealMB == 0:
                if length_ResMB == 0:
                    precision = 1
                    recall = 1
                    F1 += 1
                else:
                    F1 += 0
                    precision = 0
                    recall = 0
            else:
                if length_ResMB != 0:
                    precision = length_true_positive / length_ResMB
                    recall = length_true_positive / length_RealMB
                    if precision + recall != 0:
                        F1 += 2 * precision * recall / (precision + recall)
                else:
                    F1 += 0
                    precision = 0
                    recall = 0
            distance = ((1 - precision) ** 2 + (1 - recall)**2) ** 0.5
            Distance += distance
            Precision += precision
            Recall += recall

        print("current average Precision is: " + str(Precision / ((m+1) * (numberPara))))
        print("current average Recall is: " + str(Recall / ((m+1) * (numberPara))))

    commonDivisor = numberPara * filenumber

    # 标准差

    return F1 / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, Distance/commonDivisor, \
           ci_number / commonDivisor , use_time / commonDivisor


# test main
if __name__ == '__main__':
    file = open("../resultFile/result.txt", "a+")
    path = "../data/Child_s500_v"
    # 如果要测试多个算法,在下面列表添加算法名
    methodSet = ["LRH"]
    for method in methodSet:
        # method = input("please input name of MBs algorithm! ")


        print("start run " + method + " algorithm")
        F1, Precision, Recall, Distance, ci_number, time = evaluation(method, path, 20, k=1)
        print("F1 is: " + str("%.2f " % F1))
        print("Precision is: " + str("%.2f" % Precision))
        print("Recall is: " + str("%.2f" % Recall))
        print("Distance is: " + str("%.2f" % Distance))
        print("ci_number is: " + str("%.2f" % ci_number))
        print("time is: " + str("%.2f" % time))

        file.write(str(method) + ":")
        file.write("F1 is: " + str("%.2f " % F1))
        file.write("Precision is: " + str("%.2f" % Precision))
        file.write("Recall is: " + str("%.2f" % Recall))
        file.write("Distance is: " + str("%.2f" % Distance))
        file.write("ci_number is: " + str("%.2f" % ci_number))
        file.write("time is:" + str("%.2f" % time))
        file.write("\n\n")

    # file = open("../resultFile/result.txt", "a+")
    # methodSet = ["IAMB", "fast_IAMB", "inter_IAMB", "IAMBnPC", "GSMB", "MMMB", "PCMB", "IPCMB", "HITON_MB", "STMB"]
    # for method in methodSet:
    #     F1, Precision, Recall, time = evaluation(method, path, 20)
    #     print("F1 is: " + str(F1))
    #     print("Precision is: " + str(Precision))
    #     print("Recall is: " + str(Recall))
    #     print("Distance is: " + str(Distance))
    #     print("ci_number is: " + str(ci_number))
    #     print("time is: " + str(time))
    #
    #     file.write(str(method) + ":")
    #     file.write("F1 is: " + str(F1))
    #     file.write("Precision is: " + str(Precision))
    #     file.write("Recall is: " + str(Recall))
    #     file.write("Distance is: " + str(Distance))
    #     file.write("ci_number is: " + str(ci_number))
    #     file.write("time is:" + str(time))
    #     file.write("\n\n")


