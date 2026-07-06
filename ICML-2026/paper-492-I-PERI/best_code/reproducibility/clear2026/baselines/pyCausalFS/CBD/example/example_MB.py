#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/30 20:03
 @File    : example_MB.py
 """

from CBD.MBs.MMMB.MMMB import MMMB
from CBD.MBs.HITON.HITON_MB import HITON_MB
from CBD.MBs.semi_HITON.semi_HITON_MB import semi_HITON_MB
from CBD.MBs.PCMB.PCMB import PCMB
from CBD.MBs.IPCMB.IPCMB import IPC_MB
from CBD.MBs.GSMB import GSMB
from CBD.MBs.IAMB import IAMB
from CBD.MBs.KIAMB import KIAMB
from CBD.MBs.fast_IAMB import fast_IAMB
from CBD.MBs.inter_IAMB import inter_IAMB
from CBD.MBs.IAMBnPC import IAMBnPC
from CBD.MBs.interIAMBnPC import interIAMBnPC
from CBD.MBs.STMB import STMB
from CBD.MBs.BAMB import BAMB
from CBD.MBs.FBEDk import FBED
from CBD.MBs.MBOR import MBOR
from CBD.MBs.LCMB import LRH
from CBD.MBs.TIE_star.TIEs import TIE
from CBD.MBs.TIE_star.TIEs import TIE_p
import time
import pandas
import numpy


def example(method, data, list_target, alpha, is_discrete, k=0):
    file = open("../output/mb.txt", "w+")
    if method == "MMMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = MMMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "IAMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = IAMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "inter_IAMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = inter_IAMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "fast_IAMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = fast_IAMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "GSMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = GSMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "HITON_MB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = HITON_MB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "semi_HITON_MB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = semi_HITON_MB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "PCMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = PCMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "IPCMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = IPC_MB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "STMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = STMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "IAMBnPC":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = IAMBnPC(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "interIAMBnPC":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = interIAMBnPC(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "BAMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = BAMB(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "FBEDk":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = FBED(data, target, k, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "MBOR":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = MBOR(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "LRH":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = LRH(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "KIAMB":
        start_time = time.process_time()
        for target in list_target:
            MB, ci_num = KIAMB(data, target, alpha, k, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "TIE":
        start_time = time.process_time()
        for target in list_target:
            MB = TIE(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    elif method == "TIE_p":
        start_time = time.process_time()
        for target in list_target:
            MB = TIE_p(data, target, alpha, is_discrete)
            file.write("the MB of " + str(target) + " is:" + str(MB) + "\n")
            print("the MB of " + str(target) + " is:" + str(MB))
        end_time = time.process_time()
    else:
        raise Exception("method input error!")

    print("the running time is: " + str(end_time - start_time))
    file.write("the running time is: " + str(end_time - start_time) + "\n")
    file.close()


if __name__ == '__main__':
    method = input("algorithm name: ")
    K_flag = False
    if method == "KIAMB":
        K = float(input("k: "))
        K_flag = True
    elif method == "FBEDk":
        K = int(input("k: "))
        K_flag = True

    data_path = input("data: ")
    if data_path == "default":
        data_path = "../data/Child_s5000_v1.csv"
    data = pandas.read_csv(data_path)

    list_t = input("target variable index: ").split(",")
    list_target = []
    if list_t[0] == "all":
        _, kVar = numpy.shape(data)
        list_target = [i for i in range(kVar)]
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
        example(method, data, list_target, alpha, isdiscrete, K)
    else:
        example(method, data, list_target, alpha, isdiscrete)

# C:\pythonProject\pyCausalFS\CBD\data\Child_s500_v1.csv
