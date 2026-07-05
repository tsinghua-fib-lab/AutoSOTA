#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/30 20:03
 @File    : example_MB.py
 """

from CBD.MBs.MBOR import MBtoPC
from CBD.MBs.pc_simple import pc_simple
from CBD.MBs.MMMB.MMPC import MMPC
from CBD.MBs.PCMB.getPC import getPC
from CBD.MBs.HITON.HITON_PC import HITON_PC
from CBD.MBs.semi_HITON.semi_HITON_PC import semi_HITON_PC
import time
import pandas
import numpy
import re


def example(method, data, list_target, alpha=0.01, is_discrete=True):
    file = open("../output/pc.txt", "w+")
    if method == "MBtoPC":
        _, kVar = numpy.shape(data)
        start_time = time.process_time()
        for target in list_target:
            pc, ci_num = MBtoPC(data, target, alpha, [i for i in range(kVar)], is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc)+"\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    elif method == "pc_simple":
        start_time = time.process_time()
        for target in list_target:
            pc, ci_num = pc_simple(data, target, alpha, is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc)+"\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    elif method == "HITON_PC":
        start_time = time.process_time()
        for target in list_target:
            pc, _, _ = HITON_PC(data, target, alpha, is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc) + "\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    elif method == "MMPC":
        start_time = time.process_time()
        for target in list_target:
            pc, _, _ = MMPC(data, target, alpha, is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc)+"\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    elif method == "getPC":
        start_time = time.process_time()
        for target in list_target:
            pc, _, _ = getPC(data, target, alpha, is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc)+"\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    elif method == "semi_HITON_PC":
        start_time = time.process_time()
        for target in list_target:
            pc, _, _ = semi_HITON_PC(data, target, alpha, is_discrete)
            file.write("the pc of " + str(target) + " is:" + str(pc)+"\n")
            print("the pc of " + str(target) + " is:" + str(pc))
        end_time = time.process_time()
    else:
        raise Exception("method input error!")

    print("the running time is: " + str(end_time - start_time))
    file.write("the running time is: " + str(end_time - start_time) + "\n")
    file.close()


if __name__ == '__main__':
    method = input("algorithm name: ")

    data_path = input("data path: ")
    if data_path == "default":
        data_path = "../data/Child_s5000_v1.csv"
    data = pandas.read_csv(data_path)

    list_t = input("target variable index: ").split(",")
    list_target = []
    if re.match(list_t[0], "all"):
        _, kVar = numpy.shape(data)
        list_target = [i for i in range(kVar)]
    else:
        for i in list_t:
            list_target.append(int(i))
    alpha = float(input("alpha: "))
    isdiscrete = input("discrete data: ")
    if isdiscrete == "1":
        isdiscrete = True
    elif isdiscrete == "0":
        isdiscrete = False
    print("\n")

    example(method, data, list_target, alpha, isdiscrete)
