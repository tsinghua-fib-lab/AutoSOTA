#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/30 20:03
 @File    : example_MB.py
 """

from SSD.MBs.SLLMB import SLL
from SSD.MBs.S2TMB import S2TMB
from SSD.MBs.S2TMB import S2TMB_p
import time
import pandas
import numpy


def example(method, data, list_target):
    file = open("../output/outputSSD.txt", "w+")
    if method == "SLL":
        start_time = time.process_time()
        for target in list_target:
            PC, MB = SLL(data, target)
            file.write(str(target) + " PC: " + str(PC) + " , MB: " + str(MB)  + " .\n")
            print(str(target) + " PC: " + str(PC) + " , MB: " + str(MB))
        end_time = time.process_time()
    elif method == "S2TMB":
        start_time = time.process_time()
        for target in list_target:
            PC, MB = S2TMB(data, target)
            file.write(str(target) + " PC: " + str(PC) + " , MB: " + str(MB) + ".\n")
            print(str(target) + " PC: " + str(PC) + " , MB: " + str(MB))
        end_time = time.process_time()
    elif method == "S2TMB_p":
        start_time = time.process_time()
        for target in list_target:
            PC, MB = S2TMB_p(data, target)
            file.write(str(target) + " PC: " + str(PC) + " , MB: " + str(MB) + ".\n")
            print(str(target) + " PC: " + str(PC) + " , MB: " + str(MB))
        end_time = time.process_time()
    else:
        raise Exception("method input error!")

    print("the running time is: " + str(end_time - start_time))
    file.write("the running time is: " + str(end_time - start_time) + "\n")
    file.close()


if __name__ == '__main__':
    method = input("algorithm name: ")

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
    print("\n")

    example(method, data, list_target)

