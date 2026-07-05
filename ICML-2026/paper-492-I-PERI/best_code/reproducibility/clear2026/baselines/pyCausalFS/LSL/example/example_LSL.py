#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/30 20:03
 @File    : example_MB.py
 """

from LSL.MBs.PCDbyPCD import PCDbyPCD
from LSL.MBs.MB_by_MB import MB_by_MB
from LSL.MBs.CMB.CMB import CMB
import time
import pandas
import numpy


def example(method, data, list_target, alpha, is_discrete):
    file = open("../output/outputLSL.txt", "w+")
    if method == "PCDbyPCD":
        start_time = time.process_time()
        for target in list_target:
            parents, children, undirected = PCDbyPCD(data, target, alpha, is_discrete)
            file.write(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected) +".\n")
            print(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected))
        end_time = time.process_time()
    elif method == "MBbyMB":
        start_time = time.process_time()
        for target in list_target:
            parents, children, undirected = MB_by_MB(data, target, alpha, is_discrete)
            file.write(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected) +".\n")
            print(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected))
        end_time = time.process_time()
    elif method == "CMB":
        start_time = time.process_time()
        for target in list_target:
            parents, children, undirected = CMB(data, target, alpha, is_discrete)
            file.write(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected) +".\n")
            print(str(target) + " parents: " + str(parents) + " ,children: "+ str(children) + " ,undirected: "+str(undirected))
        end_time = time.process_time()
    else:
        raise Exception("method input error!")

    print("the Running time is: " + str(end_time - start_time))
    file.write("the Running time is: " + str(end_time - start_time) + "\n")
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
    alpha = float(input("alpha: "))
    isdiscrete = input("is_discrete: ")
    if isdiscrete == "1":
        isdiscrete = True
    elif isdiscrete == "0":
        isdiscrete = False
    print("\n")

    example(method, data, list_target, alpha, isdiscrete)

# C:\pythonProject\pyCausalFS\CBD\data\Child_s500_v1.csv
