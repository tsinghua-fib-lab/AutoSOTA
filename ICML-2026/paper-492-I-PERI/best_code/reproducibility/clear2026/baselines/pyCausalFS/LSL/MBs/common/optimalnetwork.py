#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/9/29 20:43
 @File    : optimalnetwork.py
 """

import numpy as np
import SSD.MBs.pyBN.learning.structure.score.hill_climbing as hill_climbing
def optimal_network(Z, data):
    Z = sorted(Z)
    _, kVar = np.shape(data)
    DAG = np.zeros((kVar, kVar))
    data_array = np.array(data,dtype=np.int_)
    while kVar > 0:
        kVar -= 1
        if kVar not in Z:
           data_array = np.delete(data_array, kVar, axis=1)

    z_dict = hill_climbing.hc(data_array, metric="BIC")
    c_dict = dict()
    for key,value in z_dict.items():
        if value == []:
            c_dict.setdefault(Z[key], [])
        else:
            c_list = []
            for i in value:
                c_list.append(Z[i])
                DAG[Z[key], Z[i]] = 1
            c_dict.setdefault(Z[key], c_list)
    return DAG

# import pandas as pd
#
# data = pd.read_csv("C:/pythonProject/pyCausalFS/data/child_s5000_v2.csv")
# print(data)
# print("the file read")
# print("data is: " + str(data))
# z = [1,19,10,8]
# res = optimalnetwork(z, data)
# print("res is: " + str(res))