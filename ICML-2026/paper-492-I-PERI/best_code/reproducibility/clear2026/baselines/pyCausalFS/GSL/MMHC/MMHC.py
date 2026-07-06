#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2020/11/4 20:40
 @File    : mmhc_new.py
 """

from GSL.MMHC.hc import hc
from CBD.MBs.MMMB.MMPC import MMPC
import numpy as np

# symmetry check for pc set
def symmetry(pc):

    for var in pc:
        pc_remove = []
        for par in pc[var]:
            if var not in pc[par]:
                pc_remove.append(par)
        if pc_remove:
            for par in pc_remove:
                pc[var].remove(par)
    return pc


def MMHC(data, alpha=0.01, score_function='bdeu'):
    # input:
    # data: input training data, the data must be discrete
    # score: the type of score function, currently support 'bdeu', 'bic'
    # threshold: threshold for CI test
    # output:
    # dag: a direct graph

    _, kvar = np.shape(data)
    DAG = np.zeros((kvar, kvar))
    pc = {}
    num_CI = 0
    for tar in range(kvar):
        pc_mm, _, n_c = MMPC(data, tar, alpha, True)
        num_CI += n_c
        for i in pc_mm:
            DAG[tar, i] = 1
            DAG[i, tar] = 1
        pc[str(tar)] = [str(i) for i in pc_mm]
    # check the symmetry of pc set
    # when the number of variables is large, this function may be computational costly
    # this function can be merged into the pruning process during forward and backward mmpc by transmitting the whole
    # pc set into mmpc_forward and mmpc_backward
    pc = symmetry(pc)
    # run hill-climbing

    dag_dict = hc(data, pc, score_function)

    # orient the edge
    for key, value in dag_dict.items():
        x = int(key)
        for i in value['parents']:
            y = int(i)
            DAG[y, x] = -1
            DAG[x, y] = 0
        for i in value['children']:
            z = int(i)
            DAG[x, z] = -1
            DAG[z, x] = 0

    return DAG, num_CI


# import pandas as pd
#
# from CBD.MBs.common.realMB import realMB
# from LSL.MBs.common.real_P_C_S import real_p_c_s
# if __name__ == '__main__':
#
#     data = pd.read_csv('D:/data/child_data/Child_s5000_v2.csv')
#     real_graph_path = "D:/data/child_data/Child_graph.txt"
#     _, all_number = np.shape(data)
#     real_p, real_c, real_s = real_p_c_s(all_number, real_graph_path)
#     print("real_p is:", real_p)
#     print("real_c is:", real_c)
#     DAG = MMHC(data)
#     print(DAG)
#     with open(r".\result.txt", "a+") as file:
#         file.write(str(DAG))