# coding=utf-8
# /usr/bin/env python
"""
date: 2019/7/18 15:38
desc:
"""
import numpy as np
import pandas as pd
from LSL.MBs.common.real_P_C_S import real_p_c_s
from LSL.MBs.PCDbyPCD import PCDbyPCD
from LSL.MBs.MBbyMB import MBbyMB
from LSL.MBs.CMB.CMB import CMB
import time

def evaluation_nocache(method,
               path,
               all_number_Para,
               target_list,
               real_graph_path,
               is_discrete,
               filenumber=10,
               alaph=0.01,
               ):
    # pre_set variables is zero
    length_target_list = len(target_list)
    real_p, real_c, real_s = real_p_c_s(all_number_Para, real_graph_path)
    num_re, num_undirect = 0, 0
    num_miss ,num_extra= 0, 0
    num_true = 0
    all_time = 0
    num_ci = 0
    for m in range(filenumber):
        completePath = path + str(m + 1) + ".csv"
        data = pd.read_csv(completePath)
        get_p, get_c, get_un = [[]] * length_target_list, [[]] * \
            length_target_list, [[]] * length_target_list

        for i, target in enumerate(target_list):
            if method == "PCDbyPCD":
                start_time = time.process_time()
                parents, children, PC, undirected, n_c = PCDbyPCD(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "MBbyMB":
                start_time = time.process_time()
                parents, children, PC, undirected, n_c = MBbyMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            elif method == "CMB":
                start_time = time.process_time()
                parents, children, PC, undirected, n_c = CMB(data, target, alaph, is_discrete)
                end_time = time.process_time()
            else:
                raise Exception("method input error!")
            get_p[i] = parents
            get_c[i] = children
            get_un[i] = undirected
            all_time += end_time - start_time
            num_ci += n_c

        print("use time:", all_time)

        for n, target in enumerate(target_list):

            true_diection = list((set(real_p[target]).intersection(set(get_p[n]))).union(set(real_c[target]).intersection(set(get_c[n]))))
            num_true += len(true_diection)

            reverse_direction = list((set(real_p[target]).intersection(
                set(get_c[n]))).union(set(real_c[target]).intersection(set(get_p[n]))))
            num_re += len(reverse_direction)

            undirected_direction = list(get_un[n])
            num_undirect += len(undirected_direction)

            miss_direction = list(((set(real_p[target]).difference(set(get_p[n]))).union(
                set(real_c[target]).difference(set(get_c[n])))).difference(set(reverse_direction).union(undirected_direction)))
            num_miss += len(miss_direction)

            extra_direction = list(((set(get_p[n]).difference(real_p[target])).union(
                set(get_c[n]).difference(set(real_c[target])))))
            num_extra += len(extra_direction)

    commonDivisor = length_target_list * filenumber

    return num_true / commonDivisor, num_re / commonDivisor, num_miss / commonDivisor, num_extra / commonDivisor, num_undirect / commonDivisor,num_ci / commonDivisor, all_time / commonDivisor


# test main
if __name__ == '__main__':
    #
    method_list = ["PCDbyPCD", "MBbyMB", "CMB"]
    data_path = "D:/data/Alarm_data/Alarm1_s5000_v"
    alpha = 0.01
    isdiscrete = True
    real_graph_path = "D:/data/Alarm_data/Alarm1_graph.txt"
    for method in method_list:
        pre_data = pd.read_csv('D:/data/Alarm_data/Alarm1_s5000_v1.csv')
        _, num_para = np.shape(pre_data)
        list_target = [i for i in range(11) if i != 0]
        isdiscrete = True
        file_number = 5
        print("method: ", method)
        num_true, num_re, num_miss, num_extra, num_undirected, num_ci, use_time = evaluation_nocache(
            method, data_path, num_para, list_target, real_graph_path, isdiscrete, file_number, alpha)
        print("true direction is: ", str("%.3f" % num_true))
        print("reverse is: ", str("%.3f" % num_re))
        print("miss is: ", str("%.3f" % num_miss))
        print("extra is: ", str("%.3f" % num_extra))
        print("undirected is:", str("%.3f" % num_undirected))
        print("num_ci is:", str("%.3f" % num_ci))
        print("time is:", str("%.3f" % use_time))
        with open(r".\output\outputLSL.txt", "a+") as file:
            file.write(str(method) + ": \n")
            file.write("path: " + data_path + ".\n")
            file.write("true direction is: " + str("%.3f" % num_true) + "\n")
            file.write("reverse is: " + str("%.3f" % num_re) + "\n")
            file.write("miss is: " + str("%.3f" % num_miss) + "\n")
            file.write("extra is: " + str("%.3f" % num_extra) + "\n")
            file.write("undirected is: " + str("%.3f" % num_undirected) + "\n")
            file.write("num_ci is:" + str("%.3f" % num_ci) + "\n")
            file.write("time is:" + str("%.3f" % use_time) + "\n")