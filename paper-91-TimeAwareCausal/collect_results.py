import os
import json
import numpy as np


def process_each_file(file_path):
    with open(file_path, 'r') as textFile:
        context = textFile.readlines()
        val_line, test_line = context[-2:]
        val_line, test_line = val_line.strip(), test_line.strip()
        val_line_start = val_line.index('Val  Mean&Var') + len('Val  Mean&Var') + 2
        cur_context = val_line[val_line_start+1: -1]
        vaL_res = list(eval(cur_context))

        test_line_start = test_line.index('Test  Mean&Var') + len('Test  Mean&Var') + 2
        cur_context = test_line[test_line_start+1: -1]
        test_res = list(eval(cur_context))

    return vaL_res, test_res


if __name__ == '__main__':
    file_root = './logs/ToyCircle/MMD_LSAE/'
    # file_root = './logs/ToyCircle_C/MMD_LSAE/'
    # file_root = './logs/RMNIST/MMD_LSAE/'

    files = os.listdir(file_root)
    files.sort()
    precision = 2

    val_res_list, test_res_list = [], []
    file_nums = 0
    for file_name in files:
        # filter out non-relative files
        if file_name.startswith('train') and file_name.endswith('.txt'):

            file_path = os.path.join(file_root, file_name)
            print('Reading file:{}'.format(file_path))
            file_nums += 1

            val_res, test_res = process_each_file(file_path)
            val_res_list.append(val_res)
            test_res_list.append(test_res)
            # if file_nums == 2:
            #      break

    val_mean, val_var, test_mean, test_var = [], [], [], []
    val_iterator = zip(*val_res_list)
    test_iterator = zip(*test_res_list)

    for items in val_iterator:
        mean_list = [mean_value for (mean_value, _) in items]
        var_list = [var_value for (_, var_value) in items]
        val_mean.append(mean_list)
        val_var.append(var_list)

    val_mean_each_domain = np.around(np.mean(val_mean, axis=1), decimals=precision)
    val_var_each_domain = np.around(np.mean(val_var, axis=1), decimals=precision)
    val_res_each_domain = [(mean_value, val_value) for (mean_value, val_value) in zip(val_mean_each_domain, val_var_each_domain)]


    for items in test_iterator:
        mean_list = [mean_value for (mean_value, _) in items]
        var_list = [var_value for (_, var_value) in items]
        test_mean.append(mean_list)
        test_var.append(var_list)

    test_mean_each_domain = np.around(np.mean(test_mean, axis=1), decimals=precision)
    test_var_each_domain = np.around(np.mean(test_var, axis=1), decimals=precision)
    test_res_each_domain = [(mean_value, val_value) for (mean_value, val_value) in zip(test_mean_each_domain, test_var_each_domain)]

    print('=========================================')
    print('Total Val results:{:.2f} ± {:.2f} %'.format(np.mean(val_mean), np.mean(val_var)))
    print('Total Val each domain:{}'.format(val_res_each_domain))

    print('Total Test results:{:.2f} ± {:.2f} %'.format(np.mean(test_mean), np.mean(test_var)))
    test_mean_each_seed = np.around(np.mean(test_mean, axis=0), decimals=precision)
    print('Total Test each seed:{}'.format(test_mean_each_seed))
    print('Total Test each domain:{}'.format(test_res_each_domain))
