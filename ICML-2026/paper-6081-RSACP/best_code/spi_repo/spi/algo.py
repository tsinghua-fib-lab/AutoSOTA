######################################################################################################################
# This file contains code adapted from https://github.com/msesia/arc.git
#                   by "Classification with Valid and Adaptive Coverage" (Romano et. al 2020)
#
######################################################################################################################
import numpy as np
import pandas as pd


def eval_prediction_sets(S, X, y):
    coverage = np.mean([y[i] in S[i] for i in range(len(y))])
    length = np.mean([len(S[i]) for i in range(len(y))])
    return coverage, length


def eval_prediction_intervals(S, X, y):
    coverage_list, length_list = [], []
    for i in range(len(y)):
        l_, u_ = S[i]
        coverage_list.append(y[i] >= l_ and y[i] <= u_)
        length_list.append(u_ - l_)
    coverage = np.mean(coverage_list)
    length = np.mean(length_list)
    return coverage, length


def eval_predictions(S, X, y):
    if isinstance(S[0], np.ndarray):
        return eval_prediction_sets(S, X, y)
    elif isinstance(S[0], tuple):
        return eval_prediction_intervals(S, X, y)
    else:
        raise ValueError(f'The following prediction type is not supported - {type(S[0])}')


def get_stats_class_conditional(S, X, y):
    stats_per_class = {}
    classes = np.unique(y)
    for class_ in classes:
        indices = np.where(y == class_)[0]
        coverage = np.mean([y[i] in S[i] for i in indices])
        length = np.mean([len(S[i]) for i in indices])
        stats_per_class[class_] = {'Coverage': coverage, 'Length': length}
    return stats_per_class


def get_results(S, X, y, method, alpha, beta, n_cal, n_cal_maj, n_test, dataset, n_classes_min, n_classes_maj, class_conditional=False, **kwargs):
    results = pd.DataFrame({})
    for alpha_ in S.keys():
        coverage, length = eval_predictions(S[alpha_], X, y)
        curr_result_dict = {'Method': method, 'Coverage': coverage, 'Length': length,
                            'alpha': alpha_,
                            'n_cal': n_cal, 'n_cal_maj': n_cal_maj,
                            'n_test': n_test, 'dataset': dataset, 'beta': beta,
                            'n_classes': n_classes_min, 'n_classes_maj': n_classes_maj,
                            }
        for k,v in kwargs.items():
            if v is not None:
                curr_result_dict[k] = v
        if class_conditional:  # only relevant for ImageNet exp
            n_cal_cond = n_cal // n_classes_min
            n_cal_maj_cond = n_cal_maj // n_classes_maj
            stats_per_class = get_stats_class_conditional(S[alpha_], X, y)
            for class_,class_dict in stats_per_class.items():
                curr_result_dict_cond = curr_result_dict.copy()
                del curr_result_dict_cond['Coverage']
                del curr_result_dict_cond['Length']
                curr_result_dict_cond['Class'] = class_
                curr_result_dict_cond['Conditional Coverage'] = class_dict['Coverage']
                curr_result_dict_cond['Conditional Length'] = class_dict['Length']
                curr_result_dict_cond['n_cal_cond'] = n_cal_cond
                curr_result_dict_cond['n_cal_maj_cond'] = n_cal_maj_cond
                curr_result = pd.DataFrame(curr_result_dict_cond, index=[0])
                results = pd.concat([results, curr_result])
        curr_result = pd.DataFrame(curr_result_dict, index=[0])
        results = pd.concat([results, curr_result])
    return results
