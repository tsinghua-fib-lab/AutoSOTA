#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/13 21:34
 @File    : independence_condition_test.py
 """

from CBD.MBs.common.chi_square_test import chi_square_test
from CBD.MBs.common.fisher_z_test import cond_indep_fisher_z
from CBD.MBs.common.chi_square_test import chi_square
from CBD.MBs.common.g2test import g2_test_dis


def cond_indep_test(data, target, var, cond_set=[], is_discrete=True, alpha=0.01):
    if is_discrete:
        pval, dep = g2_test_dis(data, target, var, cond_set,alpha)
        # if selected:
        #     _, pval, _, dep = chi_square_test(data, target, var, cond_set, alpha)
        # else:
        # _, _, dep, pval = chi_square(target, var, cond_set, data, alpha)
    else:
        CI, dep, pval = cond_indep_fisher_z(data, target, var, cond_set, alpha)
    return pval, dep
