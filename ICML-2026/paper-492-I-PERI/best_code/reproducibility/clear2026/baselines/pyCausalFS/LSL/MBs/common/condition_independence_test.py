#!/usr/bin/env python
# encoding: utf-8
"""
 @Time    : 2019/11/13 21:34
 @File    : independence_condition_test.py
 """

"""from CBD.MBs.common.chi_square_test import chi_square_test
from CBD.MBs.common.fisher_z_test import cond_indep_fisher_z
from CBD.MBs.common.chi_square_test import chi_square
from CBD.MBs.common.g2test import g2_test_dis"""


from baselines.pyCausalFS.CBD.MBs.common.chi_square_test import chi_square_test
from baselines.pyCausalFS.CBD.MBs.common.fisher_z_test import cond_indep_fisher_z
from baselines.pyCausalFS.CBD.MBs.common.chi_square_test import chi_square
from baselines.pyCausalFS.CBD.MBs.common.g2test import g2_test_dis


def cond_indep_test(data, target, var, cond_set=[], is_discrete=True, alpha=0.01):
    if is_discrete:
        pval, dep = g2_test_dis(data, target, var, cond_set,alpha)
        # if selected:
        #     _, pval, _, dep = chi_square_test(data, target, var, cond_set, alpha)
        # else:
        # _, _, dep, pval = chi_square(target, var, cond_set, data, alpha)
    else:
        _, _, pval = cond_indep_fisher_z(data, target, var, cond_set, alpha)
        if pval >= alpha:
            dep = - (1 - pval)
        else:
            dep = 1 - pval

    return pval, dep

def dict_cache_test(dict_cache,data, target, var, cond_set,*args,**kwargs):
    na = sorted([target, var])
    if len(cond_set):
        na.extend(sorted(cond_set))
    akey = "_".join('%s' % id for id in na)
    # print("akey: ", akey)
    if akey in dict_cache.keys():
        pvalue, dep = dict_cache[akey]
        print("cache get it!")
        print("akey: ", akey)
    else:
        pvalue, dep = cond_indep_test(data, target, var, cond_set, *args, **kwargs)
        dict_cache.setdefault(akey, [pvalue, dep])
    return  dict_cache, pvalue, dep
