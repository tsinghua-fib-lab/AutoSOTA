__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor 
from pyBN.utils.graph import topsort

import numpy as np


def lw_sample(bn, evidence={}, target=None, n=1000):
	"""
	Approximate Marginal probabilities from
	likelihood weighted sample algorithm on
	a BayesNet object.

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	*evidence* : a dictionary, where
		key = rv, value = instantiation

	Returns
	-------
	*sample_dict* : a dictionary where key = rv
		and value = another dictionary where
		key = rv instantiation and value = marginal
		probability

	Effects
	-------
	None

	Notes
	-----

	"""
	sample_dict = {}
	weight_list = np.ones(n)

	#factor_dict = dict([(var,Factor(bn, var)) for var in bn.V])
	#parent_dict = dict([(var, bn.data[var]['parents']) for var in bn.V])

	for var in bn.nodes():
	    sample_dict[var] = {}
	    for val in bn.values(var):
	        sample_dict[var][val] = 0

	for i in range(n):
	    #if i % (n/float(10)) == 0:
	     #   print 'Sample: ' , i
	    new_sample = {}
	    for rv in bn.nodes():
	        f = Factor(bn,rv)
	        # reduce_factor by parent samples
	        for p in bn.parents(rv):
	            f.reduce_factor(p,new_sample[p])
	        # if rv in evidence, choose that value and weight
	        if rv in evidence:
	            chosen_val = evidence[rv]
	            weight_list[i] *= f.cpt[bn.values(rv).index(evidence[rv])]
	        # if rv not in evidence, sample as usual
	        else:
	            choice_vals = bn.values(rv)
	            choice_probs = f.cpt
	            chosen_val = np.random.choice(choice_vals, p=choice_probs)
	            
	        new_sample[rv] = chosen_val
	    # weight the choice by the evidence likelihood    
	    for rv in new_sample:
	        sample_dict[rv][new_sample[rv]] += 1*weight_list[i]

	weight_sum = sum(weight_list)

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] /= weight_sum
	        sample_dict[rv][val] = round(sample_dict[rv][val],4)
	
	if target is not None:
		return sample_dict[target]
	else:
		return sample_dict