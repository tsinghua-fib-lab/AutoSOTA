from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor 
from pyBN.utils.graph import topsort

import numpy as np


def forward_sample(bn, n=1000):
	"""
	Approximate marginal probabilities from
	forward sampling algorithm on a BayesNet object.

	This algorithm works by
	repeatedly sampling from the BN and taking
	the ratio of observations as their marginal probabilities.

	One sample is done by first sampling from any prior random
	variables, then moving down the network in topological sort
	order - sampling from each successive random variable by
	conditioning on its parents (which have already been sampled
	higher up the network).

	Note that there is no evidence to include in this algorithm - 
	the comparative algorithm which includes evidence is the 
	likelihood weighted algorithm (see "lw_sample" function).

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	Returns
	-------
	*sample_dict* : a dictionary, where key = rv, value = another dict
					where key = instance, value = its probability value

	Notes
	-----
	- Evidence is not currently implemented.
	"""

	sample_dict = {}
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
	        for p in bn.parents(rv):
	            f.reduce_factor(p,new_sample[p])
	        choice_vals = bn.values(rv)
	        choice_probs = f.cpt
	        chosen_val = np.random.choice(choice_vals, p=choice_probs)

	        sample_dict[rv][chosen_val] += 1
	        new_sample[rv] = chosen_val

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] = int(sample_dict[rv][val]) / float(n)
	
	return sample_dict