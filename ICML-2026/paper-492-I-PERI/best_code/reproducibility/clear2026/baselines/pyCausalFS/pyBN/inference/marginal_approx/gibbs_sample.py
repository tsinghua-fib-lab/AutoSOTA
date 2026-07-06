
__author__ = """N. Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.classes.bayesnet import BayesNet
from pyBN.classes.factor import Factor 
from pyBN.utils.graph import topsort

import numpy as np



def gibbs_sample(bn, n=1000, burn=200):
	"""
	Approximate Marginal probabilities from Gibbs Sampling
	over a BayesNet object.

	Arguments
	---------
	*bn* : a BayesNet object

	*n* : an integer
		The number of samples to take

	*burn* : an integer
		The number of beginning samples to
		throw away for the MCMC mixing.

	Returns
	-------
	*sample_dict* : a dictionary where key = rv
		and value = another dictionary where
		key = rv instantiation and value = marginal
		probability

	Notes
	-----

	"""

	sample_dict ={}
	for rv in bn.nodes():
	    sample_dict[rv]={}
	    for val in bn.values(rv):
	        sample_dict[rv][val] = 0

	state = {}
	for rv in bn.nodes():
	    state[rv] = np.random.choice(bn.values(rv)) # uniform sample

	for i in range(n):
	    #if i % (n/float(10)) == 0:
	      #  print 'Sample: ' , i
	    for rv in bn.nodes():
	        # get possible values conditioned on everything else
	        parents = bn.parents(rv)
	        # no parents - prior
	        if len(parents) == 0:
	            choice_vals = bn.values(rv)
	            choice_probs = bn.cpt(rv)
	        # has parent - filter cpt
	        else:
	            f = Factor(bn,rv)
	            for p in parents:
	                f.reduce_factor(p,state[p])
	            choice_vals = bn.values(rv)
	            choice_probs = f.cpt
	        # sample over remaining possibilities
	        chosen_val = np.random.choice(choice_vals,p=choice_probs)
	        state[rv]=chosen_val
	    # update sample_dict dictionary
	    if i > burn:
	        for rv,val in state.items():
	            sample_dict[rv][val] +=1

	for rv in sample_dict:
	    for val in sample_dict[rv]:
	        sample_dict[rv][val] = round(int(sample_dict[rv][val]) / float(n-burn),4)
	
	return sample_dict