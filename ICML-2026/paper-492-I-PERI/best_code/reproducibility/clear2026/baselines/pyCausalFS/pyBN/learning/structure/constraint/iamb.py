"""
******************
IAMB STRUCTURE
LEARNING ALGORITHM
******************

The IAMB algorithm - Incremental Association Markov Blanket - proceeds
similar to the Grow-Shrink algorithm, where the growing phase adds
random variables to a given node's markov blanket, and the shrinking
phase performs a second pass of conditional independence tests to
eliminate any variables from the markov blanket for which the given
node is conditionally independent given any other node in the markov
blanket (hence doesn't belong in the markov blanket).

It significantly reduces the complexity compared to PC/GS, and is
normally used in conjuction with the mutual information test for
independence (I do so here) [2].

Lastly, it can also be used as a feature selection algorithm for
classification or general dimensionality reduction - simply return
the markov blanket of a given node, which has been shown by Daphne
Koller to be theoretically optimal for prediction [3]. 

IAMB for Feature Selection (from [1]):	
	Because the Markov blanket of a target attribute T renders
	it statistically independent from all the remaining attributes
	(see the Markov blanket definition below), all information
	that may influence its value is stored in the values
	of the attributes of its Markov blanket. Any attribute
	from the feature set outside its Markov blanket can be effectively
	ignored from the feature set without adversely affecting
	the performance of any classifier that predicts the
	value of T

References
----------
[1] Yaramakala and Maragritis, "Speculative Markov Blanket 
Discovery for Optimal Feature Selection"

[2] Tsarmardinos, et al. "Algorithms for Large Scale 
Markov Blanket Discovery" 

[3] Koller "Toward Optimal Feature Selection."
"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from pyBN.utils.independence_tests import are_independent, mi_test
from pyBN.utils.orient_edges import orient_edges_MB, orient_edges_gs2
from pyBN.utils.markov_blanket import resolve_markov_blanket
from pyBN.classes.bayesnet import BayesNet
from copy import copy

def iamb(data, alpha=0.05, feature_selection=None, debug=False):
	"""
	IAMB Algorithm for learning the structure of a
	Discrete Bayesian Network from data.

	Arguments
	---------
	*data* : a nested numpy array

	*alpha* : a float
		The type II error rate.

	*feature_selection* : None or a string
		Whether to use IAMB as a structure learning
		or feature selection algorithm.

	Returns
	-------
	*bn* : a BayesNet object or
	*mb* : the markov blanket of a node

	Effects
	-------
	None

	Notes
	-----
	- Works but there are definitely some bugs.

	Speed Test:
		*** 5 vars, 624 obs ***
			- 196 ms
	"""
	n_rv = data.shape[1]
	Mb = dict([(rv,[]) for rv in range(n_rv)])

	if feature_selection is None:
		_T = range(n_rv)
	else:
		assert (not isinstance(feature_selection, list)), 'feature_selection must be only one value'
		_T = [feature_selection]

	# LEARN MARKOV BLANKET
	for T in _T:

		V = set(range(n_rv)) - {T}
		Mb_change=True

		# GROWING PHASE
		while Mb_change:
			Mb_change = False
			# find X_max in V-Mb(T)-{T} that maximizes 
			# mutual information of X,T|Mb(T)
			# i.e. max of mi_test(data[:,(X,T,Mb(T))])
			max_val = -1
			max_x = None
			for X in V - set(Mb[T]) - {T}:
				cols = (X,T)+tuple(Mb[T])
				mi_val = mi_test(data[:,cols],test=False)
				if mi_val > max_val:
					max_val = mi_val
					max_x = X
			# if Xmax is dependent on T given Mb(T)
			cols = (max_x,T) + tuple(Mb[T])
			if max_x is not None and are_independent(data[:,cols]):
				Mb[T].append(X)
				Mb_change = True
				if debug:
					print('Adding %s to MB of %s' % (str(X), str(T)))

		# SHRINKING PHASE
		for X in Mb[T]:
			# if x is independent of t given Mb(T) - {x}
			cols = (X,T) + tuple(set(Mb[T]) - {X})
			if are_independent(data[:,cols],alpha):
				Mb[T].remove(X)
				if debug:
					print('Removing %s from MB of %s' % (str(X), str(T)))

	if feature_selection is None:
		# RESOLVE GRAPH STRUCTURE
		edge_dict = resolve_markov_blanket(Mb, data)
		if debug:
			print('Unoriented edge dict:\n %s' % str(edge_dict))
			print('MB: %s' % str(Mb))
		# ORIENT EDGES
		oriented_edge_dict = orient_edges_gs2(edge_dict,Mb,data,alpha)
		if debug:
			print('Oriented edge dict:\n %s' % str(oriented_edge_dict))

		# CREATE BAYESNET OBJECT
		value_dict = dict(zip(range(data.shape[1]),
			[list(np.unique(col)) for col in data.T]))
		bn=BayesNet(oriented_edge_dict,value_dict)

		return bn
	else:
		return Mb[_T]
























