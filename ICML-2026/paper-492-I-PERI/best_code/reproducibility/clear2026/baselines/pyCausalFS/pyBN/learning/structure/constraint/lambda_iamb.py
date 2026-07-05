"""
Lambda IAMB Code

References
----------
[1] Zhang et al, "An Improved IAMB Algorithm for Markov
Blanket Discovery"
"""

import numpy as np
from pyBN.utils.independence_tests import are_independent, entropy

def lambda_iamb(data, L=1.5, alpha=0.05, feature_selection=None):
	"""
	Lambda IAMB Algorithm for learning the structure of a
	Discrete Bayesian Network from data. This Algorithm
	is similar to the iamb algorithm, except that it allows
	for a "lambda" coefficient that helps avoid false positives.

	This algorithm was originally developed for use as a
	feature selection algorithm - discovering the markov
	blanket of a target variable is equivalent to discovering
	the relevant features for classifications.

	In practice, this algorithm does just as well as a feature
	selection method compared to IAMB when naive bayes was 
	used as a classifier, but Lambda-iamb actually does much
	better than traditional iamb when traditional iamb does
	very poorly due to high false positive rates.

	Arguments
	---------
	*data* : a nested numpy array

	*L* : a float
		The lambda hyperparameter - see [1].

	*alpha* : a float
		The type II error rate.

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----
	"""
	n_rv = data.shape[1]
	Mb = dict([(rv,{}) for rv in range(n_rv)])

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
			cols = tuple({T}) + tuple(Mb[T])
			H_tmb = entropy(data[:,cols])
			# find X1_min in V-Mb[T]-{T} that minimizes
			# entropy of T|X1_inMb[T]
			# i.e. min of entropy(data[:,(T,X,Mb[T])])
			min_val1, min_val2 = 1e7,1e7
			min_x1, min_x2 = None, None
			for X in V - Mb[T] - {T}:
				cols = (T,X)+tuple(Mb[T])
				ent_val = entropy(data[:,cols])
				if ent_val < min_val:
					min_val2, min_val1 = min_val1, ent_val
					min_x2,min_x1 = min_x1, X
					
			# if min_x1 is dependent on T given Mb[T]...
			cols = (min_x1,T) + tuple(Mb[T])
			if are_independent(data[:,cols]):
				if (min_val2 - L*min_val1) < ((1-L)*H_tmb):
					cols = (min_x2,T)+tuple(Mb[T])
					if are_independent(data[:,cols]):
						Mb[T].add(min_x1)
						Mb[T].add(min_x2)
						Mb_change = True
			else:
				Mb[T].add(X)
				Mb_change = True

		# SHRINKING PHASE
		for X in Mb[T]:
			# if x is indepdent of t given Mb[T] - {x}
			cols = (X,T) + tuple(Mb[T]-{X})
			if mi_test(data[:,cols]) > alpha:
				Mb[T].remove(X)

	if feature_selection is None:
		# RESOLVE GRAPH STRUCTURE
		edge_dict = resolve_markov_blanket(Mb, data)

		# ORIENT EDGES
		oriented_edge_dict = orient_edges_Mb(edge_dict,Mb,data,alpha)

		# CREATE BAYESNET OBJECT
		value_dict = dict(zip(range(data.shape[1]),
			[list(np.unique(col)) for col in data.T]))
		bn=BayesNet(oriented_edge_dict,value_dict)

		return bn
	else:
		return Mb[_T]








