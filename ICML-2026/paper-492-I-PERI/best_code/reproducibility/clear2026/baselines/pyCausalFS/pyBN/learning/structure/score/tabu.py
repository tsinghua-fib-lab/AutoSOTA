"""
******************
Tabu Search for
Score-based 
Structure Learning
******************

Tabu search in this case proceeds by the traditional
hill-climbing algorithm, with the additional feature
that it does not allow the algorithm to reverse any
moves that happened in the last K steps. The idea
behind tabu search is that it may give the algorithm
a better chance of escaping out of local maxima or
plateaus. Still, it is a local search technique.

"""

import numpy as np
from copy import copy, deepcopy

from pyBN.classes.bayesnet import BayesNet
from pyBN.learning.parameter.mle import mle_estimator
from pyBN.learning.parameter.bayes import bayes_estimator
from pyBN.learning.structure.score.info_scores import info_score
from pyBN.utils.independence_tests import mutual_information
from pyBN.utils.graph import would_cause_cycle


def tabu(data, k=5, metric='AIC', max_iter=100, debug=False, restriction=None):
	"""
	Tabu search for score-based structure learning.

	The algorithm maintains a list called "tabu_list",
	which consists of 3-tuples, where the first two
	elements constitute the edge which is tabued, and
	the third element is a string - either 'Addition',
	'Deletion', or 'Reversal' denoting the operation
	associated with the edge.

	Arguments
	---------
	*data* : a nested numpy array
		The data from which the Bayesian network
		structure will be learned.

	*metric* : a string
		Which score metric to use.
		Options:
			- AIC
			- BIC / MDL
			- LL (log-likelihood)

	*max_iter* : an integer
		The maximum number of iterations of the
		hill-climbing algorithm to run. Note that
		the algorithm will terminate on its own if no
		improvement is made in a given iteration.

	*debug* : boolean
		Whether to print(the scores/moves of the)
		algorithm as its happening.

	*restriction* : a list of 2-tuples
		For MMHC algorithm, the list of allowable edge additions.

	Returns
	-------
	*bn* : a BayesNet object
	
	"""
	nrow = data.shape[0]
	ncol = data.shape[1]
	
	names = range(ncol)

	# INITIALIZE NETWORK W/ NO EDGES
	# maintain children and parents dict for fast lookups
	c_dict = dict([(n,[]) for n in names])
	p_dict = dict([(n,[]) for n in names])
	
	# COMPUTE INITIAL LIKELIHOOD SCORE	
	value_dict = dict([(n, np.unique(data[:,i])) for i,n in enumerate(names)])
	bn = BayesNet(c_dict)
	mle_estimator(bn, data)
	max_score = info_score(bn, nrow, metric)

	tabu_list = [None]*k


	_iter = 0
	improvement = True

	while improvement:
		improvement = False
		max_delta = 0

		if debug:
			print('ITERATION: ' , _iter)

		### TEST ARC ADDITIONS ###
		for u in bn.nodes():
			for v in bn.nodes():
				# CHECK TABU LIST - can't delete an addition on the tabu list
				if (u,v,'Deletion') not in tabu_list:
					# CHECK EDGE EXISTENCE AND CYCLICITY
					if v not in c_dict[u] and u!=v and not would_cause_cycle(c_dict, u, v):
						# FOR MMHC ALGORITHM -> Edge Restrictions
						if restriction is None or (u,v) in restriction:
							# SCORE FOR 'V' -> gaining a parent
							old_cols = (v,) + tuple(p_dict[v]) # without 'u' as parent
							mi_old = mutual_information(data[:,old_cols])
							new_cols = old_cols + (u,) # with'u' as parent
							mi_new = mutual_information(data[:,new_cols])
							delta_score = nrow * (mi_old - mi_new)

							if delta_score > max_delta:
								if debug:
									print('Improved Arc Addition: ' , (u,v))
									print('Delta Score: ' , delta_score)
								max_delta = delta_score
								max_operation = 'Addition'
								max_arc = (u,v)

		### TEST ARC DELETIONS ###
		for u in bn.nodes():
			for v in bn.nodes():
				# CHECK TABU LIST - can't add back a deletion on the tabu list
				if (u,v,'Addition') not in tabu_list:
					if v in c_dict[u]:
						# SCORE FOR 'V' -> losing a parent
						old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
						mi_old = mutual_information(data[:,old_cols])
						new_cols = tuple([i for i in old_cols if i != u]) # without 'u' as parent
						mi_new = mutual_information(data[:,new_cols])
						delta_score = nrow * (mi_old - mi_new)

						if delta_score > max_delta:
							if debug:
								print('Improved Arc Deletion: ' , (u,v))
								print('Delta Score: ' , delta_score)
							max_delta = delta_score
							max_operation = 'Deletion'
							max_arc = (u,v)

		### TEST ARC REVERSALS ###
		for u in bn.nodes():
			for v in bn.nodes():
				# CHECK TABU LIST - can't reverse back a reversal on the tabu list
				if (u,v,'Reversal') not in tabu_list:
					if v in c_dict[u] and not would_cause_cycle(c_dict,v,u, reverse=True):
						# SCORE FOR 'U' -> gaining 'v' as parent
						old_cols = (u,) + tuple(p_dict[v]) # without 'v' as parent
						mi_old = mutual_information(data[:,old_cols])
						new_cols = old_cols + (v,) # with 'v' as parent
						mi_new = mutual_information(data[:,new_cols])
						delta1 = nrow * (mi_old - mi_new)
						# SCORE FOR 'V' -> losing 'u' as parent
						old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
						mi_old = mutual_information(data[:,old_cols])
						new_cols = tuple([u for i in old_cols if i != u]) # without 'u' as parent
						mi_new = mutual_information(data[:,new_cols])
						delta2 = nrow * (mi_old - mi_new)
						# COMBINED DELTA-SCORES
						delta_score = delta1 + delta2

						if delta_score > max_delta:
							if debug:
								print('Improved Arc Reversal: ' , (u,v))
								print('Delta Score: ' , delta_score)
							max_delta = delta_score
							max_operation = 'Reversal'
							max_arc = (u,v)


		### DETERMINE IF/WHERE IMPROVEMENT WAS MADE ###
		if max_delta != 0:
			improvement = True
			u,v = max_arc
			if max_operation == 'Addition':
				if debug:
					print('ADDING: ' , max_arc , '\n')
				c_dict[u].append(v)
				p_dict[v].append(u)
				tabu_list[_iter % 5] = (u,v,'Addition')
			elif max_operation == 'Deletion':
				if debug:
					print('DELETING: ' , max_arc , '\n')
				c_dict[u].remove(v)
				p_dict[v].remove(u)
				tabu_list[_iter % 5] = (u,v,'Deletion')
			elif max_operation == 'Reversal':
				if debug:
					print('REVERSING: ' , max_arc, '\n')
					c_dict[u].remove(v)
					p_dict[v].remove(u)
					c_dict[v].append(u)
					p_dict[u].append(v)
					tabu_list[_iter % 5] = (u,v,'Reversal')
		else:
			if debug:
				print('No Improvement on Iter: ' , _iter)

		### TEST FOR MAX ITERATION ###
		_iter += 1
		if _iter > max_iter:
			if debug:
				print('Max Iteration Reached')
			break

	
	bn = BayesNet(c_dict)

	return bn



















