"""
**********************
Greedy Hill-Climbing
for Structure Learning
**********************

Code for Searching through the space of
possible Bayesian Network structures.

Various optimization procedures are employed,
from greedy search to simulated annealing, and 
so on - mostly using scipy.optimize.

Local search - possible moves:
- Add edge
- Delete edge
- Invert edge

Strategies to improve Greedy Hill-Climbing:
- Random Restarts
	- when we get stuck, take some number of
	random steps and then start climbing again.
- Tabu List
	- keep a list of the K steps most recently taken,
	and say that the search cannt reverse (undo) any
	of these steps.
"""

#from scipy.optimize import *
import numpy as np
#from heapq import *

from SSD.MBs.pyBN.classes.bayesnet import BayesNet
from SSD.MBs.pyBN.learning.parameter import mle_estimator
from SSD.MBs.pyBN.learning.structure.score.info_scores import info_score
from SSD.MBs.pyBN.utils import mutual_information
from SSD.MBs.pyBN.utils import would_cause_cycle


def hc(data, metric='AIC', max_iter=100, debug=False, restriction=None):
	"""
	Greedy Hill Climbing search proceeds by choosing the move
	which maximizes the increase in fitness of the
	network at the current step. It continues until
	it reaches a point where there does not exist any
	feasible single move that increases the network fitness.

	It is called "greedy" because it simply does what is
	best at the current iteration only, and thus does not
	look ahead to what may be better later on in the search.

	For computational saving, a Priority Queue (python's heapq)
	can be used	to maintain the best operators and reduce the
	complexity of picking the best operator from O(n^2) to O(nlogn).
	This works by maintaining the heapq of operators sorted by their
	delta score, and each time a move is made, we only have to recompute
	the O(n) delta-scores which were affected by the move. The rest of
	the operator delta-scores are not affected.

	For additional computational efficiency, we can cache the
	sufficient statistics for various families of distributions - 
	therefore, computing the mutual information for a given family
	only needs to happen once.

	The possible moves are the following:
		- add edge
		- delete edge
		- invert edge

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
	c_dict = dict([(n, []) for n in names])
	p_dict = dict([(n, []) for n in names])
	
	# COMPUTE INITIAL LIKELIHOOD SCORE	
	value_dict = dict([(n, np.unique(data[:, i])) for i,n in enumerate(names)])
	bn = BayesNet(c_dict)
	mle_estimator(bn, data)
	max_score = info_score(bn, nrow, metric)

	# CREATE EMPIRICAL DISTRIBUTION OBJECT FOR CACHING
	#ED = EmpiricalDistribution(data,names)

	

	_iter = 0
	improvement = True

	while improvement:
		improvement = False
		max_delta = 0

		if debug:
			print('ITERATION: ', _iter)

		### TEST ARC ADDITIONS ###
		for u in bn.nodes():
			for v in bn.nodes():
				if v not in c_dict[u] and u != v and not would_cause_cycle(c_dict, u, v):
					# FOR MMHC ALGORITHM -> Edge Restrictions
					if restriction is None or (u, v) in restriction:
						# SCORE FOR 'V' -> gaining a parent
						old_cols = (v,) + tuple(p_dict[v])  # without 'u' as parent
						mi_old = mutual_information(data[:, old_cols])
						new_cols = old_cols + (u,) # with'u' as parent
						mi_new = mutual_information(data[:, new_cols])
						delta_score = nrow * (mi_old - mi_new)

						if delta_score > max_delta:
							#if debug:
							#	print('Improved Arc Addition: ' , (u,v))
							#	print('Delta Score: ' , delta_score)
							max_delta = delta_score
							max_operation = 'Addition'
							max_arc = (u,v)

		### TEST ARC DELETIONS ###
		for u in bn.nodes():
			for v in bn.nodes():
				if v in c_dict[u]:
					# SCORE FOR 'V' -> losing a parent
					old_cols = (v,) + tuple(p_dict[v]) # with 'u' as parent
					mi_old = mutual_information(data[:,old_cols])
					new_cols = tuple([i for i in old_cols if i != u]) # without 'u' as parent
					mi_new = mutual_information(data[:,new_cols])
					delta_score = nrow * (mi_old - mi_new)

					if delta_score > max_delta:
						#if debug:
						#	print('Improved Arc Deletion: ' , (u,v))
						#	print('Delta Score: ' , delta_score)
						max_delta = delta_score
						max_operation = 'Deletion'
						max_arc = (u,v)

		### TEST ARC REVERSALS ###
		for u in bn.nodes():
			for v in bn.nodes():
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
						#if debug:
						#	print('Improved Arc Reversal: ' , (u,v))
						#	print('Delta Score: ' , delta_score)
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
			elif max_operation == 'Deletion':
				if debug:
					print('DELETING: ' , max_arc , '\n')
				c_dict[u].remove(v)
				p_dict[v].remove(u)
			elif max_operation == 'Reversal':
				if debug:
					print('REVERSING: ' , max_arc, '\n')
					c_dict[u].remove(v)
					p_dict[v].remove(u)
					c_dict[v].append(u)
					p_dict[u].append(v)
		else:
			if debug:
				print('No Improvement on Iter: ' , _iter)

		### TEST FOR MAX ITERATION ###
		_iter += 1
		if _iter > max_iter:
			if debug:
				print('Max Iteration Reached')
			break


	# bn = BayesNet(c_dict)
	# print("bn is: " + str(bn.E))




	return c_dict


# import pandas as pd
#
# df = pd.read_csv("C:/pythonProject/BN_PC_algorithm/data/child_s5000_v2.csv")
# print(df)
# data = np.array(df,dtype=np.int_)
# data1 = data[:,0:10]
# print("the file read")
# print("data is: " + str(data1))
# MB = hc(data, metric ='BIC',debug=True)
# print("MBs is: " + str(MB))














