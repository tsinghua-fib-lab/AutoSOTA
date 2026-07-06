"""
*******************
Hill Climbing with
Random Restarts for
Structure Learning
*******************

Hill Climbing with random restarts proceeds
as the traditional hill climbing algorithm
does, except that when the hill climbing algorithm
reaches a local maxima or a plateau (or the actual
global maximum), it makes a K random moves - arc
additions/deletions/reversals - and continues the
algorithm. 

The hyper-parameters in this algorithm are the number
of random moves to make per random restart (m), and
the number of random restarts to make (r). It is
intuitive that the algorithm is more likely to get out
of local maxima when m is large and r is large, although
there it depends on how isolated the local maxima is and
also depends on how long you want the algorithm to run.

"""
#from scipy.optimize import *
import numpy as np
#from heapq import *
from copy import copy, deepcopy

from pyBN.classes.bayesnet import BayesNet
from pyBN.learning.parameter.mle import mle_estimator
from pyBN.learning.parameter.bayes import bayes_estimator
from pyBN.learning.structure.score.info_scores import info_score
from pyBN.utils.independence_tests import mutual_information
from pyBN.utils.graph import would_cause_cycle


def hc_rr(data, M=5, R=3, metric='AIC', max_iter=100, debug=False, restriction=None):
	"""
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
	

	_iter = 0
	improvement = True
	_restarts = 0

	while improvement:
		improvement = False
		max_delta = 0

		if debug:
			print('ITERATION: ' , _iter)

		### TEST ARC ADDITIONS ###
		for u in bn.nodes():
			for v in bn.nodes():
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
			#### RESTART WITH RANDOM MOVES ####
			if _restarts < R:
				improvement = True # make another pass of hill climbing
				_iter=0 # reset iterations
				if debug:
					print('Restart - ' , _restarts)
				_restarts+=1
				for _ in range(M):
					# 0 = Addition, 1 = Deletion, 2 = Reversal
					operation = np.random.choice([0,1,2])
					if operation == 0:
						while True:
							u,v = np.random.choice(list(bn.nodes()), size=2, replace=False)
							# IF EDGE DOESN'T EXIST, ADD IT
							if u not in p_dict[v] and u!=v and not would_cause_cycle(c_dict,u,v):
								if debug:
									print('RESTART - ADDING: ', (u,v))
								c_dict[u].append(v)
								p_dict[v].append(u)
								break
					elif operation == 1:
						while True:
							u,v = np.random.choice(list(bn.nodes()), size=2, replace=False)
							# IF EDGE EXISTS, DELETE IT
							if u in p_dict[v]:
								if debug:
									print('RESTART - DELETING: ', (u,v))
								c_dict[u].remove(v)
								p_dict[v].remove(u)
								break
					elif operation == 2:
						while True:
							u,v = np.random.choice(list(bn.nodes()), size=2, replace=False)
							# IF EDGE EXISTS, REVERSE IT
							if u in p_dict[v] and not would_cause_cycle(c_dict,v,u, reverse=True):
								if debug:
									print('RESTART - REVERSING: ', (u,v))
								c_dict[u].remove(v)
								p_dict[v].remove(u)
								c_dict[v].append(u)
								p_dict[u].append(v)
								break

		### TEST FOR MAX ITERATION ###
		_iter += 1
		if _iter > max_iter:
			if debug:
				print('Max Iteration Reached')
			break

	
	bn = BayesNet(c_dict)

	return bn















