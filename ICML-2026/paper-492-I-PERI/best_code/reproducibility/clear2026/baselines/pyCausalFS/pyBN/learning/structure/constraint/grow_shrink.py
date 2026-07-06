"""
********************
GrowShrink Algorithm
********************

"Our approach constructs Bayesian networks by first identifying each node's
Markov blankets, then connecting nodes in a maximally consistent way. 
In contrast to the majority of work, which typically uses hill-climbing approaches 
that may produce dense and causally incorrect nets, our approach yields much more 
compact causal networks by heeding independenciesin the data. Compact causal networks
facilitate fast inference and are also easier to understand. We prove that under mild 
assumptions, our approach requires time polynomial in the size of the data and the number 
of nodes. A randomized variant, also presented here, yields comparable results at
much higher speeds" [1].

This algorithm relies on the "Markov Blanket", which is for a given random variable
the set of other random variables which render the given RV conditionally independent
from the rest of the network. That is, if you observe any of the variables in a given
RV's markov blanket, then observing the values of any OTHER variables in the network
will not change your beliefs about the given random variable.

The Markov blanket of a node X is easily identifiable from the graph: 
it consists of X's parents, X's children, and the parents of all of X's children.

The runtime of this algorithm is O(|V|) [1].

References
----------
[1] Margaritis and Thrun: "Bayesian Network Induction via Local Neighborhoods", 
NIPS 2000.

[2] https://www.cs.cmu.edu/~dmarg/Papers/PhD-Thesis-Margaritis.pdf

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.utils.independence_tests import mi_test
from pyBN.utils.orient_edges import orient_edges_MB
from pyBN.utils.markov_blanket import resolve_markov_blanket
from pyBN.classes.bayesnet import BayesNet
from pyBN.utils.data import replace_strings

from copy import copy
import numpy as np
import itertools

def gs(data, alpha=0.05, feature_selection=None, debug=False):
	"""
	Perform growshink algorithm over dataset to learn
	Bayesian network structure.

	This algorithm is clearly a good candidate for
	numba JIT compilation...

	STEPS
	-----
	1. Compute Markov Blanket
	2. Compute Graph Structure
	3. Orient Edges
	4. Remove Cycles
	5. Reverse Edges
	6. Propagate Directions

	Arguments
	---------
	*data* : a nested numpy array
		Data from which you wish to learn structure

	*alpha* : a float
		Type I error rate for independence test

	Returns
	-------
	*bn* : a BayesNet object

	Effects
	-------
	None

	Notes
	-----

	Speed Test:
		*** 5 variables, 624 observations ***
		- 63.7 ms

	"""
	n_rv = data.shape[1]
	data, value_dict = replace_strings(data, return_values=True)
	

	if feature_selection is None:
		_T = range(n_rv)
	else:
		assert (not isinstance(feature_selection, list)), 'feature_selection must be only one value'
		_T = [feature_selection]

	# STEP 1 : COMPUTE MARKOV BLANKETS
	Mb = dict([(rv,[]) for rv in range(n_rv)])

	for X in _T:
		S = []

		grow_condition = True
		while grow_condition:

			grow_condition=False
			for Y in range(n_rv):
				if X!=Y and Y not in S:
					# if there exists some Y such that Y is dependent on X given S,
					# add Y to S
					cols = (X,Y) + tuple(S)
					pval = mi_test(data[:,cols])
					if pval < alpha: # dependent
						grow_condition=True # dependent -> continue searching
						S.append(Y)
		
		shrink_condition = True
		while shrink_condition:

			TEMP_S = []
			shrink_condition=False
			for Y in S:
				s_copy = copy(S)
				s_copy.remove(Y) # condition on S-{Y}
				# if X independent of Y given S-{Y}, leave Y out
				# if X dependent of Y given S-{Y}, keep it in
				cols = (X,Y) + tuple(s_copy)
				pval = mi_test(data[:,cols])
				if pval < alpha: # dependent
					TEMP_S.append(Y)
				else: # independent -> condition searching
					shrink_condition=True
		
		Mb[X] = TEMP_S
		if debug:
			print('Markov Blanket for %s : %s' % (X, str(TEMP_S)))
	
	if feature_selection is None:
		# STEP 2: COMPUTE GRAPH STRUCTURE
		# i.e. Resolve Markov Blanket
		edge_dict = resolve_markov_blanket(Mb,data)
		if debug:
			print('Unoriented edge dict:\n %s' % str(edge_dict))
		
		# STEP 3: ORIENT EDGES
		oriented_edge_dict = orient_edges_MB(edge_dict,Mb,data,alpha)
		if debug:
			print('Oriented edge dict:\n %s' % str(oriented_edge_dict))
		

		# CREATE BAYESNET OBJECT
		bn=BayesNet(oriented_edge_dict,value_dict)
		
		return bn
	else:
		return Mb[_T]














