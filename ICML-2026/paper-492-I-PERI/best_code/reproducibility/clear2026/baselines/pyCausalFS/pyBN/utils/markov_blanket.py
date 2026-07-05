"""
**************
Markov Blanket
**************

References
----------
[1] Koller and Sahami, "Toward Optimal Feature Selection."
[2] Yaramakala and Margaritis, "Speculative Markov Blanket Discovery 
for Optimal Feature Selection"
[3] Pellet and Elisseff, "Using Markov Blankets for Causal Structure Learning"

"""
from __future__ import division
from copy import copy
import itertools
from pyBN.utils.independence_tests import are_independent, mi_test


__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""


def markov_blanket(bn):
	"""
	Return the Markov Blanket dictionary from a fully
	(structurally) instantiated BayesNet object.

	The markov blanket for a given node is just
	the node's parents, children, and 
	its children's parents (i.e. spouses)

	Arguments
	---------
	*bn* : BayesNet object

	Returns
	-------
	*mb* :  a dictionary where each key 
		is a node and the value is a list of
		the key-node's markov blanket
	"""
	mb = dict([(rv,bn.parents(rv)+bn.children(rv)) for rv in bn.nodes()])

	for rv in bn.V:
		for child in bn.children(rv):
			for c_parent in bn.parents(child):
				if c_parent != rv:
					mb[rv].append(c_parent) # add spouse
	return mb

def resolve_markov_blanket(Mb, data,alpha=0.05):
	"""
	Resolving the Markov blanket is the process
	by which a PDAG is constructed from the collection
	of Markov Blankets for each node. Since an
	undirected graph is returned, the edges still need to 
	be oriented by calling some version of the 
	"orient_edges" function in "pyBN.structure_learn.orient_edges" 
	module.

	This algorithm is adapted from Margaritis, but also see [3]
	for good pseudocode.

	Arguments
	---------
	*Mb* : a dictionary, where
		key = rv and value = list of vars in rv's markov blanket

	*data* : a nested numpy array
		The dataset used to learn the Mb

	Returns
	-------
	*edge_dict* : a dictionary, where
		key = rv and value = list of rv's children

	Effects
	-------
	None

	Notes
	-----
	"""
	n_rv = data.shape[1]
	edge_dict = dict([(rv,[]) for rv in range(n_rv)])
	for X in range(n_rv):
		for Y in Mb[X]:
			# X and Y are direct neighbors if X and Y are dependent
			# given S for all S in T, where T is the smaller of
			# B(X)-{Y} and B(Y)-{X}
			if len(Mb[X]) < len(Mb[Y]):
				T = copy(Mb[X]) # shallow copy is sufficient
				if Y in T:
					T.remove(Y)
			else:
				T = copy(Mb[Y]) # shallow copy is sufficient
				if X in T:
					T.remove(X)

			# X and Y must be dependent conditioned upon
			# EVERY POSSIBLE COMBINATION of T
			direct_neighbors=True
			for i in range(len(T)):
				for S in itertools.combinations(T,i):
					cols = (X,Y) + tuple(S)
					pval = mi_test(data[:,cols])
					if pval > alpha:
						direct_neighbors=False
			if direct_neighbors:
				if Y not in edge_dict[X] and X not in edge_dict[Y]:
					edge_dict[X].append(Y)
				if X not in edge_dict[Y]:
					edge_dict[Y].append(X)
	return edge_dict

def mb_fitness(data, Mb, target=None):
	"""
	Evaluate the fitness of a Markov Blanket dictionary
	learned from a given data set based on the distance metric
	provided in [1] and [2].

	From [2]:
		A distance measure that indicates the "fitness"
		of the discovered blanket... to be the average, over all attributes
		X outside the blanket, of the expected KL-divergence between
		Pr(T | B(T)) and Pr(T | B(T) u {X}). We can expect this 
		measure to be close to zero when B(T) is an approximate
		blanket. -- 
		My Note: T is the target variable, and if the KL-divergence
		between the two distributions above is zero, then it means that
		{X} provides no new information about T and can thus be excluded
		from Mb(T) -- this is the exact definition of conditional independence.

	Notes
	-----
	- Find Pr(T|B(T)) ..
	- For each variable X outside of the B(T), calculate 
		D( Pr(T|B(T)), Pr(T|B(T)u{X}) )
	- Take the average (closer to Zero is better)

	^^^ This is basically calculating where T is independent of X given B(T)..
		i.e. Sum over all X not in B(T) of mi_test(data[:,(T,X,B(T))]) / |X|
	"""
	if target is None:
		nodes = set(Mb.keys())
	else:
		try:
			nodes = set(target)
		except TypeError:
			nodes = {target}

	fitness_dict = dict([(rv, 0) for rv in nodes])
	for T in nodes:
		non_blanket = nodes - set(Mb[T]) - {T}
		for X in non_blanket:
			pval = mi_test(data[:,(T,X)+tuple(Mb[T])])
			fitness_dict[T] += 1/pval
	return fitness_dict


	







