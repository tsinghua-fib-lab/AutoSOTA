"""
******************
Feature Selection
for Classification
******************

Many constraint-based Bayesian network structure 
learning algorithms are actually quite useful for 
general feature selection as a pre-processing task
for absolute any classification algorithm.

The way Bayesian network-based feature selection works
is by essentially learning the Markov Blanket of the
target variable from the data. As has been pointed out
in numerous papers - starting from Daphne Koller's well-known
paper "Toward Optimal Feature Selection" -- identifying the 
Markov Blanket of a target variable gives you the theoretically
optimal selection of feature variables. 

The theoretical guaruntee of Markov Blanket feature selection
stems from the definition of "Markov Blanket" - the set of
nodes which, when conditioned upon, render the target variable 
independent from EVERY other variable in the network. In other
words, if you observe the values of the Markov Blanket for the
target variable, then it is useless to observe the values of 
any variables OUTSIDE the markov blanket - because the target
variable is independent of those variables.

"""

from pyBN.learning.structure.constraint import *


def fs_gambit(data, target):
	""" 
	The Feature Selection Gambit is just a wrapper method
	for calling every possible feature selection method. This
	is a useful method if you want to just run every algorithm
	and choose the best or compare.

	It returns a single dictionary where key = the algorithm 
	method and value = the result from the algorithm 
	(i.e. the set of variables in the Markov Blanket 
	of the target variable).
	
	There are currently five algorithms:
		- iamb
		- fast_iamb
		- lambda_iamb
		- gs (grow-shrink)
		
	Arguments
	---------
	*data* : a nested numpy array

	*target* : an integer
		The column of data that corresponds
		to the target variable.

	Returns
	-------
	*fs_gambit* : a dictionary, where
		key = feature selection algorithm name, and
		value = the result from the given algorithm: the
		set of relevant features for the target (i.e. the
		target variable's Markov Blanket)
	
	Effects
	-------
	None

	Notes
	-----
	None

	"""
	algorithms = ['iamb', 'fast_iamb', 'lambda_iamb', 'gs']
	fs_gambit = {}

	for algo in algorithms:
		fs_gambit[algo] = feature_selection(data=data, target=target, method=algo)

	return fs_gambit

def feature_selection(data, target, method='iamb'):
	"""
	Wrapper for Markov Blanket-based Feature Selection. This
	function provides a unified interface for using the
	various constraint-based structure learning algorithms
	for feature selection instead of full Bayesian Network
	model learning. It simply calls "learn_structure" which
	is already a structure learning wrapper, but also passes
	in "target" denoting that feature selection should take
	place instead of full structure learning.

	Arguments
	---------
	*data* : a nested numpy array

	*method* : a string
		Which feature selection algorithm to run. The
		default is "iamb", and if an unintelligable
		method is given, "iamb" is called.

	"""
	features = learn_structure(data, method=method, feature_selection=target)
	return features





















