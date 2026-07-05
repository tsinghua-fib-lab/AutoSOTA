"""
***********
Naive Bayes
***********

Learn both the structure AND parameters
of a Naive Bayes Model from data

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
from pyBN.classes.bayesnet import BayesNet
from pyBN.learning.parameter.mle import mle_estimator
from pyBN.learning.parameter.bayes import bayes_estimator

def naive_bayes(data, target, estimator='mle'):
	"""
	Learn naive bayes model from data.

	The Naive Bayes model is a Tree-based
	model where all random variables have
	the same parent (the "target" variable).
	From a probabilistic standpoint, the implication
	of this model is that all random variables 
	(i.e. features) are assumed to be
	conditionally independent of any other random variable,
	conditioned upon the single parent (target) variable.

	It turns out that this model performs quite well
	as a classifier, and can be used as such. Moreover,
	this model is quite fast and simple to learn/create
	from a computational standpoint.

	Note that this function not only learns the structure,
	but ALSO learns the parameters.

	Arguments
	---------
	*data* : a nested numpy array

	*target* : an integer
		The target variable column in *data*

	Returns
	-------
	*bn* : a BayesNet object,
		with the structure instantiated.

	Effects
	-------
	None

	Notes
	-----

	"""	
	value_dict = dict(zip(range(data.shape[1]),
		[list(np.unique(col)) for col in data.T]))

	edge_dict = {target:[v for v in value_dict if v!=target]}
	edge_dict.update(dict([(rv,[]) for rv in value_dict if rv!=target]))

	bn = BayesNet(edge_dict,value_dict)
	if estimator == 'bayes':
		bayes_estimator(bn,data)
	else:
		mle_estimator(bn,data)
	return bn

















