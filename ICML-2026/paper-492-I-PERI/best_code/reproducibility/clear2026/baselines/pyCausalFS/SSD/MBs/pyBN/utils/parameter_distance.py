"""
****************
BN Parameter 
Distance Metrics
****************

Code for computing the Parametric distance 
between 2+ Bayesian Networks using various
metrics. Parametric distance only involves a
comparison of the conditional probability values.
For structural distance metrics, see "structure_distance.py"

Since a Bayesian network is simply joint probability distribution,
any distibution distance metric can be used.

Metrics
-------
*euclidean*
*manhattan*
*minkowski*
*kl_divergence*
*js_divergence*
*hellinger*

"""
from __future__ import division

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np
#from pyBN.classes.bayesnet import BayesNet



def euclidean(x,y):
	"""
	Euclidean distance is a well-known distance metric.

	It is defined as follows:
		Sum of ( sqrt of (x - y)^2 )
	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	distance = np.sum( np.sqrt( ( x.flat_cpt() - y.flat_cpt() )**2 ) )
	return distance

def manhattan(x,y):
	"""
	Manhattan distance is a well-known distance metric.

	It is defined as follows:
		Sum of ( |x-y| )
	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	distance = np.sum( np.abs( x.flat_cpt() - y.flat_cpt() ) )
	return distance

def minkowski(x,y,p=1.5):
	"""
	Minkowski distance is a metric where the user
	can specify the exponent to which the Parameter
	difference values is raised.

	It is defined as follows:
		(Sum of (x-y)^p)^(1/p)

	Note that when p = 1, this is equivalent to Manhattan Distance.
	Note that when p = 2, this is equivalent to Euclidean Distance.

	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	distance = np.sum( ( x.flat_cpt() - y.flat_cpt() )**2 )**( 1/p )
	return distance

def kl_divergence(x,y):
	"""
	KL-Divergence is a well-known metric, although
	it is not technically a valid distance measure
	because it is NOT symmetric - i.e. KL(X,Y) != KL(Y,X)
	
	It is defined as follows:
		Sum of x * log(x/y)
	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	distance = np.sum( x.flat_cpt() * np.log( x.flat_cpt() / y.flat_cpt() ) )
	return distance
	
def js_divergence(x,y):
	"""
	Jensen-Shannon Divergence is similar to KL-Divergence,
	but it is symmetric and therefore is a valid distance metric.

	It is defined as follows:
		JSD = 1/2 * KL(X,Z) + 1/2 * KL(Y,Z),
			where Z = 1/2 * (X + Y)
	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	z = 0.5 * ( x.flat_cpt() + y.flat_cpt() )
	distance = 0.5 * kl_divergence(x,z) + 0.5 * kl_divergence(y,z)
	return distance


def hellinger(x,y):
	"""
	Hellinger is a well-known distance metric.
	
	It is defined as follows:
	1/Sqrt(2) * Sqrt(Sum of ((sqrt(x) - sqrt(y))^2))
	"""
	assert (isinstance(x, BayesNet) and isinstance(y, BayesNet)), 'Must pass in BayesNet objects.'
	assert (x==y), 'Passed-in BayesNet objects are not structurally equal.'

	distance = ( 1 / np.sqrt( 2 ) ) * np.sqrt( np.sum( ( np.sqrt( x.flat_cpt() ) - np.sqrt( y.flat_cpt() ) )**2) )
	return distance


















