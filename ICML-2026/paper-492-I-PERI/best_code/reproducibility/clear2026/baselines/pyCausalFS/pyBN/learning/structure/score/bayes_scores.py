"""
Various Bayesian scoring metrics for evaluating
the fitness of a BN structure during score-based
structure learning.

Bayesian scoring functions:
	BD (Bayesian Dirichlet) (1995)
	BDe ("'e'" for likelihood-equivalence) (1995)
	BDeu ("'u'" for uniform joint distribution) (1991)
	K2 (1992)

References
----------
[1] Daly, et al. Learning Bayesian Network Equivalence Classes 
with Ant Colony Optimization.

"""
from __future__ import division

import numpy as np
from scipy.special import gamma, gammaln
from pyBN.learning.parameter.mle import mle_estimator, mle_fast
from pyBN.classes.empiricaldistribution import EmpiricalDistribution


def BDe(bn, data, ess=1, ed=None): 
	"""
	Unique Bayesian score with the property that I-equivalent
	networks have the same score.

	As Data Rows -> infinity, BDe score converges to the BIC score.

	Arguments
	---------
	*bn* : a BayesNet object
		Needed to get the parent relationships, etc.
	
	*data* : a numpy ndarray
		Needed to learn the empirical distribuion
	
	*ess* : an integer
		Equivalent sample size

	*ed* : an EmpiricalDistribution object
		Used to cache multiple lookups in structure learning.

	Notes
	-----
	*a_ijk* : a vector
		The number of times where x_i=k | parents(x_i)=j -> i.e. the mle counts

	*a_ij* : a vector summed over k's in a_ijk

	*n_ijk* : a vector prior (sample size or calculation)
		"ess" for BDe metric

	*n_ij* : a vector prior summed over k's in n_ijk
	
	"""
	counts_dict = mle_fast(bn, data, counts=True, np=True)
	a_ijk = []
	bdeu = 1
	for rv, value in counts_dict.items():
		nijk = value['cpt']
		nijk_prime = ess
		k2 *= gamma(nijk+nijk_prime)/gamma(nijk_prime)
		nij_prime = nijk_prime*(len(cpt)/bn.card(rv))
		nij = np.mean(nijk.reshape(-1, bn.card(rv)), axis=1) # sum along parents
		k2 *= gamma(nij_prime) / gamma(nij+nij_prime)

	return bdeu


def BDeu(bn, data, ess=1, ed=None):
	"""
	Unique Bayesian score with the property that I-equivalent
	networks have the same score.

	As Data Rows -> infinity, BDe score converges to the BIC score.

	Nijk_prime = ess/len(bn.cpt(rv))

	Arguments
	---------
	*bn* : a BayesNet object
		Needed to get the parent relationships, etc.
	
	*data* : a numpy ndarray
		Needed to learn the empirical distribuion
	
	*ess* : an integer
		Equivalent sample size

	*ed* : an EmpiricalDistribution object
		Used to cache multiple lookups in structure learning.

	Notes
	-----
	*a_ijk* : a vector
		The number of times where x_i=k | parents(x_i)=j -> i.e. the mle counts

	*a_ij* : a vector summed over k's in a_ijk

	*n_ijk* : a vector prior (sample size or calculation)
		ess/(card(x_i)*len(cpt(x_i)/card(x_i))) for x_i for BDe metric

	*n_ij* : a vector prior summed over k's in n_ijk
	
	"""
	counts_dict = mle_fast(bn, data, counts=True, np=True)
	a_ijk = []
	bdeu = 1
	for rv, value in counts_dict.items():
		nijk = value['cpt']
		nijk_prime = ess / len(nijk)
		k2 *= gamma(nijk+nijk_prime)/gamma(nijk_prime)
		nij_prime = nijk_prime*(len(cpt)/bn.card(rv))
		nij = np.mean(nijk.reshape(-1, bn.card(rv)), axis=1) # sum along parents
		k2 *= gamma(nij_prime) / gamma(nij+nij_prime)

	return bdeu

def K2(bn, data, ed=None):
	"""
	K2 is bayesian posterior probability of structure given the data,
	where N'ijk = 1.
	"""
	counts_dict = mle_fast(bn, data, counts=True, np=True)
	a_ijk = []
	k2 = 1
	for rv, value in counts_dict.items():
		nijk = value['cpt']
		nijk_prime = 1
		k2 *= gamma(nijk+nijk_prime)/gamma(nijk_prime)
		nij_prime = nijk_prime*(len(cpt)/bn.card(rv))
		nij = np.mean(nijk.reshape(-1, bn.card(rv)), axis=1) # sum along parents
		k2 *= gamma(nij_prime) / gamma(nij+nij_prime)

	return k2









