"""
Various metrics for evaluating
score-based search for Bayesian
Networks structure learning.

Generally, score-based structure learning
involves finding the structure which maximizes
some function of the likelihood of the data, given
the structure and parameters of the learned BN.

It is important to take advantage of the decomposability
of the scores. That is, if you add/delete/reverse an edge,
then you only need to re-calculate the score based on that
local change (usually just the child node's CPT) to get the
difference from the original graph. 

Here are a few which are (or can be) implemented:

Information-theoretic scoring functions:
	LL (Log-likelihood) (1912-22)
	MDL/BIC (Minimum description length/Bayesian Information Criterion) (1978)
	AIC (Akaike Information Criterion) (1974)
	NML (Normalized Minimum Likelihood) (2008)
	MIT (Mutual Information Tests) (2006)

References
----------
[1]
http://www.lx.it.pt/~asmc/pub/publications/09-TA/09-c-ta.pdf

"""
from __future__ import division
import numpy as np

from pyBN.utils.independence_tests import mutual_information, entropy


def info_score(bn, nrow, metric='BIC'):
	if metric.upper() == 'LL':
		score = log_likelihood(bn, nrow)
	elif metric.upper() == 'BIC':
		score = BIC(bn, nrow)
	elif metric.upper() == 'AIC':
		score = AIC(bn, nrow)
	else:
		score = BIC(bn, nrow)

	return score
	

##### INFORMATION-THEORETIC SCORING FUNCTIONS #####

def log_likelihood(bn, nrow):
	"""
	Determining log-likelihood of the parameters
	of a Bayesian Network. This is a quite simple
	score/calculation, but it is useful as a straight-forward
	structure learning score.

	Semantically, this can be considered as the evaluation
	of the log-likelihood of the data, given the structure
	and parameters of the BN:
		- log( P( D | Theta_G, G ) )
		where Theta_G are the parameters and G is the structure.

	However, for computational reasons it is best to take
	advantage of the decomposability of the log-likelihood score.
	
	As an example, if you add an edge from A->B, then you simply
	need to calculate LOG(P'(B|A)) - Log(P(B)), and if the value
	is positive then the edge improves the fitness score and should
	therefore be included. 

	Even more, you can expand and manipulate terms to calculate the
	difference between the new graph and the original graph as follows:
		Score(G') - Score(G) = M * I(X,Y),
		where M is the number of data points and I(X,Y) is
		the marginal mutual information calculated using
		the empirical distribution over the data.

	In general, the likelihood score decomposes as follows:
		LL(D | Theta_G, G) = 
			M * Sum over Variables ( I ( X , Parents(X) ) ) - 
			M * Sum over Variables ( H( X ) ),
		where 'I' is mutual information and 'H' is the entropy,
		and M is the number of data points

	Moreover, it is clear to see that H(X) is independent of the choice
	of graph structure (G). Thus, we must only determine the difference
	in the mutual information score of the original graph which had a given
	node and its original parents, and the new graph which has a given node
	and new parents.

	NOTE: This assumes the parameters have already
	been learned for the BN's given structure.

	LL = LL - f(N)*|B|, where f(N) = 0

	Arguments
	---------
	*bn* : a BayesNet object
		Must have both structure and parameters
		instantiated.
	Notes
	-----
	NROW = data.shape[0]
	mi_score = 0
	ent_score = 0
	for rv in bn.nodes():
		cols = tuple([bn.V.index(rv)].extend([bn.V.index(p) for p in bn.parents(rv)]))
		mi_score += mutual_information(data[:,cols])
		ent_score += entropy(data[:,bn.V.index(rv)])
	
	return NROW * (mi_score - ent_score)
	"""
	return np.sum(np.log(nrow * (bn.flat_cpt()+1e-7)))

def MDL(bn, nrow):
	"""
	Minimum Description Length score - it is
	equivalent to BIC
	"""
	return BIC(bn, nrow)

def BIC(bn, nrow):
	"""
	Bayesian Information Criterion.

	BIC = LL - f(N)*|B|, where f(N) = log(N)/2

	"""
	log_score = log_likelihood(bn, nrow)
	penalty = 0.5 * bn.num_params() * np.log(max(bn.num_edges(),1))
	return log_score - penalty

def AIC(bn, nrow):
	"""
	Aikaike Information Criterion

	AIC = LL - f(N)*|B|, where f(N) = 1

	"""
	log_score = log_likelihood(bn, nrow)
	penalty = len(bn.flat_cpt())
	return log_score - penalty

















