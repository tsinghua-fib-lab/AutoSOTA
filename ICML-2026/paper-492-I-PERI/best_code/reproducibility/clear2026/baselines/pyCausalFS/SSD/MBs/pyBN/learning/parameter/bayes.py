"""
*******************
Bayesian Estimation
Parameter Learning
*******************

"""
from __future__ import division

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np


def bayes_estimator(bn, data, equiv_sample=None, prior_dict=None, nodes=None):
	"""
	Bayesian Estimation method of parameter learning.
	This method proceeds by either 1) assuming a uniform prior
	over the parameters based on the Dirichlet distribution
	with an equivalent sample size = *sample_size*, or
	2) assuming a prior as specified by the user with the 
	*prior_dict* argument. The prior distribution is then
	updated from observations in the data based on the
	Multinomial distribution - for which the Dirichlet
	is a "conjugate prior."

	Note that the Bayesian and MLE estimators essentially converge
	to the same set of values as the size of the dataset increases.

	Also note that, unlike the structure learning algorithms, the
	parameter learning functions REQUIRE a passed-in BayesNet object
	because there MUST be some pre-determined structure for which
	we can actually learn the parameters. You can't learn parameters
	without structure - so structure must always be there first!

	Finally, note that this function can be used to calculate only
	ONE conditional probability table in a BayesNet object by
	passing in a subset of random variables with the "nodes"
	argument - this is mostly used for score-based structure learning,
	where a single cpt needs to be quickly recalculate after the
	addition/deletion/reversal of an arc.

	Arguments
	---------
	*bn* : a BayesNet object

	*data* : a nested numpy array
		Data from which to learn parameters

	*equiv_sample* : an integer
		The "equivalent sample size" (see function summary)

	*prior_dict* : a dictionary, where key = random variable
		and for each key the value is another dictionary where
		key = an instantiation for the random variable and the
		value is its FREQUENCY (an integer value, NOT its relative
		proportion/probability).
	
	*nodes* : a list of strings
		Which nodes to learn the parameters for - if None,
		all nodes will be used as expected.
	
	Returns
	-------
	None

	Effects
	-------
	- modifies/sets bn.data to the learned parameters

	Notes
	-----

	"""
	if equiv_sample is None:
		equiv_sample = len(data)

	if nodes is None:
		nodes = list(bn.nodes())

	for i, n in enumerate(nodes):
		bn.F[n]['values'] = list(np.unique(data[:,i]))

	obs_dict = dict([(rv,[]) for rv in nodes])
	# set empty conditional probability table for each RV
	for rv in nodes:
		# get number of values in the CPT = product of scope vars' cardinalities
		p_idx = int(np.prod([bn.card(p) for p in bn.parents(rv)])*bn.card(rv))
		bn.F[rv]['cpt'] = [equiv_sample/p_idx]*p_idx
	
	# loop through each row of data
	for row in data:
		# store the observation of each variable in the row
		obs_dict = dict([(rv,row[rv]) for rv in nodes])
		# loop through each RV and increment its observed parent-self value
		for rv in nodes:
			rv_dict= { n: obs_dict[n] for n in obs_dict if n in bn.scope(rv) }
			offset = bn.cpt_indices(target=rv,val_dict=rv_dict)[0]
			bn.F[rv]['cpt'][offset]+=1

	
	for rv in nodes:
		cpt = bn.cpt(rv)
		for i in range(0,len(bn.cpt(rv)),bn.card(rv)):
			temp_sum = float(np.sum(cpt[i:(i+bn.card(rv))]))
			for j in range(bn.card(rv)):
				cpt[i+j] /= (temp_sum)
				cpt[i+j] = round(cpt[i+j],5)







