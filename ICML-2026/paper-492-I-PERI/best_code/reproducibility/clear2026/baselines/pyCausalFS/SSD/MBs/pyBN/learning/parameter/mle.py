"""
*****************************
Maximum Likelihood Estimation
Parameter Learning
*****************************

"""
from __future__ import division

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def mle_fast(bn, data, nodes=None, counts=False, np=False):
	"""
	Maximum Likelihood estimation that is about 100 times as
	fast as the original mle_estimator function - but returns
	the same result
	"""
	def merge_cols(data, cols):
		if len(cols) == 1:
			return data[cols[0]]
		elif len(cols) > 1:
			data = data[cols].astype('str')
			ncols = len(cols)
			for i in range(len(data)):
				data.ix[i,0] = ''.join(data.ix[i,0:ncols])
			data = data.astype('int')
			return data.ix[:,0]

	if nodes is None:
		nodes = list(bn.nodes())
	else:
		if not isinstance(nodes, list):
			nodes = list(nodes)

	F = dict([(rv, {}) for rv in nodes])
	for i, n in enumerate(nodes):
		F[n]['values'] = list(np.unique(data.ix[:,i]))
		bn.F[n]['values'] = list(np.unique(data.ix[:,i]))

	for rv in nodes:
		parents = bn.parents(rv)
		if len(parents)==0:
			if np:
				F[rv]['cpt'] = np.histogram(data.ix[:,rv], bins=bn.card(rv))[0]
			else:
				F[rv]['cpt'] = list(np.histogram(data.ix[:,rv], bins=bn.card(rv))[0])
		else:
			if np:
				F[rv]['cpt'] = np.histogram2d(merge_cols(data,parents),data.ix[:,rv], 
				bins=[np.prod([bn.card(p) for p in parents]),bn.card(rv)])[0].flatten()
			else:
				F[rv]['cpt'] = list(np.histogram2d(merge_cols(data,parents),data.ix[:,rv], 
					bins=[np.prod([bn.card(p) for p in parents]),bn.card(rv)])[0].flatten())
	if counts:
		return F
	else:
		for rv in nodes:
			F[rv]['parents'] = [var for var in nodes if rv in bn.E[var]]
			for i in range(0,len(F[rv]['cpt']),bn.card(rv)):
				temp_sum = float(np.sum(F[rv]['cpt'][i:(i+bn.card(rv))]))
				for j in range(bn.card(rv)):
					F[rv]['cpt'][i+j] /= (temp_sum+1e-7)
					F[rv]['cpt'][i+j] = round(F[rv]['cpt'][i+j],5)
		bn.F = F




def mle_estimator(bn, data, nodes=None, counts=False):
	"""
	Maximum Likelihood Estimation is a frequentist
	method for parameter learning, where there is NO
	prior distribution. Instead, the frequencies/counts
	for each parameter start at 0 and are simply incremented
	as the relevant parent-child values are observed in the
	data. 

	This can be a risky method for small datasets, because if a 
	certain parent-child instantiation is never observed in the
	data, then its probability parameter will be ZERO (even if you
	know it should at least have a very small probability). 

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
		The associated network structure for which
		the parameters will be learned

	*data* : a nested numpy array

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
	- Currently doesn't return correct solution

	data attributes:
		"numoutcomes" : an integer
	    "vals" : a list
	    "parents" : a list or None
	    "children": a list or None
	    "cprob" : a nested python list

	- Do not want to alter bn.data directly!

	"""
	if nodes is None:
		nodes = list(bn.nodes())
	else:
		if not isinstance(nodes, list):
			nodes = list(nodes)

	F = dict([(rv, {}) for rv in nodes])
	for i, n in enumerate(nodes):
		F[n]['values'] = list(np.unique(data[:,i]))
		bn.F[n]['values'] = list(np.unique(data[:,i]))

	obs_dict = dict([(rv,[]) for rv in nodes])
	# set empty conditional probability table for each RV
	for rv in nodes:
		# get number of values in the CPT = product of scope vars' cardinalities
		p_idx = int(np.prod([bn.card(p) for p in bn.parents(rv)])*bn.card(rv))
		F[rv]['cpt'] = [0]*p_idx
		bn.F[rv]['cpt'] = [0]*p_idx
	
	# loop through each row of data
	for row in data:
		# store the observation of each variable in the row
		for rv in nodes:
			obs_dict[rv] = row[rv]
		
		#obs_dict = dict([(rv,row[rv]) for rv in nodes])
		# loop through each RV and increment its observed parent-self value
		for rv in nodes:
			rv_dict= { n: obs_dict[n] for n in obs_dict if n in bn.scope(rv) }
			offset = bn.cpt_indices(target=rv,val_dict=rv_dict)[0]
			F[rv]['cpt'][offset]+=1

	if counts:
		return F
	else:
		for rv in nodes:
			F[rv]['parents'] = [var for var in nodes if rv in bn.E[var]]
			for i in range(0,len(F[rv]['cpt']),bn.card(rv)):
				temp_sum = float(np.sum(F[rv]['cpt'][i:(i+bn.card(rv))]))
				for j in range(bn.card(rv)):
					F[rv]['cpt'][i+j] /= (temp_sum+1e-7)
					F[rv]['cpt'][i+j] = round(F[rv]['cpt'][i+j],5)
		bn.F = F





