"""
*************
Chow-Liu Tree
*************

Calculate the KL divergence (i.e. run
mi_test) between every pair of nodes,
then select the maximum spanning tree from that
connected graph. This is the Chow-Liu tree.

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.utils.independence_tests import mi_test
from pyBN.classes.bayesnet import BayesNet
import operator
import numpy as np


def chow_liu(data,edges_only=False):
	"""
	Perform Chow-Liu structure learning algorithm
	over an entire dataset, and return the BN-tree.


	Arguments
	---------
	*data* : a nested numpy array
		The data from which we will learn. It should be
		the entire dataset.

	Returns
	-------
	*bn* : a BayesNet object
		The structure-learned BN.

	Effects
	-------
	None

	Notes
	-----

	"""
	value_dict = dict(zip(range(data.shape[1]),
		[list(np.unique(col)) for col in data.T]))

	n_rv = data.shape[1]

	edge_list = [(i,j,mi_test(data[:,(i,j)],chi2_test=False)) \
					for i in range(n_rv) for j in range(i+1,n_rv)]
	
	edge_list.sort(key=operator.itemgetter(2), reverse=True) # sort by weight
	vertex_cache = {edge_list[0][0]} # start with first vertex..
	mst = dict((rv, []) for rv in range(n_rv))

	for i,j,w in edge_list:
		if i in vertex_cache and j not in vertex_cache:
			mst[i].append(j)
			vertex_cache.add(j)
		elif i not in vertex_cache and j in vertex_cache:
			mst[j].append(i)
			vertex_cache.add(i)
	
	if edges_only == True:
		return mst, value_dict

	bn=BayesNet(mst,value_dict)
	return bn












