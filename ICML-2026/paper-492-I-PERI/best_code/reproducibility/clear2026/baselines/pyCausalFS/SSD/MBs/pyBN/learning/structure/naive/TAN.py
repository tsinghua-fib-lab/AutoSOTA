"""
**************
Tree Augmented 
Naive Bayes
**************

TAN is considered to be quite useful
as a Bayesian network classifier.

"""


def TAN(data, target):
	"""
	Learn a Tree-Augmented Naive Bayes structure
	from data.

	The algorithm from Friedman's paper
	proceeds as follows:

	- Learn a tree structure
		- I will use chow-liu algorithm
	- ADD a class label C to the graph, and an edge 
	from C to each node in the graph.
	"""
	reduced_data = data[:,:-target,:]
	tree, value_dict = chow_liu(reduce_data,edges_only=True)
	tree[target] = [tree.keys()]

	bn = BayesNet(tree, value_dict)
	return vn
