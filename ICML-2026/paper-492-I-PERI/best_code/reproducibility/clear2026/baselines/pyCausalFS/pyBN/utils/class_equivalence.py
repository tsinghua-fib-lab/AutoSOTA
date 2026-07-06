"""
****************
Equivalence Code
****************

This code is for testing whether or not
two Bayesian networks belong to the same
equivalence class - i.e. they have the same
edges when viewed as undirected graphs.

Also, this code is for actually generating
equivalent Bayesian networks from a given BN.

"""

def are_class_equivalent(x,y):
	"""
	Check whether two Bayesian networks belong
	to the same equivalence class.
	"""
	are_equivalent = True

	if set(list(x.nodes())) != set(list(y.nodes())):
		are_equivalent = False
	else:
		for rv in x.nodes():
			rv_x_neighbors = set(x.parents(rv)) + set(y.children(rv))
			rv_y_neighbors = set(y.parents(rv)) + set(y.children(rv))
			if rv_x_neighbors != rv_y_neighbors:
				are_equivalent =  False
				break
	return are_equivalent