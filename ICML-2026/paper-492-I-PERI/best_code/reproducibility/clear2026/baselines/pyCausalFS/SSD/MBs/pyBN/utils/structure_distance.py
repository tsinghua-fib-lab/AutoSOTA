"""
****************
BN Structure 
Distance Metrics
****************

Code for computing the structural Distance
between 2+ Bayesian networks. Note, this
code compares only structure - i.e. the
conditional (in)dependencies. For a comparison
of conditional probability values of 2+ BNs,
see "parameter_distance.py"

For computing distance metrics that include
both structure AND parameters, see "hybrid_distance.py"

Metrics
-------
1. Missing Edges: 
	counts edges that are present in the original structure but
	are missing in the learned structure. (lower is better) [1]

2. Extra Edges: 
	counts edges that are found in the learned structure but are
	not present in the original structure. (lower is better) [1]

3. Incorrect Edge Direction: 
	counts directed edges in the learned structure
	that are oriented incorrectly. (lower is better) [1]

4. Hamming Distance: 
	Describes the number of changes that have to be made
	to a network for it to turn into the one that it is being compared with. It is the
	sum of measures 1, 2, and 3. Versions of the Hamming distance metric have
	been proposed by Acid and de Campos (2003), Tsamardinos et al. (2006), and
	Perrier et al. (2008). (lower is better) [1]


References
----------
[1] Martijn de Jongh and Marek J. Druzdzel
	"A Comparison of Structural Distance Measures
	for Causal Bayesian Network Models"


"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

import numpy as np

def missing_edges(x,y):
	"""
	Counts edges that are present in the original structure (x) but
	are missing in the learned structure (y). (lower is better)
	"""
	missing = 0
	for rv in x.nodes():
		if not y.has_node(rv):
			missing += x.degree(rv)
		else:
			for child in x.children(rv):
				if child not in y.children(rv):
					missing += 1
	return missing

def extra_edges(x,y):
	"""
	Counts edges that are found in the learned structure but are
	not present in the original structure. (lower is better)
	"""
	extra = 0
	for rv in y.nodes():
		if not x.has_node(rv):
			extra += y.degree(rv)
		else:
			for child in y.children(rv):
				if child not in x.children(rv):
					extra += 1
	return extra

def incorrect_direction_edges(x,y):
	"""
	Counts directed edges in the learned structure
	that are oriented incorrectly. (lower is better)
	"""
	incorrect_direction = 0
	for rv in x.nodes():
		if x.has_node(rv):
			for child in x.children(rv): # child is rv's child in x
				if rv in y.parents(rv): # child is rv's parent in y
					incorrect_direction += 1
	return incorrect_direction

def hamming(x,y):
	return missing_edges(x,y) + extra_edges(x,y) + incorrect_direction_edges(x,y)
	








