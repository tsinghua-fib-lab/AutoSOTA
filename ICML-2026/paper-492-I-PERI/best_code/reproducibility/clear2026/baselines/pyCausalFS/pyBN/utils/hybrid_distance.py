"""
****************
BN Hybrid
Distance Metrics
****************

Code for computing the distance between 
2+ Bayesian networks that takes into account
both structure AND parameters. 

Any structure-based distance metric can be
combined with any parameter-based distance metric
to compute the hybrid distance.

Also, we support a type of supervised approach to
hybrid distance measurement where the relative importance
placed on the structure and distance metrics are varied
such that maximum (or minimum) closeness is achieved.

"""

def hybrid_distance(x,y, alpha, s='hamming', p='euclidean'):
	structure_score = hamming(x,y)
	parameter_score = euclidean(x,y)

	hybrid_score = structure_score*alpha + parameter_score*(1-alpha)

	return hybrid_score
