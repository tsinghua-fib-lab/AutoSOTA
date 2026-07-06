"""
Learn a Multi-Dimensional Bayesian Network Classifier
from data, and use it to predict a vector of class
labels from a vector of feature values.

An 'MBC' consists of two subgraphs:
	- G_c : the class subgraph,
		which includes edges only between class variables
	- G_x : the feature subgraph,
		which includes edges only between feature variables

Additionally, there is a collection of "bridge" edges:	
	- E_cx : the class-feature bridge,
		which includes the "bridge" edges between class and
		feature variables.

In an MBC, both the class sugraph and the feature subgraph can
have their own form of BN structure - e.g. empty, directed tree,
forest of trees, polytrees, or general DAGs. These 5 types of
network structures over two subgraphs means that there are 25
types of MBC structures.

To learn MBC's from data, the algorithm typically proceeds by
first learning G_c and G_x seperately using whatever structure
learning algorithm the user chooses, and then looks to optimize
the selection of "bridge" edges through further optimization
procedures.

To actually perform classification using an MBC, the traditional
Most Probable Explanation (MAP inference) procedure is employed.
The result is a vector of class predictions, which can then be
compared with the ground truth class labels to measure accuracy,
and so on.


References
----------
[1] Bielza, C., Li, G., Larranaga, P. "Multi-dimensional
classification with Bayesian networks."
[2] de Waal, P., van der Gaag, L. "Inference and Learning in
Multi-dimensional Bayesian Network Classifiers."
[3] van der Gaag, L., de Waal, P. "Multi-dimensional Bayesian
Network Classifiers."

"""

__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

from pyBN.learning.structure.score.hill_climbing import hc
from pyBN.learning.structure.score.random_restarts import hc_rr
from pyBN.learning.structure.score.tabu import tabu

from pyBN.classes.bayesnet import BayesNet


def bridge(c_bn, f_bn, data):
	"""
	Make a Multi-Dimensional Bayesian Network by
	bridging two Bayesian network structures. This happens by
	placing edges from c_bn -> f_bn using a heuristic 
	optimization procedure.

	This can be used to create a Multi-Dimensional Bayesian
	Network classifier from two already-learned Bayesian networks -
	one of which is a BN containing all the class variables, the other
	containing all the feature variables.

	Arguments
	---------
	*c_bn* : a BayesNet object with known structure

	*f_bn* : a BayesNet object with known structure.

	Returns
	-------
	*m_bn* : a merged/bridge BayesNet object,
		whose structure contains *c_bn*, *f_bn*, and some bridge
		edges between them.
	"""
	restrict = []
	for u in c_bn:
		for v in f_bn:
			restrict.append((u,v)) # only allow edges from c_bn -> f_bn

	bridge_bn = hc_rr(data, restriction=restrict)

	m_bn = bridge_bn.E
	m_bn.update(c_bn.E)
	m_bn.update(f_bn.E)

	mbc_bn = BayesNet(E=m_bn)

def mdbn(data, f_cols, c_cols, f_struct='DAG', c_struct='DAG', wrapper=False):
	"""
	Learn the structure of a Multi-Dimensional Bayesian Network - 
	typically used for Classification.

	Note that this structure does not have to be used for classification,
	since it simply returns a Bayesian Network - albeit with a more
	unqiue structure than tradiitonally found. If there are any other
	applications of this bipartite-like BN structure learning, this
	algorithm can certainly be used.

	"""
	f_data = data[:,f_cols]
	c_data = data[:,c_cols]

	f_bn = hc_rr(f_data)
	c_bn = hc_rr(c_data)

	mbc_bn = bridge(c_bn=c_bn, f_bn=f_bn, data=data)

	return mbc_bn






