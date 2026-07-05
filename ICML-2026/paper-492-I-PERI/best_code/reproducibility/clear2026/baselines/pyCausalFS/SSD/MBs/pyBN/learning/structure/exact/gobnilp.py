"""
GOBNILP Solver for Exact Bayesian network
Structure Learning with Integer Linear
Optimization.

This code relies on the "pyGOBN" package, which
implements a Python wrapper for the GOBNILP
project - which itself builds on the SCIP
Optimization engine for building Integer Linear
Programs to solve the BN structure learning problem
EXACTLY.

Th pyGOBN module - which must be installed from source - allows 
users to create/alter a vast number of global parameter settings 
(i.e. wall-time, parent limits, sparsity, equivalent 
sample size, etc), along with various network constraints 
(i.e. required edges, disallowed edges, and independence
requirements).

NOTE: You must download and install pyGOBN to use this functionality.
Visit github.com/ncullen93/pyGOBN to get the source code, then run
'python setup.py install' in the main pyGOBN directory to install the
module for your local python distribution.

"""
__author__ = """Nicholas Cullen <ncullen.th@dartmouth.edu>"""

def ilp(data, settings=None, edge_reqs=None, nonedge_reqs=None, ind_reqs=None):
	"""
	Learning the EXACT optimal Bayesian network structure from data using
	Integer Linear Programming from the GOBNILP project.
	"""
	try:
		from pyGOBN import GOBN
	except ImportError:
		print('You must download & install pyGOBN to use this functionality.\
		 Visit github.com/ncullen93/pyGOBN')

	gobn = GOBN()
	gobn.set_settings(settings)
	gobn.set_constraints(edge_reqs=edge_reqs, nonedge_reqs=nonedge_reqs, ind_reqs=ind_reqs)
